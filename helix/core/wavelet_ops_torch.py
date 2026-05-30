"""PyTorch wavelet sparsification ops (GPU, batched, long-signal capable).

DWT/IDWT via FFT-based circular convolution -> exact 'periodization' boundary,
O(N log N), batched over signals, GPU-resident, and free of the N^2 dense-matrix
blow-up that rules matmul out for long (~36k-sample) optical chunks.

Recipe calibrated to machine-precision perfect reconstruction (see the project
notes): analysis convolves with reversed dec filters + downsample [0::2];
synthesis upsamples into [1::2], convolves with reversed rec filters, sums,
rolls by -dec_len. It is a self-consistent DWT (PR exact) but NOT coefficient-
identical to pywt's phase — compare sparsity within this backend.

Coefficient representation: list ``[cA, cD_L, …, cD_1]`` of (n_signals, len_j)
torch tensors (same layout as the numpy backend).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pywt
import torch
import torch.fft as _fft

from helix.core.wavelet import SparseResult, ThresholdSpec


@lru_cache(maxsize=32)
def _filters(name: str, device: str, dtype: torch.dtype):
    w = pywt.Wavelet(name)
    rev = lambda a: torch.tensor(np.asarray(a, np.float64)[::-1].copy(), device=device, dtype=dtype)
    return rev(w.dec_lo), rev(w.dec_hi), rev(w.rec_lo), rev(w.rec_hi), w.dec_len


def _cfft(h, N):
    hp = torch.zeros(N, device=h.device, dtype=h.dtype)
    hp[: h.numel()] = h
    return _fft.rfft(hp)


def _dwt1(x, dlo, dhi):
    N = x.shape[-1]
    Xf = _fft.rfft(x, dim=-1)
    lo = _fft.irfft(Xf * _cfft(dlo, N), n=N, dim=-1)
    hi = _fft.irfft(Xf * _cfft(dhi, N), n=N, dim=-1)
    return lo[..., 0::2], hi[..., 0::2]


def _idwt1(cA, cD, rlo, rhi, L):
    M = cA.shape[-1]
    N = 2 * M
    shp = cA.shape[:-1] + (N,)
    up_a = torch.zeros(shp, device=cA.device, dtype=cA.dtype); up_a[..., 1::2] = cA
    up_d = torch.zeros(shp, device=cA.device, dtype=cA.dtype); up_d[..., 1::2] = cD
    rec = _fft.irfft(_fft.rfft(up_a, dim=-1) * _cfft(rlo, N), n=N, dim=-1) \
        + _fft.irfft(_fft.rfft(up_d, dim=-1) * _cfft(rhi, N), n=N, dim=-1)
    return torch.roll(rec, -L, dims=-1)


def _safe_level(n, filt_len, requested):
    """Largest level <= requested where every sub-band stays >= filter length
    (and lengths remain even) so the FFT conv never underflows the filter."""
    lev, m = 0, n
    while lev < requested and m >= filt_len and m % 2 == 0:
        m //= 2
        lev += 1
    return max(lev, 1)


def _wavedec(x, name, level):
    dlo, dhi, _, _, dec_len = _filters(name, str(x.device), x.dtype)
    level = _safe_level(x.shape[-1], dec_len, level)
    coeffs, a = [], x
    for _ in range(level):
        a, d = _dwt1(a, dlo, dhi)
        coeffs.append(d)
    coeffs.append(a)
    return coeffs[::-1]


def _waverec(coeffs, name):
    _, _, rlo, rhi, L = _filters(name, str(coeffs[0].device), coeffs[0].dtype)
    a = coeffs[0]
    for d in coeffs[1:]:
        a = _idwt1(a[..., : d.shape[-1]], d, rlo, rhi, L)
    return a


def _max_level(n, name):
    return pywt.dwt_max_level(n, pywt.Wavelet(name).dec_len)


def _detail_threshold_per_signal(coeffs, frac, energy):
    det = torch.cat([c for c in coeffs[1:]], dim=-1).abs()   # (n_sig, D)
    D = det.shape[-1]
    if frac is not None:
        k = max(1, int(frac * D))
        return torch.topk(det, k, dim=-1).values[:, -1]      # (n_sig,)
    srt, _ = torch.sort(det, dim=-1, descending=True)
    csum = torch.cumsum(srt ** 2, dim=-1)
    tot = csum[:, -1:].clamp_min(1e-30)
    kc = (csum < energy * tot).sum(dim=-1).clamp(0, D - 1)
    return srt.gather(-1, kc[:, None]).squeeze(-1)


def _apply(c, t, func, torch):
    a = c.abs()
    if func == "soft":
        return torch.sign(c) * (a - t).clamp_min(0)
    if func == "garrote":
        return torch.where(a >= t, c - t * t / torch.where(a == 0, torch.ones_like(c), c), torch.zeros_like(c))
    return c * (a >= t)        # hard


def sparsify(image, wavelet: str, level: int, mode: str, th: ThresholdSpec, sigma=None) -> SparseResult:
    x = image if isinstance(image, torch.Tensor) else torch.as_tensor(np.asarray(image, np.float32))
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    lev = min(level, _max_level(x.shape[-1], wavelet))
    coeffs = _wavedec(x, wavelet, lev)
    band_sigma = torch.stack([c.abs().median() / 0.6745 for c in coeffs])   # per-band (reporting)
    if sigma is not None:                                                    # per-signal (thresholding)
        nsig = sigma if isinstance(sigma, torch.Tensor) else torch.as_tensor(np.asarray(sigma, np.float32),
                                                                             device=x.device)
    else:
        nsig = coeffs[-1].abs().median(dim=-1).values / 0.6745

    out = [coeffs[0]]
    if th.method == "universal":
        for c in coeffs[1:]:
            t = th.scale * nsig[..., None] * float(np.sqrt(2.0 * np.log(max(c.shape[-1], 2))))
            out.append(_apply(c, t, th.func, torch))
    else:
        tvec = _detail_threshold_per_signal(
            coeffs, th.keep if th.method == "topk" else None,
            th.energy if th.method == "energy" else None)[:, None]
        for c in coeffs[1:]:
            out.append(c * (c.abs() >= tvec))

    n_kept = int(sum(int(torch.count_nonzero(c)) for c in out))
    n_total = int(sum(c.numel() for c in out))
    return SparseResult(coeffs=out, n_kept=n_kept, n_total=n_total,
                        sigma_per_band=band_sigma.cpu().numpy(),
                        wavelet=wavelet, level=lev, mode=mode)


def reconstruct(coeffs, wavelet: str, level: int, mode: str, n_time: int):
    return _waverec(coeffs, wavelet)[..., :n_time]
