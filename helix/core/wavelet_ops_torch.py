"""PyTorch wavelet sparsification ops (GPU, batched, long-signal capable).

DWT/IDWT via FFT-based circular convolution -> exact 'periodization' boundary,
O(N log N), batched over signals, GPU-resident, and free of the N^2 dense-matrix
blow-up that rules matmul out for long (~36k-sample) optical chunks.

**Coefficient-identical to pywt** (== the numpy and jax backends). The
periodization filter bank is matched to pywt's exact convention:
  analysis:  cA = roll(circular_conv(x, dec_lo), -dec_len//2)[0::2]   (dec_hi → cD)
  synthesis: upsample into [0::2], circular-correlate with the dec filters,
             roll by +dec_len//2, sum lowpass+highpass.
Verified against ``pywt.wavedec``/``pywt.waverec`` to ~1e-14 (float64) across
coif/sym/db/haar and lengths up to 36864. Requires the signal length to be a
multiple of 2^level (even at every level — the optical pipeline pads to this);
that guarantee, plus pywt's ``dwt_max_level`` cap, keeps every sub-band even and
>= filter length so the FFT conv reproduces pywt exactly.

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
    """pywt-exact periodization uses the (non-reversed) decomposition filters for
    both analysis and synthesis (orthonormal: synthesis == adjoint of analysis)."""
    w = pywt.Wavelet(name)
    t = lambda a: torch.tensor(np.asarray(a, np.float64), device=device, dtype=dtype)
    return t(w.dec_lo), t(w.dec_hi), w.dec_len


def _cfft(h, N):
    hp = torch.zeros(N, device=h.device, dtype=h.dtype)
    hp[: h.numel()] = h
    return _fft.rfft(hp)


def _dwt1(x, dlo, dhi, L):
    """pywt periodization: cA = roll(circ_conv(x, dec_lo), -L//2)[0::2] (cD ← dec_hi)."""
    N = x.shape[-1]
    Xf = _fft.rfft(x, dim=-1)
    lo = torch.roll(_fft.irfft(Xf * _cfft(dlo, N), n=N, dim=-1), -(L // 2), dims=-1)
    hi = torch.roll(_fft.irfft(Xf * _cfft(dhi, N), n=N, dim=-1), -(L // 2), dims=-1)
    return lo[..., 0::2], hi[..., 0::2]


def _idwt1(cA, cD, dlo, dhi, L):
    """Adjoint of _dwt1: upsample into [0::2], circular-correlate (conj in freq)
    with the dec filters, roll by +L//2, sum. Exact inverse for orthonormal."""
    M = cA.shape[-1]
    N = 2 * M
    shp = cA.shape[:-1] + (N,)
    up_a = torch.zeros(shp, device=cA.device, dtype=cA.dtype); up_a[..., 0::2] = cA
    up_d = torch.zeros(shp, device=cA.device, dtype=cA.dtype); up_d[..., 0::2] = cD
    rec = _fft.irfft(_fft.rfft(up_a, dim=-1) * torch.conj(_cfft(dlo, N)), n=N, dim=-1) \
        + _fft.irfft(_fft.rfft(up_d, dim=-1) * torch.conj(_cfft(dhi, N)), n=N, dim=-1)
    return torch.roll(rec, L // 2, dims=-1)


def _wavedec(x, name, level):
    dlo, dhi, dec_len = _filters(name, str(x.device), x.dtype)
    level = min(level, _max_level(x.shape[-1], name))      # same cap as pywt/numpy
    coeffs, a = [], x
    for _ in range(level):
        if a.shape[-1] % 2:
            raise ValueError(
                f"torch periodization DWT needs an even length at every level "
                f"(got {a.shape[-1]}); pad the signal to a multiple of 2^level.")
        a, d = _dwt1(a, dlo, dhi, dec_len)
        coeffs.append(d)
    coeffs.append(a)
    return coeffs[::-1]


def _waverec(coeffs, name):
    dlo, dhi, L = _filters(name, str(coeffs[0].device), coeffs[0].dtype)
    a = coeffs[0]
    for d in coeffs[1:]:
        a = _idwt1(a[..., : d.shape[-1]], d, dlo, dhi, L)
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
