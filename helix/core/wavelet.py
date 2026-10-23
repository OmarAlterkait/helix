"""The TPC chain's wavelet transform: per-row periodization DWT and its inverse (torch, any device).

Coefficient-identical to ``pywt.wavedec`` / ``pywt.waverec`` with ``mode="periodization"``
(verified to ~1e-14 in float64 across coif/sym/db/haar and lengths up to 36864):
  analysis:  cA = roll(circular_conv(x, dec_lo), -dec_len//2)[0::2]   (dec_hi -> cD)
  synthesis: upsample into [0::2], circular-correlate with the dec filters, roll by
             +dec_len//2, sum lowpass + highpass.
The signal length must be a multiple of 2^level (even at every level); with pywt's
``dwt_max_level`` cap that keeps every sub-band even and >= the filter length.

Coefficients: a list ``[cA, cD_L, ..., cD_1]`` of (n_signals, len_j) tensors. torch is
imported when a function runs, never at import (corpus readers and the tokenizer stay
torch-free).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np


@dataclass
class SparseResult:
    """One plane's thresholded band list and its per-band threshold sigma
    (:meth:`helix.core.coeff_event.CoeffEvent.from_sparse_results`)."""
    coeffs: Any
    n_kept: int
    n_total: int
    sigma_per_band: Any
    wavelet: str
    level: int
    mode: str


def _max_level(n, name):
    import pywt
    return pywt.dwt_max_level(n, pywt.Wavelet(name).dec_len)


@lru_cache(maxsize=32)
def _filters(name: str, device: str, dtype):
    """pywt-exact periodization uses the (non-reversed) decomposition filters for
    both analysis and synthesis (orthonormal: synthesis == adjoint of analysis)."""
    import pywt
    import torch
    w = pywt.Wavelet(name)
    t = lambda a: torch.tensor(np.asarray(a, np.float64), device=device, dtype=dtype)  # noqa: E731
    return t(w.dec_lo), t(w.dec_hi), w.dec_len


def _cfft(h, N):
    import torch
    hp = torch.zeros(N, device=h.device, dtype=h.dtype)
    hp[: h.numel()] = h
    return torch.fft.rfft(hp)


def _dwt1(x, dlo, dhi, L):
    """pywt periodization: cA = roll(circ_conv(x, dec_lo), -L//2)[0::2] (cD <- dec_hi),
    i.e. cA[k] = sum_n dec_lo[n] x[(2k + L//2 - n) mod N]: one stride-2 conv1d with both
    (flipped) filters over the circularly padded signal. Computes only the kept half,
    in O(N L) (L <= 18 here): ~10x faster than the full-length FFT convolution."""
    import torch
    import torch.nn.functional as F
    N = x.shape[-1]
    off = L // 2 - L + 1                                   # x index of filter tap L-1 at k = 0
    xr = x.reshape(-1, 1, N)
    if off or L > 2:
        xr = F.pad(xr, (-off, L - 2 + off), mode="circular")
    w = torch.stack([dlo.flip(0), dhi.flip(0)])[:, None, :]
    y = F.conv1d(xr, w, stride=2)
    shp = x.shape[:-1] + (N // 2,)
    return y[:, 0].reshape(shp), y[:, 1].reshape(shp)


def _idwt1(cA, cD, dlo, dhi, L):
    """Adjoint of _dwt1: upsample into [0::2], circular-correlate (conj in freq)
    with the dec filters, roll by +L//2, sum. Exact inverse for orthonormal."""
    import torch
    fft = torch.fft
    M = cA.shape[-1]
    N = 2 * M
    shp = cA.shape[:-1] + (N,)
    up_a = torch.zeros(shp, device=cA.device, dtype=cA.dtype); up_a[..., 0::2] = cA
    up_d = torch.zeros(shp, device=cA.device, dtype=cA.dtype); up_d[..., 0::2] = cD
    rec = fft.irfft(fft.rfft(up_a, dim=-1) * torch.conj(_cfft(dlo, N)), n=N, dim=-1) \
        + fft.irfft(fft.rfft(up_d, dim=-1) * torch.conj(_cfft(dhi, N)), n=N, dim=-1)
    return torch.roll(rec, L // 2, dims=-1)


def wavedec(image, wavelet: str = "coif3", level: int = 4, mode: str = "periodization"):
    """Forward DWT of ``(n_signals, n_ticks)`` -> ``([cA, cD_L, ..., cD_1], lev)``, ``lev`` the
    effective level (clamped to what the length allows, as pywt clamps it)."""
    import torch
    if mode != "periodization":
        raise ValueError(f"only 'periodization' is implemented (got {mode!r})")
    x = image if isinstance(image, torch.Tensor) else torch.as_tensor(np.asarray(image, np.float32))
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    lev = min(level, _max_level(x.shape[-1], wavelet))
    dlo, dhi, L = _filters(wavelet, str(x.device), x.dtype)
    coeffs, a = [], x
    for _ in range(lev):
        if a.shape[-1] % 2:
            raise ValueError(f"the periodization DWT needs an even length at every level (got {a.shape[-1]}); "
                             f"pad the signal to a multiple of 2^level")
        a, d = _dwt1(a, dlo, dhi, L)
        coeffs.append(d)
    coeffs.append(a)
    return coeffs[::-1], lev


def reconstruct(coeffs, wavelet: str, n_time: int):
    """Inverse DWT of a band list ``[cA, cD_L, ..., cD_1]`` (tensors or arrays) -> (n_signals, n_time)."""
    import torch
    coeffs = [c if isinstance(c, torch.Tensor) else torch.as_tensor(np.asarray(c, np.float32)) for c in coeffs]
    dlo, dhi, L = _filters(wavelet, str(coeffs[0].device), coeffs[0].dtype)
    a = coeffs[0]
    for d in coeffs[1:]:
        a = _idwt1(a[..., : d.shape[-1]], d, dlo, dhi, L)
    return a[..., :n_time]
