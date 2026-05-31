"""NumPy/pywt wavelet sparsification ops (CPU reference backend).

Coefficient representation: a list ``[cA, cD_L, …, cD_1]`` of (n_signals, len_j)
arrays, exactly as returned by ``pywt.wavedec(..., axis=1)``.

`universal` is VisuShrink: a per-signal noise sigma (MAD of the finest detail
band, or caller-supplied) applied to all detail bands as t = scale·σ·sqrt(2 ln N).
"""
from __future__ import annotations

import numpy as np
import pywt

from helix.core.wavelet import SparseResult, ThresholdSpec


def _mad_sigma(c: np.ndarray) -> float:
    return float(np.median(np.abs(c))) / 0.6745


def _noise_sigma(coeffs, sigma):
    """Per-signal noise sigma: caller-supplied, else MAD of the finest detail band."""
    if sigma is not None:
        return np.asarray(sigma, dtype=np.float32)
    return (np.median(np.abs(coeffs[-1]), axis=-1) / 0.6745).astype(np.float32)


def _apply(c, t, func):
    a = np.abs(c)
    if func == "soft":
        return (np.sign(c) * np.maximum(a - t, 0.0)).astype(np.float32)
    if func == "garrote":
        return np.where(a >= t, c - t * t / np.where(a == 0, 1.0, c), 0.0).astype(np.float32)
    return (c * (a >= t)).astype(np.float32)            # hard


def _detail_threshold_per_signal(coeffs, frac, energy):
    det = np.concatenate([c for c in coeffs[1:]], axis=-1)
    a = np.abs(det); D = a.shape[-1]
    if frac is not None:
        k = max(1, int(frac * D))
        return np.partition(a, D - k, axis=-1)[..., D - k]
    srt = np.sort(a, axis=-1)[..., ::-1]
    csum = np.cumsum(srt ** 2, axis=-1)
    tot = np.maximum(csum[..., -1:], 1e-30)
    kc = np.clip((csum < energy * tot).sum(axis=-1), 0, D - 1)
    return np.take_along_axis(srt, kc[..., None], axis=-1)[..., 0]


def sparsify(image, wavelet: str, level: int, mode: str, th: ThresholdSpec, sigma=None) -> SparseResult:
    img = np.asarray(image, dtype=np.float32)
    lev = min(level, pywt.dwt_max_level(img.shape[-1], pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedec(img, wavelet, level=lev, mode=mode, axis=-1)
    band_sigma = np.array([_mad_sigma(c) for c in coeffs], dtype=np.float32)   # per-band (reporting)
    nsig = _noise_sigma(coeffs, sigma)                                          # per-signal (thresholding)

    if th.method == "universal":
        out = []
        for i, c in enumerate(coeffs):
            if i == 0 and not th.threshold_approx:        # keep approx untouched (default / optical)
                out.append(coeffs[0]); continue
            lf = np.sqrt(2.0 * np.log(max(c.shape[-1], 2)))
            if th.per_band_sigma:                          # per-band MAD sigma (TPC / original)
                t = th.scale * band_sigma[i] * lf
            else:                                          # single per-signal sigma (optical)
                t = th.scale * nsig[..., None] * lf
            out.append(_apply(c, t, th.func))
    else:  # topk / energy (approx kept untouched)
        out = [coeffs[0]]
        tvec = _detail_threshold_per_signal(
            coeffs, th.keep if th.method == "topk" else None,
            th.energy if th.method == "energy" else None)[..., None]
        for c in coeffs[1:]:
            out.append((c * (np.abs(c) >= tvec)).astype(np.float32))

    n_kept = sum(int(np.count_nonzero(c)) for c in out)
    n_total = sum(c.size for c in out)
    return SparseResult(coeffs=out, n_kept=n_kept, n_total=n_total,
                        sigma_per_band=band_sigma, wavelet=wavelet, level=lev, mode=mode)


def reconstruct(coeffs, wavelet: str, level: int, mode: str, n_time: int):
    return pywt.waverec(coeffs, wavelet, mode=mode, axis=-1)[..., :n_time]
