"""Precomputed DWT/IDWT matrices for matmul-based transforms (jax/torch).

Builds the full transform matrix from pywt's filter bank by applying the DWT to
each standard basis vector. Exact by construction — matches pywt to float32
precision. Built in numpy/pywt once, then converted to the backend array type.

    W_fwd, W_inv, band_slices = build_dwt_matrices('coif3', 4321, 4)
    coeffs = image @ W_fwd      # (n_signals, n_coeffs)
    recon  = coeffs @ W_inv     # (n_signals, n_ticks)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pywt


@lru_cache(maxsize=16)
def build_dwt_matrices(wavelet: str, n_ticks: int, level: int, mode: str = "periodization"):
    """Return (W_fwd (n_ticks, n_coeffs), W_inv (n_coeffs, n_ticks), band_slices)."""
    ref_coeffs = pywt.wavedec(np.zeros(n_ticks), wavelet, level=level, mode=mode)
    band_lengths = [len(c) for c in ref_coeffs]
    n_coeffs = sum(band_lengths)

    band_slices = []
    offset = 0
    for bl in band_lengths:
        band_slices.append(slice(offset, offset + bl))
        offset += bl

    W_fwd = np.zeros((n_ticks, n_coeffs), dtype=np.float32)
    for i in range(n_ticks):
        e = np.zeros(n_ticks, dtype=np.float64)
        e[i] = 1.0
        flat = np.concatenate(pywt.wavedec(e, wavelet, level=level, mode=mode))
        W_fwd[i] = flat.astype(np.float32)

    W_inv = np.zeros((n_coeffs, n_ticks), dtype=np.float32)
    for j in range(n_coeffs):
        unit_coeffs = []
        for s in band_slices:
            band = np.zeros(s.stop - s.start, dtype=np.float64)
            if s.start <= j < s.stop:
                band[j - s.start] = 1.0
            unit_coeffs.append(band)
        rec = pywt.waverec(unit_coeffs, wavelet, mode=mode)[:n_ticks]
        W_inv[j] = rec.astype(np.float32)

    return W_fwd, W_inv, band_slices
