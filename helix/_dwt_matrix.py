"""Precomputed DWT/IDWT matrices for GPU matmul.

Builds the full transformation matrix from pywt's filter bank by applying
the DWT to each standard basis vector. Exact by construction — matches
pywt to float32 precision.

Usage:
    W_fwd, W_inv, band_slices = build_dwt_matrices('coif3', 4321, 4)
    coeffs_flat = image @ W_fwd     # (n_wires, n_coeffs)
    recon = coeffs_flat @ W_inv     # (n_wires, n_ticks)
"""

from __future__ import annotations

import numpy as np
import pywt
from functools import lru_cache


@lru_cache(maxsize=8)
def build_dwt_matrices(wavelet: str, n_ticks: int, level: int):
    """Build forward and inverse DWT matrices.

    Parameters
    ----------
    wavelet : str
        Wavelet name (e.g. 'coif3').
    n_ticks : int
        Signal length.
    level : int
        Decomposition level.

    Returns
    -------
    W_fwd : ndarray (n_ticks, n_coeffs), float32
        Forward DWT matrix. coeffs = signal @ W_fwd
    W_inv : ndarray (n_coeffs, n_ticks), float32
        Inverse DWT matrix. signal = coeffs @ W_inv
    band_slices : list of slice
        Slice for each band in the coefficient vector [cA, cD_L, ..., cD_1].
    """
    ref_coeffs = pywt.wavedec(np.zeros(n_ticks), wavelet, level=level)
    band_lengths = [len(c) for c in ref_coeffs]
    n_coeffs = sum(band_lengths)

    band_slices = []
    offset = 0
    for bl in band_lengths:
        band_slices.append(slice(offset, offset + bl))
        offset += bl

    # Build W_fwd: column i = DWT of unit vector e_i
    W_fwd = np.zeros((n_ticks, n_coeffs), dtype=np.float32)
    for i in range(n_ticks):
        e = np.zeros(n_ticks, dtype=np.float64)
        e[i] = 1.0
        coeffs = pywt.wavedec(e, wavelet, level=level)
        flat = np.concatenate(coeffs)
        W_fwd[i] = flat.astype(np.float32)

    # Build W_inv: row j = IDWT of unit coefficient e_j
    W_inv = np.zeros((n_coeffs, n_ticks), dtype=np.float32)
    for j in range(n_coeffs):
        unit_coeffs = []
        for s in band_slices:
            band = np.zeros(s.stop - s.start, dtype=np.float64)
            if s.start <= j < s.stop:
                band[j - s.start] = 1.0
            unit_coeffs.append(band)
        rec = pywt.waverec(unit_coeffs, wavelet)[:n_ticks]
        W_inv[j] = rec.astype(np.float32)

    return W_fwd, W_inv, band_slices
