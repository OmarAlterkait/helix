"""Back-compat shim — TPC group ops moved to helix.tpc.coherent_ops_numpy.

The wavelet helper functions (per-wire DWT/threshold) are now subsumed by
helix.core.wavelet; small standalone equivalents are kept here so legacy
scripts that imported them still resolve.
"""
import numpy as np
import pywt

from helix.tpc.coherent_ops_numpy import (
    group_median, broadcast_groups, signal_mask, temporal_dilate,
    masked_group_mean, mad_sigma_per_wire, xblock_kernel,
)

__all__ = [
    "group_median", "broadcast_groups", "signal_mask", "temporal_dilate",
    "masked_group_mean", "mad_sigma_per_wire", "xblock_kernel",
    "per_wire_dwt", "per_wire_idwt", "estimate_subband_sigma", "hard_threshold",
]


def per_wire_dwt(image, wavelet, level):
    return pywt.wavedec(image, wavelet, level=level, axis=1)


def per_wire_idwt(coeffs, wavelet, n_time):
    return pywt.waverec(coeffs, wavelet, axis=1)[:, :n_time]


def estimate_subband_sigma(coeffs):
    return np.array([float(np.median(np.abs(c))) / 0.6745 for c in coeffs], dtype=np.float32)


def hard_threshold(coeffs, sigma_per_band, kappa, include_approx):
    out = []
    for i, c in enumerate(coeffs):
        if i == 0 and not include_approx:
            out.append(c.copy()); continue
        t = kappa * sigma_per_band[i] * np.sqrt(2.0 * np.log(max(c.shape[-1], 2)))
        out.append((c * (np.abs(c) >= t)).astype(np.float32))
    return out
