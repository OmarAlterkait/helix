"""NumPy/SciPy implementations of pipeline primitives."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
import pywt


def group_median(image: np.ndarray, group_size: int) -> np.ndarray:
    """Per-group, per-tick median. Returns (n_groups, n_ticks)."""
    nw, nt = image.shape
    ng = (nw + group_size - 1) // group_size
    pad = ng * group_size - nw
    if pad > 0:
        padded = np.concatenate([image, np.full((pad, nt), np.nan, dtype=image.dtype)], axis=0)
    else:
        padded = image
    return np.nanmedian(padded.reshape(ng, group_size, nt), axis=1)


def broadcast_groups(group_arr: np.ndarray, n_wires: int, group_size: int) -> np.ndarray:
    """Broadcast (n_groups, n_ticks) → (n_wires, n_ticks)."""
    ng = group_arr.shape[0]
    return np.repeat(group_arr[:, None, :], group_size, axis=1).reshape(ng * group_size, group_arr.shape[1])[:n_wires]


def signal_mask(residual: np.ndarray, sigma: np.ndarray, nsigma: float) -> np.ndarray:
    """Binary mask: True where |residual| > nsigma * sigma."""
    return np.abs(residual) > nsigma * sigma[:, None]


def temporal_dilate(mask: np.ndarray, ticks: int) -> np.ndarray:
    """Dilate mask along the time axis only."""
    if ticks <= 1:
        return mask
    struct = np.ones((1, ticks), dtype=bool)
    return ndimage.binary_dilation(mask, structure=struct)


def masked_group_mean(image: np.ndarray, mask: np.ndarray, group_size: int):
    """Mean of unflagged wires per (group, tick).

    Returns (estimate, n_unflagged) both shaped (n_groups, n_ticks).
    """
    nw, nt = image.shape
    ng = (nw + group_size - 1) // group_size
    pad = ng * group_size - nw
    if pad > 0:
        img_p = np.concatenate([image, np.zeros((pad, nt), dtype=image.dtype)], axis=0)
        mask_p = np.concatenate([mask, np.ones((pad, nt), dtype=bool)], axis=0)
    else:
        img_p = image
        mask_p = mask
    unflag = ~mask_p.reshape(ng, group_size, nt)
    nuf = unflag.sum(axis=1).astype(np.float32)
    est = (img_p.reshape(ng, group_size, nt) * unflag).sum(axis=1) / np.maximum(nuf, 1.0)
    return est, nuf


def xblock_kernel(estimate: np.ndarray, beta: float) -> np.ndarray:
    """3-tap spatial high-pass filter across groups: (-β, 1, -β)."""
    prev = np.empty_like(estimate)
    prev[0] = estimate[0]
    prev[1:] = estimate[:-1]
    nxt = np.empty_like(estimate)
    nxt[-1] = estimate[-1]
    nxt[:-1] = estimate[1:]
    return -beta * prev + estimate - beta * nxt


def per_wire_dwt(image: np.ndarray, wavelet: str, level: int) -> list[np.ndarray]:
    """Per-wire 1D DWT. Returns list of coefficient arrays [cA, cD_L, ..., cD_1]."""
    nw, nt = image.shape
    coeffs_list = pywt.wavedec(image[0], wavelet, level=level)
    out = [np.empty((nw, c.shape[0]), dtype=np.float32) for c in coeffs_list]
    for i, c in enumerate(coeffs_list):
        out[i][0] = c
    for w in range(1, nw):
        coeffs_list = pywt.wavedec(image[w], wavelet, level=level)
        for i, c in enumerate(coeffs_list):
            out[i][w] = c
    return out


def per_wire_idwt(coeffs: list[np.ndarray], wavelet: str, n_time: int) -> np.ndarray:
    """Per-wire 1D inverse DWT. Returns (n_wires, n_time)."""
    nw = coeffs[0].shape[0]
    out = np.empty((nw, n_time), dtype=np.float32)
    for w in range(nw):
        single = [c[w] for c in coeffs]
        rec = pywt.waverec(single, wavelet)[:n_time]
        out[w] = rec
    return out


def estimate_subband_sigma(coeffs: list[np.ndarray]) -> np.ndarray:
    """MAD-based sigma estimate per subband. Returns array of length len(coeffs)."""
    sigma = np.empty(len(coeffs), dtype=np.float32)
    for i, c in enumerate(coeffs):
        sigma[i] = float(np.median(np.abs(c))) / 0.6745
    return sigma


def hard_threshold(coeffs: list[np.ndarray], sigma_per_band: np.ndarray,
                   kappa: float, include_approx: bool) -> list[np.ndarray]:
    """Donoho-Johnstone universal hard threshold per subband."""
    out = []
    for i, c in enumerate(coeffs):
        if i == 0 and not include_approx:
            out.append(c.copy())
            continue
        n_j = c.shape[-1]
        t_j = kappa * sigma_per_band[i] * np.sqrt(2.0 * np.log(max(n_j, 2)))
        out.append(np.where(np.abs(c) >= t_j, c, 0.0).astype(np.float32))
    return out
