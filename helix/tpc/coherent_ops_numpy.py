"""NumPy/SciPy primitives for TPC coherent-noise removal (group/wire ops).

(The wavelet primitives that used to share _numpy_ops now live in helix.core.)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d


def group_median(image: np.ndarray, group_size: int) -> np.ndarray:
    """Per-group, per-tick median via partition (O(n)); last partial group handled separately."""
    nw, nt = image.shape
    n_full = nw // group_size
    rem = nw % group_size
    ng = n_full + (1 if rem > 0 else 0)
    out = np.empty((ng, nt), dtype=image.dtype)
    if n_full > 0:
        full = image[:n_full * group_size].reshape(n_full, group_size, nt)
        mid = group_size // 2
        p = np.partition(full, mid, axis=1)
        if group_size % 2 == 0:
            out[:n_full] = (p[:, mid - 1, :] + p[:, mid, :]) * 0.5
        else:
            out[:n_full] = p[:, mid, :]
    if rem > 0:
        last = image[n_full * group_size:]
        mid_r = rem // 2
        pr = np.partition(last, mid_r, axis=0)
        if rem % 2 == 0:
            out[n_full] = (pr[mid_r - 1] + pr[mid_r]) * 0.5
        else:
            out[n_full] = pr[mid_r]
    return out


def broadcast_groups(group_arr: np.ndarray, n_wires: int, group_size: int) -> np.ndarray:
    return group_arr[np.arange(n_wires) // group_size]


def signal_mask(residual: np.ndarray, sigma: np.ndarray, nsigma: float) -> np.ndarray:
    return np.abs(residual) > nsigma * sigma[:, None]


def temporal_dilate(mask: np.ndarray, ticks: int) -> np.ndarray:
    if ticks <= 1:
        return mask
    return maximum_filter1d(mask.view(np.uint8), size=ticks, axis=1).astype(bool)


def masked_group_mean(image: np.ndarray, mask: np.ndarray, group_size: int):
    """Mean of unflagged wires per (group, tick). Returns (estimate, n_unflagged)."""
    nw, nt = image.shape
    n_full = nw // group_size
    rem = nw % group_size
    ng = n_full + (1 if rem > 0 else 0)
    est = np.empty((ng, nt), dtype=np.float32)
    nuf = np.empty((ng, nt), dtype=np.float32)
    if n_full > 0:
        fi = image[:n_full * group_size].reshape(n_full, group_size, nt)
        fm = mask[:n_full * group_size].reshape(n_full, group_size, nt)
        unflag = ~fm
        nuf[:n_full] = unflag.sum(axis=1).astype(np.float32)
        est[:n_full] = (fi * unflag).sum(axis=1) / np.maximum(nuf[:n_full], 1.0)
    if rem > 0:
        li = image[n_full * group_size:]
        lm = mask[n_full * group_size:]
        lu = ~lm
        nuf[n_full] = lu.sum(axis=0).astype(np.float32)
        est[n_full] = (li * lu).sum(axis=0) / np.maximum(nuf[n_full], 1.0)
    return est, nuf


def mad_sigma_per_wire(residual: np.ndarray) -> np.ndarray:
    """MAD per-wire noise sigma via partition. Returns (n_wires,) float32."""
    abs_res = np.abs(residual)
    nt = abs_res.shape[1]
    mid = nt // 2
    p = np.partition(abs_res, mid, axis=1)
    if nt % 2 == 1:
        return (p[:, mid] / 0.6745).astype(np.float32)
    return ((p[:, mid - 1] + p[:, mid]) / (2.0 * 0.6745)).astype(np.float32)


def xblock_kernel(estimate: np.ndarray, beta: float) -> np.ndarray:
    """3-tap spatial high-pass across groups: (-beta, 1, -beta). Single allocation."""
    out = estimate.copy()
    out[0] -= beta * estimate[0]
    out[1:] -= beta * estimate[:-1]
    out[-1] -= beta * estimate[-1]
    out[:-1] -= beta * estimate[1:]
    return out
