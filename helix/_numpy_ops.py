"""NumPy/SciPy implementations of pipeline primitives."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d
import pywt


def group_median(image: np.ndarray, group_size: int) -> np.ndarray:
    """Per-group, per-tick median. Returns (n_groups, n_ticks).

    Uses ``np.partition`` (O(n) per group) instead of ``np.nanmedian``
    (O(n log n) with NaN bookkeeping).  The partial last group -- the only
    one that would contain NaN padding -- is handled separately on its
    actual (non-padded) wires so ``nanmedian`` is never needed.
    """
    nw, nt = image.shape
    n_full = nw // group_size
    rem = nw % group_size
    ng = n_full + (1 if rem > 0 else 0)
    out = np.empty((ng, nt), dtype=image.dtype)

    # Full groups: partition-based median (no NaN, no sort)
    if n_full > 0:
        full = image[:n_full * group_size].reshape(n_full, group_size, nt)
        mid = group_size // 2
        partitioned = np.partition(full, mid, axis=1)
        if group_size % 2 == 0:
            out[:n_full] = (partitioned[:, mid - 1, :] + partitioned[:, mid, :]) * 0.5
        else:
            out[:n_full] = partitioned[:, mid, :]

    # Partial last group: median of only the real wires (no padding needed)
    if rem > 0:
        last = image[n_full * group_size:]          # (rem, nt)
        mid_r = rem // 2
        partitioned_r = np.partition(last, mid_r, axis=0)
        if rem % 2 == 0:
            out[n_full] = (partitioned_r[mid_r - 1] + partitioned_r[mid_r]) * 0.5
        else:
            out[n_full] = partitioned_r[mid_r]

    return out


def broadcast_groups(group_arr: np.ndarray, n_wires: int, group_size: int) -> np.ndarray:
    """Broadcast (n_groups, n_ticks) -> (n_wires, n_ticks)."""
    wire_to_group = np.arange(n_wires) // group_size
    return group_arr[wire_to_group]


def signal_mask(residual: np.ndarray, sigma: np.ndarray, nsigma: float) -> np.ndarray:
    """Binary mask: True where |residual| > nsigma * sigma."""
    return np.abs(residual) > nsigma * sigma[:, None]


def temporal_dilate(mask: np.ndarray, ticks: int) -> np.ndarray:
    """Dilate mask along the time axis only.

    Uses ``maximum_filter1d`` which is ~4x faster than ``binary_dilation``
    for a 1-D structuring element applied along a single axis.
    """
    if ticks <= 1:
        return mask
    return maximum_filter1d(mask.view(np.uint8), size=ticks, axis=1).astype(bool)


def masked_group_mean(image: np.ndarray, mask: np.ndarray, group_size: int):
    """Mean of unflagged wires per (group, tick).

    Returns (estimate, n_unflagged) both shaped (n_groups, n_ticks).

    Avoids allocating a padded copy by handling the partial last group
    (the only one that would need padding) separately.
    """
    nw, nt = image.shape
    n_full = nw // group_size
    rem = nw % group_size
    ng = n_full + (1 if rem > 0 else 0)

    est = np.empty((ng, nt), dtype=np.float32)
    nuf = np.empty((ng, nt), dtype=np.float32)

    if n_full > 0:
        full_img = image[:n_full * group_size].reshape(n_full, group_size, nt)
        full_mask = mask[:n_full * group_size].reshape(n_full, group_size, nt)
        unflag = ~full_mask
        nuf[:n_full] = unflag.sum(axis=1).astype(np.float32)
        est[:n_full] = (full_img * unflag).sum(axis=1) / np.maximum(nuf[:n_full], 1.0)

    if rem > 0:
        last_img = image[n_full * group_size:]
        last_mask = mask[n_full * group_size:]
        last_unflag = ~last_mask
        nuf[n_full] = last_unflag.sum(axis=0).astype(np.float32)
        est[n_full] = (last_img * last_unflag).sum(axis=0) / np.maximum(nuf[n_full], 1.0)

    return est, nuf


def xblock_kernel(estimate: np.ndarray, beta: float) -> np.ndarray:
    """3-tap spatial high-pass filter across groups: (-beta, 1, -beta).

    Single allocation instead of two temporaries.
    """
    out = estimate.copy()
    out[0] -= beta * estimate[0]
    out[1:] -= beta * estimate[:-1]
    out[-1] -= beta * estimate[-1]
    out[:-1] -= beta * estimate[1:]
    return out


def mad_sigma_per_wire(residual: np.ndarray) -> np.ndarray:
    """MAD-based per-wire noise sigma via partition (O(n) per row).

    Returns (n_wires,) float32.  Equivalent to::

        sigma[w] = median(|residual[w]|) / 0.6745

    but ~6x faster than a Python loop calling ``np.median`` per wire.
    """
    abs_res = np.abs(residual)
    nt = abs_res.shape[1]
    mid = nt // 2
    p = np.partition(abs_res, mid, axis=1)
    if nt % 2 == 1:
        return (p[:, mid] / 0.6745).astype(np.float32)
    return ((p[:, mid - 1] + p[:, mid]) / (2.0 * 0.6745)).astype(np.float32)


def per_wire_dwt(image: np.ndarray, wavelet: str, level: int) -> list[np.ndarray]:
    """Per-wire 1D DWT. Returns list of coefficient arrays [cA, cD_L, ..., cD_1].

    Uses pywt's built-in ``axis`` parameter to process all wires in a single
    C-level call instead of looping over wires in Python.
    """
    return pywt.wavedec(image, wavelet, level=level, axis=1)


def per_wire_idwt(coeffs: list[np.ndarray], wavelet: str, n_time: int) -> np.ndarray:
    """Per-wire 1D inverse DWT. Returns (n_wires, n_time).

    Uses pywt's built-in ``axis`` parameter to process all wires in a single
    C-level call instead of looping over wires in Python.
    """
    return pywt.waverec(coeffs, wavelet, axis=1)[:, :n_time]


def _partition_median_flat(arr: np.ndarray) -> float:
    """Median of a flat array via ``np.partition`` (O(n) vs O(n log n) sort)."""
    n = arr.size
    mid = n // 2
    p = np.partition(arr, mid)
    if n % 2 == 0:
        return float((p[mid - 1] + p[mid]) * 0.5)
    return float(p[mid])


def estimate_subband_sigma(coeffs: list[np.ndarray]) -> np.ndarray:
    """MAD-based sigma estimate per subband. Returns array of length len(coeffs).

    Uses ``np.partition`` for O(n) median instead of full sort.
    """
    sigma = np.empty(len(coeffs), dtype=np.float32)
    for i, c in enumerate(coeffs):
        sigma[i] = _partition_median_flat(np.abs(c).ravel()) / 0.6745
    return sigma


def hard_threshold(coeffs: list[np.ndarray], sigma_per_band: np.ndarray,
                   kappa: float, include_approx: bool) -> list[np.ndarray]:
    """Donoho-Johnstone universal hard threshold per subband.

    Uses multiply-mask ``c * (|c| >= t)`` which avoids the ``np.where``
    three-array branch and is marginally faster.
    """
    out = []
    for i, c in enumerate(coeffs):
        if i == 0 and not include_approx:
            out.append(c.copy())
            continue
        n_j = c.shape[-1]
        t_j = kappa * sigma_per_band[i] * np.sqrt(2.0 * np.log(max(n_j, 2)))
        out.append((c * (np.abs(c) >= t_j)).astype(np.float32))
    return out
