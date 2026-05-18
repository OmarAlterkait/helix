"""Benchmark current vs optimized _numpy_ops implementations.

Generates a realistic-sized test plane (1231 x 2701), times each function
in both its current and optimized form, verifies outputs match, and reports
speedup per function and total pipeline speedup.

Usage:
    python3 tests/bench_numpy.py
"""

from __future__ import annotations

import time
import numpy as np
import pywt
from scipy.ndimage import binary_dilation, maximum_filter1d


# ---------------------------------------------------------------------------
# Parameters matching a realistic U-plane
# ---------------------------------------------------------------------------
NW = 1231
NT = 2701
GS = 64  # group_size
WAVELET = "coif3"
DWT_LEVEL = 4
KAPPA = 1.0
BETA = 0.15
NSIGMA = 3.0
DILATION_TICKS = 9
INCLUDE_APPROX = True
N_WARMUP = 2
N_ITER = 10


# ---------------------------------------------------------------------------
# Current implementations (copied from _numpy_ops.py for side-by-side)
# ---------------------------------------------------------------------------

def group_median_current(image, group_size):
    nw, nt = image.shape
    ng = (nw + group_size - 1) // group_size
    pad = ng * group_size - nw
    if pad > 0:
        padded = np.concatenate(
            [image, np.full((pad, nt), np.nan, dtype=image.dtype)], axis=0
        )
    else:
        padded = image
    return np.nanmedian(padded.reshape(ng, group_size, nt), axis=1)


def broadcast_groups_current(group_arr, n_wires, group_size):
    ng = group_arr.shape[0]
    return np.repeat(group_arr[:, None, :], group_size, axis=1).reshape(
        ng * group_size, group_arr.shape[1]
    )[:n_wires]


def signal_mask_current(residual, sigma, nsigma):
    return np.abs(residual) > nsigma * sigma[:, None]


def temporal_dilate_current(mask, ticks):
    if ticks <= 1:
        return mask
    struct = np.ones((1, ticks), dtype=bool)
    return ndimage.binary_dilation(mask, structure=struct)


# need the import for current
from scipy import ndimage


def masked_group_mean_current(image, mask, group_size):
    nw, nt = image.shape
    ng = (nw + group_size - 1) // group_size
    pad = ng * group_size - nw
    if pad > 0:
        img_p = np.concatenate(
            [image, np.zeros((pad, nt), dtype=image.dtype)], axis=0
        )
        mask_p = np.concatenate(
            [mask, np.ones((pad, nt), dtype=bool)], axis=0
        )
    else:
        img_p = image
        mask_p = mask
    unflag = ~mask_p.reshape(ng, group_size, nt)
    nuf = unflag.sum(axis=1).astype(np.float32)
    est = (img_p.reshape(ng, group_size, nt) * unflag).sum(axis=1) / np.maximum(
        nuf, 1.0
    )
    return est, nuf


def xblock_kernel_current(estimate, beta):
    prev = np.empty_like(estimate)
    prev[0] = estimate[0]
    prev[1:] = estimate[:-1]
    nxt = np.empty_like(estimate)
    nxt[-1] = estimate[-1]
    nxt[:-1] = estimate[1:]
    return -beta * prev + estimate - beta * nxt


def per_wire_dwt_current(image, wavelet, level):
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


def per_wire_idwt_current(coeffs, wavelet, n_time):
    nw = coeffs[0].shape[0]
    out = np.empty((nw, n_time), dtype=np.float32)
    for w in range(nw):
        single = [c[w] for c in coeffs]
        rec = pywt.waverec(single, wavelet)[:n_time]
        out[w] = rec
    return out


def estimate_subband_sigma_current(coeffs):
    sigma = np.empty(len(coeffs), dtype=np.float32)
    for i, c in enumerate(coeffs):
        sigma[i] = float(np.median(np.abs(c))) / 0.6745
    return sigma


def hard_threshold_current(coeffs, sigma_per_band, kappa, include_approx):
    out = []
    for i, c in enumerate(coeffs):
        if i == 0 and not include_approx:
            out.append(c.copy())
            continue
        n_j = c.shape[-1]
        t_j = kappa * sigma_per_band[i] * np.sqrt(2.0 * np.log(max(n_j, 2)))
        out.append(np.where(np.abs(c) >= t_j, c, 0.0).astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Optimized implementations
# ---------------------------------------------------------------------------

def group_median_opt(image, group_size):
    """Partition-based median; only nanmedian for the partial last group."""
    nw, nt = image.shape
    n_full = nw // group_size
    rem = nw % group_size
    ng = n_full + (1 if rem > 0 else 0)
    out = np.empty((ng, nt), dtype=image.dtype)

    if n_full > 0:
        full = image[: n_full * group_size].reshape(n_full, group_size, nt)
        mid = group_size // 2
        partitioned = np.partition(full, mid, axis=1)
        if group_size % 2 == 0:
            out[:n_full] = (partitioned[:, mid - 1, :] + partitioned[:, mid, :]) * 0.5
        else:
            out[:n_full] = partitioned[:, mid, :]

    if rem > 0:
        last = image[n_full * group_size :]
        mid_r = rem // 2
        partitioned_r = np.partition(last, mid_r, axis=0)
        if rem % 2 == 0:
            out[n_full] = (partitioned_r[mid_r - 1] + partitioned_r[mid_r]) * 0.5
        else:
            out[n_full] = partitioned_r[mid_r]

    return out


def broadcast_groups_opt(group_arr, n_wires, group_size):
    """Index-based broadcast: avoids repeat + reshape allocation."""
    wire_to_group = np.arange(n_wires) // group_size
    return group_arr[wire_to_group]


def temporal_dilate_opt(mask, ticks):
    """maximum_filter1d is faster than binary_dilation for 1D structuring."""
    if ticks <= 1:
        return mask
    return maximum_filter1d(mask.view(np.uint8), size=ticks, axis=1).astype(bool)


def masked_group_mean_opt(image, mask, group_size):
    """Avoid padding by handling the last partial group separately."""
    nw, nt = image.shape
    n_full = nw // group_size
    rem = nw % group_size
    ng = n_full + (1 if rem > 0 else 0)

    est = np.empty((ng, nt), dtype=np.float32)
    nuf = np.empty((ng, nt), dtype=np.float32)

    if n_full > 0:
        full_img = image[: n_full * group_size].reshape(n_full, group_size, nt)
        full_mask = mask[: n_full * group_size].reshape(n_full, group_size, nt)
        unflag = ~full_mask
        nuf[:n_full] = unflag.sum(axis=1).astype(np.float32)
        est[:n_full] = (full_img * unflag).sum(axis=1) / np.maximum(
            nuf[:n_full], 1.0
        )

    if rem > 0:
        last_img = image[n_full * group_size :]
        last_mask = mask[n_full * group_size :]
        last_unflag = ~last_mask
        nuf[n_full] = last_unflag.sum(axis=0).astype(np.float32)
        est[n_full] = (last_img * last_unflag).sum(axis=0) / np.maximum(
            nuf[n_full], 1.0
        )

    return est, nuf


def xblock_kernel_opt(estimate, beta):
    """Single-allocation: in-place subtract neighbors."""
    out = estimate.copy()
    out[0] -= beta * estimate[0]
    out[1:] -= beta * estimate[:-1]
    out[-1] -= beta * estimate[-1]
    out[:-1] -= beta * estimate[1:]
    return out


def per_wire_dwt_opt(image, wavelet, level):
    """Batch DWT using axis=1 -- no Python loop over wires."""
    return pywt.wavedec(image, wavelet, level=level, axis=1)


def per_wire_idwt_opt(coeffs, wavelet, n_time):
    """Batch IDWT using axis=1 -- no Python loop over wires."""
    return pywt.waverec(coeffs, wavelet, axis=1)[:, :n_time]


def _partition_median_flat(arr):
    """Median of a flat array via partition (avoids full sort)."""
    n = arr.size
    mid = n // 2
    p = np.partition(arr, mid)
    if n % 2 == 0:
        return (p[mid - 1] + p[mid]) * 0.5
    return float(p[mid])


def estimate_subband_sigma_opt(coeffs):
    """Partition-based median(|c|) per subband."""
    sigma = np.empty(len(coeffs), dtype=np.float32)
    for i, c in enumerate(coeffs):
        sigma[i] = _partition_median_flat(np.abs(c).ravel()) / 0.6745
    return sigma


def hard_threshold_opt(coeffs, sigma_per_band, kappa, include_approx):
    """Multiply-mask: c * (|c| >= t_j) avoids np.where branch overhead."""
    out = []
    for i, c in enumerate(coeffs):
        if i == 0 and not include_approx:
            out.append(c.copy())
            continue
        n_j = c.shape[-1]
        t_j = kappa * sigma_per_band[i] * np.sqrt(2.0 * np.log(max(n_j, 2)))
        out.append((c * (np.abs(c) >= t_j)).astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def bench(name, fn_current, fn_opt, args_current, args_opt=None, compare_fn=None):
    """Time current vs optimized, verify correctness, report speedup."""
    if args_opt is None:
        args_opt = args_current
    if compare_fn is None:
        compare_fn = default_compare

    # Warmup
    for _ in range(N_WARMUP):
        r_cur = fn_current(*args_current)
        r_opt = fn_opt(*args_opt)

    # Time current
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        r_cur = fn_current(*args_current)
    t_cur = (time.perf_counter() - t0) / N_ITER

    # Time optimized
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        r_opt = fn_opt(*args_opt)
    t_opt = (time.perf_counter() - t0) / N_ITER

    # Compare
    max_diff = compare_fn(r_cur, r_opt)
    speedup = t_cur / t_opt if t_opt > 0 else float("inf")

    print(
        f"  {name:30s}  current={t_cur*1000:8.2f} ms  opt={t_opt*1000:8.2f} ms  "
        f"speedup={speedup:5.2f}x  max_diff={max_diff:.2e}"
    )
    return t_cur, t_opt


def default_compare(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        diffs = []
        for x, y in zip(a, b):
            diffs.append(
                float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
            )
        return max(diffs)
    return 0.0


def compare_tuple(a, b):
    """For (est, nuf) tuple returns."""
    diffs = []
    for x, y in zip(a, b):
        diffs.append(
            float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        )
    return max(diffs)


def main():
    rng = np.random.default_rng(42)
    image = rng.standard_normal((NW, NT)).astype(np.float32)

    print(f"\nBenchmark: _numpy_ops optimizations")
    print(f"Plane size: {NW} wires x {NT} ticks = {NW*NT:,} pixels")
    print(f"Iterations: {N_ITER} (+ {N_WARMUP} warmup)\n")

    total_cur = 0.0
    total_opt = 0.0

    # --- group_median ---
    tc, to = bench(
        "group_median",
        group_median_current,
        group_median_opt,
        (image, GS),
    )
    total_cur += tc
    total_opt += to

    # --- broadcast_groups ---
    gm = group_median_opt(image, GS)
    tc, to = bench(
        "broadcast_groups",
        broadcast_groups_current,
        broadcast_groups_opt,
        (gm, NW, GS),
    )
    total_cur += tc
    total_opt += to

    # --- mad_sigma_per_wire ---
    gm_full = broadcast_groups_opt(gm, NW, GS)
    residual = image - gm_full

    def mad_sigma_loop(residual):
        nw = residual.shape[0]
        sigma = np.empty(nw, dtype=np.float32)
        for w in range(nw):
            sigma[w] = float(np.median(np.abs(residual[w]))) / 0.6745
        return sigma

    def mad_sigma_partition(residual):
        abs_res = np.abs(residual)
        nt = abs_res.shape[1]
        mid = nt // 2
        p = np.partition(abs_res, mid, axis=1)
        if nt % 2 == 1:
            return (p[:, mid] / 0.6745).astype(np.float32)
        return ((p[:, mid - 1] + p[:, mid]) / (2.0 * 0.6745)).astype(np.float32)

    tc, to = bench(
        "mad_sigma_per_wire",
        mad_sigma_loop,
        mad_sigma_partition,
        (residual,),
    )
    total_cur += tc
    total_opt += to

    sigma = mad_sigma_partition(residual)

    # --- signal_mask --- (same function, just for total timing)
    tc, to = bench(
        "signal_mask (unchanged)",
        signal_mask_current,
        signal_mask_current,  # same
        (residual, sigma, NSIGMA),
    )
    total_cur += tc
    total_opt += to

    # --- temporal_dilate ---
    mask = signal_mask_current(residual, sigma, NSIGMA)
    tc, to = bench(
        "temporal_dilate",
        temporal_dilate_current,
        temporal_dilate_opt,
        (mask, DILATION_TICKS),
    )
    total_cur += tc
    total_opt += to

    # --- masked_group_mean ---
    mask_d = temporal_dilate_opt(mask, DILATION_TICKS)
    tc, to = bench(
        "masked_group_mean",
        masked_group_mean_current,
        masked_group_mean_opt,
        (image, mask_d, GS),
        compare_fn=compare_tuple,
    )
    total_cur += tc
    total_opt += to

    # --- xblock_kernel ---
    est, nuf = masked_group_mean_opt(image, mask_d, GS)
    tc, to = bench(
        "xblock_kernel",
        xblock_kernel_current,
        xblock_kernel_opt,
        (est, BETA),
    )
    total_cur += tc
    total_opt += to

    # --- per_wire_dwt ---
    tc, to = bench(
        "per_wire_dwt",
        per_wire_dwt_current,
        per_wire_dwt_opt,
        (image, WAVELET, DWT_LEVEL),
    )
    total_cur += tc
    total_opt += to

    # --- per_wire_idwt ---
    coeffs_cur = per_wire_dwt_current(image, WAVELET, DWT_LEVEL)
    coeffs_opt = per_wire_dwt_opt(image, WAVELET, DWT_LEVEL)
    tc, to = bench(
        "per_wire_idwt",
        per_wire_idwt_current,
        per_wire_idwt_opt,
        args_current=(coeffs_cur, WAVELET, NT),
        args_opt=(coeffs_opt, WAVELET, NT),
    )
    total_cur += tc
    total_opt += to

    # --- estimate_subband_sigma ---
    tc, to = bench(
        "estimate_subband_sigma",
        estimate_subband_sigma_current,
        estimate_subband_sigma_opt,
        (coeffs_opt,),
    )
    total_cur += tc
    total_opt += to

    # --- hard_threshold ---
    sigma_band = estimate_subband_sigma_opt(coeffs_opt)
    tc, to = bench(
        "hard_threshold",
        hard_threshold_current,
        hard_threshold_opt,
        (coeffs_opt, sigma_band, KAPPA, INCLUDE_APPROX),
    )
    total_cur += tc
    total_opt += to

    # --- Summary ---
    print(f"\n{'='*80}")
    print(
        f"  {'TOTAL':30s}  current={total_cur*1000:8.2f} ms  opt={total_opt*1000:8.2f} ms  "
        f"speedup={total_cur/total_opt:.2f}x"
    )
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
