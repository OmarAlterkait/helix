"""JAX/GPU implementations of pipeline primitives."""

from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap


@partial(jit, static_argnums=(1,))
def group_median(image: jnp.ndarray, group_size: int) -> jnp.ndarray:
    ng = image.shape[0] // group_size
    nt = image.shape[1]
    blocks = image.reshape(ng, group_size, nt)
    return jnp.median(blocks, axis=1)


def broadcast_groups(group_arr: jnp.ndarray, n_wires: int, group_size: int) -> jnp.ndarray:
    ng = group_arr.shape[0]
    return jnp.repeat(group_arr[:, None, :], group_size, axis=1).reshape(ng * group_size, group_arr.shape[1])[:n_wires]


@jit
def signal_mask(residual: jnp.ndarray, sigma: jnp.ndarray, nsigma: float) -> jnp.ndarray:
    return jnp.abs(residual) > nsigma * sigma[:, None]


@partial(jit, static_argnums=(1,))
def temporal_dilate(mask: jnp.ndarray, ticks: int) -> jnp.ndarray:
    if ticks <= 1:
        return mask
    kernel = jnp.ones(ticks)
    dilated = vmap(lambda row: jnp.convolve(row, kernel, mode='same'))(mask.astype(jnp.float32))
    return dilated > 0.5


@partial(jit, static_argnums=(2,))
def masked_group_mean(image: jnp.ndarray, mask: jnp.ndarray, group_size: int):
    ng = image.shape[0] // group_size
    nt = image.shape[1]
    unflag = ~mask.reshape(ng, group_size, nt)
    nuf = unflag.sum(axis=1).astype(jnp.float32)
    est = (image.reshape(ng, group_size, nt) * unflag).sum(axis=1) / jnp.maximum(nuf, 1.0)
    return est, nuf


@jit
def xblock_kernel(estimate: jnp.ndarray, beta: float) -> jnp.ndarray:
    prev = jnp.concatenate([estimate[:1], estimate[:-1]], axis=0)
    nxt = jnp.concatenate([estimate[1:], estimate[-1:]], axis=0)
    return -beta * prev + estimate - beta * nxt


@jit
def hard_threshold_band(coeffs: jnp.ndarray, threshold: float) -> jnp.ndarray:
    return jnp.where(jnp.abs(coeffs) >= threshold, coeffs, 0.0)


def pad_to_groups(image: jnp.ndarray, group_size: int) -> jnp.ndarray:
    nw, nt = image.shape
    ng = (nw + group_size - 1) // group_size
    pad = ng * group_size - nw
    if pad > 0:
        return jnp.concatenate([image, jnp.zeros((pad, nt), dtype=image.dtype)], axis=0)
    return image


def pad_mask_to_groups(mask: jnp.ndarray, group_size: int) -> jnp.ndarray:
    nw, nt = mask.shape
    ng = (nw + group_size - 1) // group_size
    pad = ng * group_size - nw
    if pad > 0:
        return jnp.concatenate([mask, jnp.ones((pad, nt), dtype=bool)], axis=0)
    return mask


# ── DWT via matmul ────────────────────────────────────────────────────

_dwt_cache: dict = {}


def _get_dwt_matrices(wavelet: str, n_ticks: int, level: int):
    """Get or build cached DWT matrices on GPU."""
    key = (wavelet, n_ticks, level)
    if key not in _dwt_cache:
        from helix._dwt_matrix import build_dwt_matrices
        W_fwd_np, W_inv_np, slices = build_dwt_matrices(wavelet, n_ticks, level)
        _dwt_cache[key] = (
            jnp.array(W_fwd_np),
            jnp.array(W_inv_np),
            slices,
        )
    return _dwt_cache[key]


def dwt_matmul(image: jnp.ndarray, wavelet: str, level: int):
    """GPU DWT via matmul. Returns (flat_coeffs, band_slices)."""
    W_fwd, _, slices = _get_dwt_matrices(wavelet, image.shape[1], level)
    return image @ W_fwd, slices


def idwt_matmul(coeffs: jnp.ndarray, wavelet: str, n_ticks: int, level: int):
    """GPU IDWT via matmul."""
    _, W_inv, _ = _get_dwt_matrices(wavelet, n_ticks, level)
    return coeffs @ W_inv


def wavelet_pipeline_jax(image: jnp.ndarray, wavelet: str, level: int,
                         kappa: float, include_approx: bool):
    """Full GPU wavelet pipeline: DWT (matmul) → threshold → IDWT (matmul).

    Everything on GPU in a single JIT-compiled function.
    Returns (reconstructed, thresholded_coeffs, sigma_per_band, n_kept, n_total).
    """
    W_fwd, W_inv, slices = _get_dwt_matrices(wavelet, image.shape[1], level)
    return _wavelet_kernel(image, W_fwd, W_inv,
                           tuple((s.start, s.stop, i > 0 or include_approx)
                                 for i, s in enumerate(slices)),
                           kappa)


@partial(jit, static_argnums=(3, 4))
def _wavelet_kernel(image, W_fwd, W_inv, band_info, kappa):
    coeffs = image @ W_fwd
    thresholded = jnp.zeros_like(coeffs)
    sigmas = []

    for start, stop, do_thresh in band_info:
        band = jax.lax.dynamic_slice(coeffs, (0, start), (coeffs.shape[0], stop - start))
        sigma = jnp.median(jnp.abs(band)) / 0.6745
        sigmas.append(sigma)
        if do_thresh:
            n_j = stop - start
            t_j = kappa * sigma * jnp.sqrt(2.0 * jnp.log(jnp.float32(n_j)))
            band = jnp.where(jnp.abs(band) >= t_j, band, 0.0)
        thresholded = jax.lax.dynamic_update_slice(thresholded, band, (0, start))

    recon = thresholded @ W_inv
    return recon, thresholded, jnp.array(sigmas), jnp.count_nonzero(thresholded), thresholded.size
