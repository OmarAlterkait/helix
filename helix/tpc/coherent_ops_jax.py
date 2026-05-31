"""JAX/GPU primitives for TPC coherent-noise removal (group/wire ops)."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap


@partial(jit, static_argnums=(1,))
def group_median(image: jnp.ndarray, group_size: int) -> jnp.ndarray:
    # nanmedian so NaN-padded rows (the partial last group) are ignored -> median
    # over the real wires only, matching the numpy backend. Full groups have no NaN.
    ng = image.shape[0] // group_size
    nt = image.shape[1]
    return jnp.nanmedian(image.reshape(ng, group_size, nt), axis=1)


def broadcast_groups(group_arr: jnp.ndarray, n_wires: int, group_size: int) -> jnp.ndarray:
    ng = group_arr.shape[0]
    return jnp.repeat(group_arr[:, None, :], group_size, axis=1).reshape(
        ng * group_size, group_arr.shape[1])[:n_wires]


@jit
def signal_mask(residual: jnp.ndarray, sigma: jnp.ndarray, nsigma: float) -> jnp.ndarray:
    return jnp.abs(residual) > nsigma * sigma[:, None]


@partial(jit, static_argnums=(1,))
def temporal_dilate(mask: jnp.ndarray, ticks: int) -> jnp.ndarray:
    if ticks <= 1:
        return mask
    kernel = jnp.ones(ticks)
    dilated = vmap(lambda row: jnp.convolve(row, kernel, mode="same"))(mask.astype(jnp.float32))
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


def pad_to_groups(image: jnp.ndarray, group_size: int) -> jnp.ndarray:
    nw, nt = image.shape
    ng = (nw + group_size - 1) // group_size
    pad = ng * group_size - nw
    if pad > 0:
        return jnp.concatenate([image, jnp.zeros((pad, nt), dtype=image.dtype)], axis=0)
    return image


def pad_to_groups_nan(image: jnp.ndarray, group_size: int) -> jnp.ndarray:
    """Pad with NaN (not 0) so a group median over the padded block ignores the
    padding via nanmedian — used for the partial last group's coherent estimate."""
    nw, nt = image.shape
    ng = (nw + group_size - 1) // group_size
    pad = ng * group_size - nw
    if pad > 0:
        return jnp.concatenate([image, jnp.full((pad, nt), jnp.nan, dtype=image.dtype)], axis=0)
    return image


def pad_mask_to_groups(mask: jnp.ndarray, group_size: int) -> jnp.ndarray:
    nw, nt = mask.shape
    ng = (nw + group_size - 1) // group_size
    pad = ng * group_size - nw
    if pad > 0:
        return jnp.concatenate([mask, jnp.ones((pad, nt), dtype=bool)], axis=0)
    return mask
