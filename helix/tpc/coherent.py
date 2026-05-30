"""Multi-pass coherent noise removal for TPC wire planes.

Pass 1: group median → residual → mask(kσ) → dilate → masked mean → α·subtract.
Passes 2..n: detect on the cleaned output → augment mask → re-estimate from the
ORIGINAL image → α·subtract. Using the original image for estimation (the
cleaned output only for detection) reduces bias: the better mask excludes more
sub-threshold signal from the coherent estimate.

Backend ops live in coherent_ops_{numpy,jax}; torch is not yet implemented
(the optical side, not coherent removal, is the torch workhorse).
"""
from __future__ import annotations

from typing import Any

from helix.core import backend as _backend
from helix.tpc.config import DetectorConfig


def remove_coherent(image: Any, config: DetectorConfig, sigma_per_wire: Any | None = None) -> Any:
    be = _backend.get_backend()
    if be == "numpy":
        return _remove_numpy(image, config, sigma_per_wire)
    if be == "jax":
        return _remove_jax(image, config, sigma_per_wire)
    raise NotImplementedError(
        f"coherent removal has no '{be}' backend yet; use 'numpy' or 'jax' "
        f"(add coherent_ops_{be}.py to extend, following the wavelet pattern)")


def _remove_numpy(image, config, sigma):
    import numpy as np
    from helix.tpc.coherent_ops_numpy import (
        group_median, broadcast_groups, signal_mask,
        temporal_dilate, masked_group_mean, mad_sigma_per_wire)
    nw, nt = image.shape
    gs, nsigma = config.group_size, config.mask_threshold_nsigma

    gm_full = broadcast_groups(group_median(image, gs), nw, gs)
    residual = image - gm_full
    if sigma is None:
        sigma = mad_sigma_per_wire(residual)

    mask = temporal_dilate(signal_mask(residual, sigma, nsigma), config.temporal_dilation_ticks)
    est, nuf = masked_group_mean(image, mask, gs)
    cleaned = image - broadcast_groups(nuf / float(gs), nw, gs) * broadcast_groups(est, nw, gs)

    for _ in range(config.n_passes - 1):
        detect = temporal_dilate(signal_mask(cleaned, sigma, nsigma), config.temporal_dilation_ticks)
        mask = mask | detect
        est, nuf = masked_group_mean(image, mask, gs)
        cleaned = image - broadcast_groups(nuf / float(gs), nw, gs) * broadcast_groups(est, nw, gs)
    return cleaned


def _remove_jax(image, config, sigma):
    import jax.numpy as jnp
    from helix.tpc.coherent_ops_jax import (
        group_median, broadcast_groups, signal_mask, temporal_dilate,
        masked_group_mean, pad_to_groups, pad_mask_to_groups)
    nw, nt = image.shape
    gs, nsigma = config.group_size, config.mask_threshold_nsigma

    image_j = jnp.asarray(image, dtype=jnp.float32)
    image_p = pad_to_groups(image_j, gs)
    gm_full = broadcast_groups(group_median(image_p, gs), nw, gs)
    residual = image_j - gm_full
    sigma = jnp.median(jnp.abs(residual), axis=1) / 0.6745 if sigma is None else jnp.asarray(sigma, jnp.float32)

    mask = temporal_dilate(signal_mask(residual, sigma, nsigma), config.temporal_dilation_ticks)
    est, nuf = masked_group_mean(pad_to_groups(image_j, gs), pad_mask_to_groups(mask, gs), gs)
    cleaned = image_j - broadcast_groups(nuf / float(gs), nw, gs) * broadcast_groups(est, nw, gs)

    for _ in range(config.n_passes - 1):
        detect = temporal_dilate(signal_mask(cleaned, sigma, nsigma), config.temporal_dilation_ticks)
        mask = mask | detect
        est, nuf = masked_group_mean(pad_to_groups(image_j, gs), pad_mask_to_groups(mask, gs), gs)
        cleaned = image_j - broadcast_groups(nuf / float(gs), nw, gs) * broadcast_groups(est, nw, gs)
    return cleaned
