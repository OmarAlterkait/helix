"""Two-pass coherent noise removal.

Pass 1: group median → residual → mask(kσ) → dilate → masked mean → α × subtract
Pass 2: detect on cleaned1 → augment mask → re-estimate from ORIGINAL → α × subtract

The second pass uses the cleaned output only for detection — estimation
and subtraction always operate on the original image. This reduces bias
because the better mask excludes more sub-threshold signal from the estimate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from helix.config import DetectorConfig
from helix._backend import get_backend


def remove_coherent(
    image: Any,
    config: DetectorConfig,
    sigma_per_wire: Any | None = None,
) -> Any:
    if get_backend() == "jax":
        return _remove_jax(image, config, sigma_per_wire)
    return _remove_numpy(image, config, sigma_per_wire)


def _remove_numpy(image: np.ndarray, config: DetectorConfig, sigma: np.ndarray | None) -> np.ndarray:
    from helix._numpy_ops import (
        group_median, broadcast_groups, signal_mask,
        temporal_dilate, masked_group_mean,
        mad_sigma_per_wire,
    )
    nw, nt = image.shape
    gs = config.group_size
    nsigma = config.mask_threshold_nsigma

    # Compute sigma if not provided
    gm = group_median(image, gs)
    gm_full = broadcast_groups(gm, nw, gs)
    residual = image - gm_full
    if sigma is None:
        sigma = mad_sigma_per_wire(residual)

    # Pass 1
    mask = signal_mask(residual, sigma, nsigma)
    mask = temporal_dilate(mask, config.temporal_dilation_ticks)
    est, nuf = masked_group_mean(image, mask, gs)
    est_full = broadcast_groups(est, nw, gs)
    alpha = broadcast_groups(nuf / float(gs), nw, gs)
    cleaned = image - alpha * est_full

    # Additional passes: detect on cleaned → augment mask → re-estimate from original
    for _ in range(config.n_passes - 1):
        detect = signal_mask(cleaned, sigma, nsigma)
        detect = temporal_dilate(detect, config.temporal_dilation_ticks)
        mask = mask | detect
        est, nuf = masked_group_mean(image, mask, gs)
        est_full = broadcast_groups(est, nw, gs)
        alpha = broadcast_groups(nuf / float(gs), nw, gs)
        cleaned = image - alpha * est_full

    return cleaned


def _remove_jax(image, config: DetectorConfig, sigma):
    import jax.numpy as jnp
    from helix._jax_ops import (
        group_median, broadcast_groups, signal_mask,
        temporal_dilate, masked_group_mean,
        pad_to_groups, pad_mask_to_groups,
    )
    nw, nt = image.shape
    gs = config.group_size
    nsigma = config.mask_threshold_nsigma

    image_j = jnp.asarray(image, dtype=jnp.float32)
    image_p = pad_to_groups(image_j, gs)

    gm = group_median(image_p, gs)
    gm_full = broadcast_groups(gm, nw, gs)
    residual = image_j - gm_full

    if sigma is None:
        sigma = jnp.median(jnp.abs(residual), axis=1) / 0.6745
    else:
        sigma = jnp.asarray(sigma, dtype=jnp.float32)

    # Pass 1
    mask = signal_mask(residual, sigma, nsigma)
    mask = temporal_dilate(mask, config.temporal_dilation_ticks)
    mask_p = pad_mask_to_groups(mask, gs)
    image_p = pad_to_groups(image_j, gs)

    est, nuf = masked_group_mean(image_p, mask_p, gs)
    est_full = broadcast_groups(est, nw, gs)
    alpha = broadcast_groups(nuf / float(gs), nw, gs)
    cleaned = image_j - alpha * est_full

    # Additional passes
    for _ in range(config.n_passes - 1):
        detect = signal_mask(cleaned, sigma, nsigma)
        detect = temporal_dilate(detect, config.temporal_dilation_ticks)
        mask = mask | detect
        mask_p = pad_mask_to_groups(mask, gs)
        image_p = pad_to_groups(image_j, gs)

        est, nuf = masked_group_mean(image_p, mask_p, gs)
        est_full = broadcast_groups(est, nw, gs)
        alpha = broadcast_groups(nuf / float(gs), nw, gs)
        cleaned = image_j - alpha * est_full

    return cleaned
