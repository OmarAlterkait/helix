"""Single-pass coherent noise removal.

Four operations, each matched to the correlation structure of the noise:

1. Mask:     group-relative threshold |dig − median| > 3σ
2. Temporal:  ±4 tick dilation (coh ∥ leak in time → exclude)
3. Kernel:   3-tap (−β, 1, −β) spatial high-pass (coh ⊥ leak in space → filter)
4. Subtract: α × kernel(estimate)
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
    """Remove coherent noise from a single plane image.

    Parameters
    ----------
    image : array (n_wires, n_ticks), float32
        Pedestal-subtracted digitized readout.
    config : DetectorConfig
        Algorithm parameters.
    sigma_per_wire : array (n_wires,), optional
        Per-wire intrinsic noise sigma. If None, estimated from data via MAD
        on the group-relative residual.

    Returns
    -------
    cleaned : array (n_wires, n_ticks), float32
    """
    if get_backend() == "jax":
        return _remove_jax(image, config, sigma_per_wire)
    return _remove_numpy(image, config, sigma_per_wire)


def _remove_numpy(image: np.ndarray, config: DetectorConfig, sigma: np.ndarray | None) -> np.ndarray:
    from helix._numpy_ops import (
        group_median, broadcast_groups, signal_mask,
        temporal_dilate, masked_group_mean, xblock_kernel,
    )
    nw, nt = image.shape
    gs = config.group_size

    gm = group_median(image, gs)
    gm_full = broadcast_groups(gm, nw, gs)
    residual = image - gm_full

    if sigma is None:
        sigma = np.empty(nw, dtype=np.float32)
        for w in range(nw):
            sigma[w] = float(np.median(np.abs(residual[w]))) / 0.6745

    mask = signal_mask(residual, sigma, config.mask_threshold_nsigma)
    mask = temporal_dilate(mask, config.temporal_dilation_ticks)
    est, nuf = masked_group_mean(image, mask, gs)
    est = xblock_kernel(est, config.beta)

    ng = (nw + gs - 1) // gs
    est_full = broadcast_groups(est, nw, gs)
    alpha = broadcast_groups(nuf / float(gs), nw, gs)
    return image - alpha * est_full


def _remove_jax(image, config: DetectorConfig, sigma):
    import jax.numpy as jnp
    from helix._jax_ops import (
        group_median, broadcast_groups, signal_mask,
        temporal_dilate, masked_group_mean, xblock_kernel,
        pad_to_groups, pad_mask_to_groups,
    )
    nw, nt = image.shape
    gs = config.group_size

    image_j = jnp.asarray(image, dtype=jnp.float32)
    image_p = pad_to_groups(image_j, gs)

    gm = group_median(image_p, gs)
    gm_full = broadcast_groups(gm, nw, gs)
    residual = image_j - gm_full

    if sigma is None:
        sigma = jnp.median(jnp.abs(residual), axis=1) / 0.6745
    else:
        sigma = jnp.asarray(sigma, dtype=jnp.float32)

    mask = signal_mask(residual, sigma, config.mask_threshold_nsigma)
    mask = temporal_dilate(mask, config.temporal_dilation_ticks)
    mask_p = pad_mask_to_groups(mask, gs)
    image_p = pad_to_groups(image_j, gs)

    est, nuf = masked_group_mean(image_p, mask_p, gs)
    est = xblock_kernel(est, config.beta)

    est_full = broadcast_groups(est, nw, gs)
    alpha = broadcast_groups(nuf / float(gs), nw, gs)
    return image_j - alpha * est_full
