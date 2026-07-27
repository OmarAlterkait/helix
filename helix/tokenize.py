"""Coefficient normalisation — the stateless front half of the FM tokenizer.

Pure numpy, zero learnable parameters: this is DATA shaping, and it lives in helix
because the *logic* belongs with the model's representation, while the transform
that runs it in a DataLoader worker is pimm's (see COEFF_CORPUS_DESIGN.md §1).

The corpus stores RAW coefficients plus a per-``(plane, band)`` table
``norm_sigma``; the model consumes ``arcsinh(value / sigma)``. That split is what
lets the corpus be basis-faithful while the normalisation stays a tokenizer
concern.

**The indexing trap this module exists to close.** ``norm_sigma`` rows are ordered
by POSITION in ``gids`` — the sorted plane set actually present — not by gid
value. They coincide only when gids are contiguous ``0..G-1``. A detector with a
dead or absent plane (gids ``[0,1,2,4,5]``) makes ``norm_sigma[gid]`` silently
select the wrong plane's sigma, or run off the end. Always resolve through
:func:`gid_rows`.
"""
from __future__ import annotations

import numpy as np

__all__ = ["gid_rows", "sigma_for_rows", "normalize_values", "denormalize_values"]


def gid_rows(plane_gid, gids):
    """Map each row's gid VALUE to its ROW INDEX in ``norm_sigma`` / ``gids``.

    Raises if a row carries a gid absent from ``gids`` — silently dropping or
    wrapping such a row would mis-normalise it.
    """
    gids = np.asarray(gids, np.int64)
    order = np.argsort(gids)
    pos = np.searchsorted(gids[order], np.asarray(plane_gid, np.int64))
    pos = np.clip(pos, 0, gids.size - 1)
    row = order[pos]
    if not np.array_equal(gids[row], np.asarray(plane_gid, np.int64)):
        missing = np.setdiff1d(np.unique(plane_gid), gids)
        raise ValueError(
            f"plane_gid contains gids absent from the shard's gids: {missing.tolist()}")
    return row.astype(np.int32)


def sigma_for_rows(plane_gid, band, gids, norm_sigma):
    """Per-row sigma: ``norm_sigma[row_of(gid), band]`` (never ``norm_sigma[gid]``)."""
    ns = np.asarray(norm_sigma, np.float32)
    if ns.ndim != 2:
        raise ValueError(f"norm_sigma must be (n_gid, n_bands), got {ns.shape}")
    if ns.shape[0] != len(gids):
        raise ValueError(
            f"norm_sigma has {ns.shape[0]} rows but {len(gids)} gids — the table "
            "is row-indexed by position in gids")
    b = np.asarray(band, np.int64)
    if b.size and int(b.max()) >= ns.shape[1]:
        raise ValueError(f"band {int(b.max())} >= n_bands {ns.shape[1]}")
    return ns[gid_rows(plane_gid, gids), b]


def normalize_values(value, plane_gid, band, gids, norm_sigma, *, sigma_norm=1.0,
                     eps=1e-6):
    """``arcsinh(value / sigma)`` with the per-(plane, band) sigma.

    ``sigma_norm`` is carried for provenance only: the old pipeline stored
    ``value * SIGMA/sigma`` and then took ``arcsinh(v / SIGMA)``, so SIGMA
    cancels. Storing RAW values makes that cancellation explicit — the scalar
    does not affect the result and defaults to 1.
    """
    sig = np.maximum(sigma_for_rows(plane_gid, band, gids, norm_sigma), eps)
    return np.arcsinh(np.asarray(value, np.float32) / sig).astype(np.float32)


def denormalize_values(tok, plane_gid, band, gids, norm_sigma, *, eps=1e-6):
    """Inverse of :func:`normalize_values` (for reconstruction / debugging)."""
    sig = np.maximum(sigma_for_rows(plane_gid, band, gids, norm_sigma), eps)
    return (np.sinh(np.asarray(tok, np.float32)) * sig).astype(np.float32)
