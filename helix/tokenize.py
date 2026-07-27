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


# ---- the patch tokenizer (port of research vit_tpc.assemble_tpc_band) ------

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchConfig:
    """Patch geometry + time-coordinate constants of the FM tokenizer.

    Defaults reproduce ``research/coeff_foundation_model/vit_tpc.py`` exactly:
    PW=16 wires x PT=8 band-ticks -> 128 slots, over the first 4 bands
    (A4, D4, D3, D2 — the FM dropped D1). ``FM_PW``/``FM_PT`` env-var mutation of
    module globals is replaced by explicit fields.
    """
    pw: int = 16
    pt: int = 8
    n_bands: int = 4                                  # A4,D4,D3,D2 (D1 dropped)
    lev: tuple = (4, 4, 3, 2)                         # DWT level per band
    delta: tuple = (-2.38, 0.62, 0.75, 0.50)          # per-band tick offset
    toff: tuple = (-17.4, 2.6, 5.5)                   # U,V,Y sensor->drift (pb_labels.TOFF)

    @property
    def n_slot(self) -> int:
        return self.pw * self.pt


def assemble(band, plane_gid, wire, tau, value, *, gids, n_wires, band_lengths,
             norm_sigma, cfg=PatchConfig(), value_clean=None, dead_frac=0.0,
             rng=None):
    """Coefficient rows -> per-band 2-D patch tokens (stateless, pure numpy).

    Faithful port of ``vit_tpc.assemble_tpc_band``. The one substantive change is
    normalisation: the old cache stored ``val = raw * SIGMA/sigma_tab`` and the
    tokenizer took ``arcsinh(val/SIGMA)``, so SIGMA cancelled and the result was
    ``arcsinh(raw/sigma_tab)``. The corpus now stores RAW values, so we compute
    ``arcsinh(raw/norm_sigma)`` directly — identical output, one less place for a
    baked-in constant to drift.

    Returns numpy arrays; converting to tensors is the caller's job (the pimm
    transform), keeping this importable without torch.
    """
    band = np.asarray(band, np.int64); plane_gid = np.asarray(plane_gid, np.int64)
    wire = np.asarray(wire, np.int64); tau = np.asarray(tau, np.int64)
    value = np.asarray(value, np.float32)
    bl = np.asarray(band_lengths, np.int64)
    pw, pt, nslot = cfg.pw, cfg.pt, cfg.n_slot

    keep = band < cfg.n_bands                        # D1 (and beyond) dropped
    band, plane_gid, wire, tau = band[keep], plane_gid[keep], wire[keep], tau[keep]
    value = value[keep]
    clean = None if value_clean is None else np.asarray(value_clean, np.float32)[keep]

    val = normalize_values(value, plane_gid, band, gids, norm_sigma)
    target = (normalize_values(clean, plane_gid, band, gids, norm_sigma)
              if clean is not None else np.zeros_like(val))

    wb, tb = wire // pw, tau // pt
    key = (plane_gid << 40) | (band << 36) | (wb << 18) | tb
    uniq, cell = np.unique(key, return_inverse=True)
    n_cells = len(uniq)
    slot = (wire % pw) * pt + (tau % pt)
    cell_band = ((uniq >> 36) & 0xF).astype(np.int64)
    cell_gid = (uniq >> 40).astype(np.int64)
    cell_wb = ((uniq >> 18) & 0x3FFFF).astype(np.int64)
    cell_tb = (uniq & 0x3FFFF).astype(np.int64)

    occ = np.zeros((n_cells, nslot), bool)
    inp = np.zeros((n_cells, nslot), np.float32)
    tgt = np.zeros((n_cells, nslot), np.float32)
    occ[cell, slot] = True
    inp[cell, slot] = val
    tgt[cell, slot] = target

    # valid slots: wire-in-block < plane n_wires, tick-in-block < band length.
    # n_wires is resolved through gid_rows, NOT n_wires[gid] — they coincide only
    # for contiguous gids (the old code assumed that).
    nw = np.asarray(n_wires, np.int64)[gid_rows(cell_gid, gids)]
    Lb = bl[cell_band]
    wi = np.arange(pw)[None, :]
    ti = np.arange(pt)[None, :]
    wok = (cell_wb[:, None] * pw + wi) < nw[:, None]
    tok = (cell_tb[:, None] * pt + ti) < Lb[:, None]
    valid = (wok[:, :, None] & tok[:, None, :]).reshape(n_cells, nslot)

    dead = np.zeros((n_cells, pw), bool)
    if dead_frac > 0:                                 # wire-kill augmentation
        rng = np.random.default_rng() if rng is None else rng
        kill = rng.random((n_cells, pw)) < dead_frac
        dead = kill & ((cell_wb[:, None] * pw + wi) < nw[:, None])
        ks = np.repeat(dead, pt, axis=1)
        inp[ks] = 0.0
        occ[ks] = False                               # killed -> not active input
        # targets/valid unchanged: the model must still predict a dead wire

    # RoPE time coord: grid-CENTER drift time per (tick-block, band), minus the
    # per-plane sensor->drift offset. Center (not survivor-max) so it is
    # band-aligned and occupancy-independent; TOFF so planes share a zero.
    dec = (1 << np.asarray(cfg.lev, np.int64)).astype(np.float32)
    toff = np.asarray(cfg.toff, np.float32)
    center_tau = cell_tb.astype(np.float32) * pt + pt / 2.0
    cell_t = ((center_tau + np.asarray(cfg.delta, np.float32)[cell_band]) * dec[cell_band]
              - toff[cell_gid % 3]).astype(np.float32)

    return dict(
        band=band, val=val, target=target, cell=cell.astype(np.int64),
        slot=slot.astype(np.int64), n_cells=n_cells,
        occ=occ.astype(np.float32), inp=inp, tgt=tgt, valid=valid,
        dead=dead.astype(np.float32), cell_band=cell_band, cell_gid=cell_gid,
        cell_t=cell_t, cell_wire=(cell_wb * pw).astype(np.float32),
    )
