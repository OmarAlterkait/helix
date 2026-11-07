"""Coefficient normalisation — the stateless front half of the FM tokenizer.

Pure numpy, zero learnable parameters: this is DATA shaping, and it lives in helix
because the *logic* belongs with the model's representation, while the transform
that runs it in a DataLoader worker is pimm's (see COEFF_CORPUS_DESIGN at tag archive-2026-11 §1).

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


def normalize_values(value, plane_gid, band, gids, norm_sigma, *, eps=1e-6):
    """``arcsinh(value / sigma)`` with the per-(plane, band) sigma."""
    sig = np.maximum(sigma_for_rows(plane_gid, band, gids, norm_sigma), eps)
    return np.arcsinh(np.asarray(value, np.float32) / sig).astype(np.float32)


def denormalize_values(tok, plane_gid, band, gids, norm_sigma, *, eps=1e-6):
    """Inverse of :func:`normalize_values` (for reconstruction / debugging)."""
    sig = np.maximum(sigma_for_rows(plane_gid, band, gids, norm_sigma), eps)
    return (np.sinh(np.asarray(tok, np.float32)) * sig).astype(np.float32)


# ---- the patch tokenizer ----------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchConfig:
    """Patch geometry + time-coordinate constants of the FM tokenizer.

    PW=16 wires x PT=8 band-ticks -> 128 slots, over the first 4 bands (A4, D4,
    D3, D2; D1 is dropped).
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


#: Keys a recorded tokenizer cfg may hold that configure nothing any more. ``cell_t`` chose the
#: RoPE time coordinate; the only accepted value was the patch's grid centre (a coordinate from
#: the cell's own coefficients would leak the masked target into its position), and every
#: resolved config written before its removal records it.
RETIRED_KEYS = ("cell_t",)


def patch_config(cfg) -> PatchConfig:
    """A :class:`PatchConfig` from a config dict (a run's recorded one included): lists become
    tuples (JSON has none) and :data:`RETIRED_KEYS` are dropped."""
    if isinstance(cfg, PatchConfig):
        return cfg
    return PatchConfig(**{k: tuple(v) if isinstance(v, list) else v
                          for k, v in dict(cfg or {}).items() if k not in RETIRED_KEYS})


def _require_cfg(cfg):
    """No silent `PatchConfig()` fallback anywhere in this module: the patch
    geometry must be the one the model was trained with, so the caller says
    which."""
    if cfg is None:
        raise ValueError(
            "cfg is REQUIRED: pass the PatchConfig the model was trained with "
            "(a default geometry would silently mis-tokenize a pw/pt variant).")
    return cfg


# --------------------------------------------------------------------------
# cell identity
# --------------------------------------------------------------------------
# Bit layout of a cell key. A cell is one (plane, band, wire-block, tick-block)
# patch, and the key packs those four into one int64 so np.unique can find the
# distinct cells in a single pass.
#
#   bits 40+    plane_gid
#   bits 36-39  band        (4 bits -> up to 16 bands)
#   bits 18-35  wire block  (18 bits)
#   bits 0-17   tick block  (18 bits)
_CELL_BAND_SHIFT, _CELL_WB_SHIFT, _CELL_GID_SHIFT = 36, 18, 40
_CELL_BLOCK_MASK, _CELL_BAND_MASK = 0x3FFFF, 0xF
#: plane_gid occupies bits 40..62 — bit 63 is the sign, so it must stay clear.
_CELL_GID_MASK = (1 << 23) - 1


def cell_key(plane_gid, band, wire, tau, cfg=None):
    """Pack coordinates into the cell id the tokenizer groups by.

    Public because anything that wants to attach per-cell information to a
    tokenized event — a probe target, a truth label, an attention mask — has to
    agree with :func:`assemble` on which coefficients share a cell, and the only
    way to agree is to run the same packing.

    ``wire``/``tau`` are RAW coordinates; the patch division happens here, so a
    caller never has to know ``pw``/``pt``. Returns int64, directly comparable
    against the keys :func:`assemble` builds.
    """
    cfg = _require_cfg(cfg)
    plane_gid = np.asarray(plane_gid, np.int64)
    band = np.asarray(band, np.int64)
    wb = np.asarray(wire, np.int64) // cfg.pw
    tb = np.asarray(tau, np.int64) // cfg.pt
    # Check BOTH ends of every field. Guarding only .max() left the two cheapest
    # routes to the aliasing this raises about wide open: a single negative tau
    # is all-ones in two's complement, so `| tb` sets every bit and the key
    # becomes exactly -1 — collapsing rows from different planes, bands and
    # wires into ONE cell, while max(tb)=0 sails past an upper-bound-only check.
    # plane_gid was unbounded entirely: gid 2**24 wraps to gid 0.
    for name, a, hi in (("wire block", wb, _CELL_BLOCK_MASK),
                        ("tick block", tb, _CELL_BLOCK_MASK),
                        ("band", band, _CELL_BAND_MASK),
                        ("plane_gid", plane_gid, _CELL_GID_MASK)):
        if not a.size:
            continue
        lo_v, hi_v = int(a.min()), int(a.max())
        if lo_v < 0 or hi_v > hi:
            raise ValueError(
                f"{name} out of range [0, {hi}]: got [{lo_v}, {hi_v}]. The key "
                f"would silently alias distinct cells (a negative value sets "
                f"every bit; an over-range one wraps).")
    return ((plane_gid << _CELL_GID_SHIFT) | (band << _CELL_BAND_SHIFT)
            | (wb << _CELL_WB_SHIFT) | tb)


def unpack_cell_key(key):
    """Inverse of :func:`cell_key`: ``(plane_gid, band, wire_block, tick_block)``."""
    key = np.asarray(key, np.int64)
    return (key >> _CELL_GID_SHIFT,
            (key >> _CELL_BAND_SHIFT) & _CELL_BAND_MASK,
            (key >> _CELL_WB_SHIFT) & _CELL_BLOCK_MASK,
            key & _CELL_BLOCK_MASK)


def tick_of_tau(tau, plane_gid, band, cfg=None):
    """Physical drift time (raw ticks) of the coefficient at ``(band, tau)``.

    The forward half of the tokenizer's time convention:
    ``(tau + delta[band]) * 2**lev[band] - toff[plane_gid % 3]``.
    :func:`tau_of_tick` inverts it.
    """
    cfg = _require_cfg(cfg)
    band = np.asarray(band, np.int64)
    dec = (1 << np.asarray(cfg.lev, np.int64)).astype(np.float32)
    delta = np.asarray(cfg.delta, np.float32)
    toff = np.asarray(cfg.toff, np.float32)
    return ((np.asarray(tau, np.float32) + delta[band]) * dec[band]
            - toff[np.asarray(plane_gid, np.int64) % 3]).astype(np.float32)


def tau_of_tick(tick, plane_gid, band, band_lengths, cfg=None):
    """Inverse of :func:`tick_of_tau`: which ``tau`` in ``band`` covers ``tick``.

    Rounds (not floors) and clips into ``[0, band_lengths[band])``, matching the
    tokenizer's own convention.

    ``band_lengths`` is REQUIRED and comes from the shard's ``/config`` — it is a
    property of the CORPUS (it moves with ``n_ticks_raw`` and ``dwt_level``),
    while ``cfg`` is a property of the CHECKPOINT. A hard-coded copy that drifts
    from the corpus makes a probe gather the WRONG cell's features, which does
    not crash — it returns a number near the geometry null that looks like a
    result.
    """
    cfg = _require_cfg(cfg)
    band = np.asarray(band, np.int64)
    dec = (1 << np.asarray(cfg.lev, np.int64)).astype(np.float64)
    delta = np.asarray(cfg.delta, np.float64)
    toff = np.asarray(cfg.toff, np.float64)
    lens = np.asarray(band_lengths, np.int64)
    t_sig = np.asarray(tick, np.float64) + toff[np.asarray(plane_gid, np.int64) % 3]
    tau = np.round(t_sig / dec[band] - delta[band]).astype(np.int64)
    return np.clip(tau, 0, lens[band] - 1)


def pixel_cells(plane_gid, wire, tick, band_lengths, cfg=None):
    """The ``n_bands`` cell keys covering a raw pixel ``(plane_gid, wire, tick)``.

    Public for the same reason :func:`cell_key` is: anything attaching per-cell
    information to a tokenized event — a probe target above all — must agree with
    :func:`assemble` on which cell a pixel lands in, and the only way to agree is
    to run the same arithmetic. Returns ``(len(pixels), n_bands)`` int64.
    """
    cfg = _require_cfg(cfg)
    plane_gid = np.asarray(plane_gid, np.int64)
    wire = np.asarray(wire, np.int64)
    out = np.empty((plane_gid.size, cfg.n_bands), np.int64)
    for b in range(cfg.n_bands):
        tau = tau_of_tick(tick, plane_gid, np.full(plane_gid.shape, b), band_lengths, cfg)
        out[:, b] = cell_key(plane_gid, np.full(plane_gid.shape, b), wire, tau, cfg)
    return out


def assemble(band, plane_gid, wire, tau, value, *, gids, n_wires, band_lengths,
             norm_sigma, cfg=None, dead_frac=0.0, rng=None, wire_valid=None):
    """Coefficient rows -> per-band 2-D patch tokens (stateless, pure numpy).

    Values are normalised to ``arcsinh(raw / norm_sigma)`` per (plane, band).

    ``wire_valid`` (G, >= max n_wires) bool, rows in ``gids`` order: False marks an
    excluded channel (dead, noisy, not instrumented; real detectors). Its slots are
    not valid (no target), like slots past ``n_wires``: the detector recorded nothing
    there to predict. ``None`` = every wire valid (simulation).

    Returns numpy arrays; converting to tensors is the caller's job (the pimm
    transform), keeping this importable without torch.
    """
    cfg = _require_cfg(cfg)
    band = np.asarray(band, np.int64); plane_gid = np.asarray(plane_gid, np.int64)
    wire = np.asarray(wire, np.int64); tau = np.asarray(tau, np.int64)
    value = np.asarray(value, np.float32)
    bl = np.asarray(band_lengths, np.int64)
    pw, pt, nslot = cfg.pw, cfg.pt, cfg.n_slot

    keep = band < cfg.n_bands                        # D1 (and beyond) dropped
    band, plane_gid, wire, tau = band[keep], plane_gid[keep], wire[keep], tau[keep]
    value = value[keep]

    sig_row = np.maximum(sigma_for_rows(plane_gid, band, gids, norm_sigma), 1e-6)
    ratio = (np.asarray(value, np.float32) / sig_row).astype(np.float32)
    val = np.arcsinh(ratio).astype(np.float32)          # == normalize_values(...)

    wb, tb = wire // pw, tau // pt
    key = cell_key(plane_gid, band, wire, tau, cfg)
    uniq, cell = np.unique(key, return_inverse=True)
    n_cells = len(uniq)
    slot = (wire % pw) * pt + (tau % pt)
    cell_gid, cell_band, cell_wb, cell_tb = (
        a.astype(np.int64) for a in unpack_cell_key(uniq))

    occ = np.zeros((n_cells, nslot), bool)
    inp = np.zeros((n_cells, nslot), np.float32)
    tgt = np.zeros((n_cells, nslot), np.float32)
    occ[cell, slot] = True
    # (cell, slot) is a bijection with (plane_gid, band, wire, tau), so two rows
    # landing in one slot means the event carries DUPLICATE coordinates. The
    # scatter below is last-write-wins, so that coefficient would vanish with no
    # error and `inp[cell, slot] == val` would quietly stop holding — the exact
    # invariant the FM's loss gathers on. occ.sum() detects it exactly, for the
    # cost of one bool reduction over a grid we just allocated.
    n_written = int(occ.sum())
    if n_written != cell.shape[0]:
        raise ValueError(
            f"duplicate coefficient coordinates: {cell.shape[0] - n_written} of "
            f"{cell.shape[0]} rows share a (plane_gid, band, wire, tau) with another "
            f"row, so they would be silently overwritten in the token grid.")
    inp[cell, slot] = val
    # The target IS the (normalised) input: real detector data has no noise-free
    # counterpart, so masked prediction infers the hidden coefficients from context
    # (Noise2Self). `tgt` differs from `inp` only where augmentation (dead_frac)
    # blanked the input.
    tgt[cell, slot] = val

    # valid slots: wire-in-block < plane n_wires, tick-in-block < band length.
    # n_wires is resolved through gid_rows, NOT n_wires[gid] — they coincide only
    # for contiguous gids.
    nw = np.asarray(n_wires, np.int64)[gid_rows(cell_gid, gids)]
    Lb = bl[cell_band]
    wi = np.arange(pw)[None, :]
    ti = np.arange(pt)[None, :]
    wok = (cell_wb[:, None] * pw + wi) < nw[:, None]
    if wire_valid is not None:
        wv = np.asarray(wire_valid, bool)
        w_abs = np.minimum(cell_wb[:, None] * pw + wi, wv.shape[1] - 1)
        wok &= wv[gid_rows(cell_gid, gids)[:, None], w_abs]
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

    # RoPE time coord: the centre of the patch's tick range, which depends only on WHICH cell it
    # is (never on its contents: `t_phys` is the masked tokens' position), TOFF-corrected so
    # planes share a zero.
    dec = (1 << np.asarray(cfg.lev, np.int64)).astype(np.float32)
    toff = np.asarray(cfg.toff, np.float32)
    center_tau = cell_tb.astype(np.float32) * pt + pt / 2.0
    cell_t = ((center_tau + np.asarray(cfg.delta, np.float32)[cell_band]) * dec[cell_band]
              - toff[cell_gid % 3]).astype(np.float32)

    return dict(
        band=band, val=val, cell=cell.astype(np.int64),
        slot=slot.astype(np.int64), n_cells=n_cells,
        occ=occ.astype(np.float32), inp=inp, tgt=tgt, valid=valid,
        dead=dead.astype(np.float32), cell_band=cell_band, cell_gid=cell_gid,
        # The sorted unique cell keys, in CELL-INDEX order — so `cell_key(...)`
        # for a pixel can be searchsorted into this to find its feature row.
        # Returned rather than recomputed by callers: `np.unique` already
        # produced it here, and a second derivation is a second chance to drift.
        cell_key=uniq.astype(np.int64),
        cell_t=cell_t, cell_wire=(cell_wb * pw).astype(np.float32),
        # Integer block indices. cell_wire carries wb*pw already, but as float32
        # for RoPE; cell_tb has no exact twin (`cell_t` is a float physical
        # time). Without these two the token grid cannot be mapped back to
        # (wire, tau) — see `detokenize`.
        cell_wb=cell_wb, cell_tb=cell_tb,
    )


# ---- the inverse: tokens -> coefficient rows -------------------------------

def _rows_from_grid(occ_mask, values, cell_band, cell_gid, cell_wb, cell_tb, *,
                    gids, norm_sigma, cfg, space="asinh"):
    """Shared core: an occupancy mask over the (cell, slot) grid -> coeff rows.

    ``space`` says what ``values`` holds, because the inverse differs:

      ``"asinh"``  the token, ``arcsinh(raw/sigma)`` -> ``sinh(v) * sigma``.
                   Correct for ``inp``/``tgt``.
      ``"ratio"``  the dimensionless coefficient ``raw/sigma`` -> ``v * sigma``.
                   Correct for the CATEGORICAL head, whose read-back is
                   ``sum_k p_k * E[raw/sigma | bin k]`` and is ALREADY an
                   expectation of ``sinh``.

    Passing a categorical posterior mean with ``space="asinh"`` applies ``sinh``
    to a mean over many bins, which is Jensen-biased ~31% low; hence the
    explicit parameter rather than a convention.

    ``(cell, slot)`` is a bijection with ``(plane_gid, band, wire, tau)`` given
    the per-cell block indices, so this inverts the scatter exactly. What differs
    between the two public entry points is only WHERE the mask comes from: known
    occupancy (``detokenize``) or predicted occupancy (``decode_prediction``).
    """
    cfg = _require_cfg(cfg)
    pw, pt = cfg.pw, cfg.pt
    cell, slot = np.nonzero(np.asarray(occ_mask))
    band = np.asarray(cell_band, np.int64)[cell]
    plane_gid = np.asarray(cell_gid, np.int64)[cell]
    wire = np.asarray(cell_wb, np.int64)[cell] * pw + slot // pt
    tau = np.asarray(cell_tb, np.int64)[cell] * pt + slot % pt
    # undo the per-(plane, band) normalisation; see `space` above for which one
    sig = np.maximum(sigma_for_rows(plane_gid, band, gids, norm_sigma), 1e-6)
    v = np.asarray(values, np.float32)[cell, slot]
    if space == "asinh":
        v = np.sinh(v)
    elif space != "ratio":
        raise ValueError(f"space must be 'asinh' or 'ratio', got {space!r}")
    value = (v * sig).astype(np.float32)
    return dict(band=band.astype(np.uint8), plane_gid=plane_gid.astype(np.int32),
                wire=wire.astype(np.int32), tau=tau.astype(np.int32), value=value)


def detokenize(tok, *, gids, norm_sigma, cfg=None, values_key="inp"):
    """Exact inverse of :func:`assemble` — tokens back to coefficient rows.

    Returns ``{band, plane_gid, wire, tau, value}``, the same columns a
    ``CoeffEvent`` carries, so a round trip is directly comparable against the
    rows that went in (up to ordering, and minus the bands ``assemble`` drops:
    it keeps only ``band < cfg.n_bands``).

    This is the verifiable half of the inverse. ``decode_prediction`` handles
    model output, where occupancy is predicted rather than known; that one cannot
    be checked against ground truth, so it is built on this.

    Exact only for ``dead_frac == 0``: the wire-kill augmentation zeroes ``inp``
    and clears ``occ``, which is deliberately destructive.
    """
    cfg = _require_cfg(cfg)
    return _rows_from_grid(np.asarray(tok["occ"]).astype(bool), tok[values_key],
                           tok["cell_band"], tok["cell_gid"],
                           tok["cell_wb"], tok["cell_tb"],
                           gids=gids, norm_sigma=norm_sigma, cfg=cfg)


def decode_prediction(occ_logit, values, tok, *, gids, norm_sigma,
                      cfg=None, threshold=0.0, respect_valid=True,
                      space="asinh"):
    """Model output -> coefficient rows.

    ``occ_logit`` and ``values`` are the FM's two heads over the ``(cell, slot)``
    grid. A coefficient is emitted where the occupancy logit exceeds
    ``threshold`` (0.0 == probability 0.5) and, unless ``respect_valid=False``,
    where the slot is geometrically real — a slot past the plane's wire count or
    the band's length cannot hold a coefficient no matter what the head says.

    For a CATEGORICAL head, pass ``decode_categorical(...)`` output with
    ``space="ratio"``. Do NOT pass bin centres with the default ``space="asinh"``:
    that applies ``sinh`` to a posterior mean over many bins and reads back ~31%
    of the charge.
    """
    cfg = _require_cfg(cfg)
    occ = np.asarray(occ_logit) > threshold
    if respect_valid and "valid" in tok:
        occ = occ & np.asarray(tok["valid"]).astype(bool)
    return _rows_from_grid(occ, values, tok["cell_band"], tok["cell_gid"],
                           tok["cell_wb"], tok["cell_tb"],
                           gids=gids, norm_sigma=norm_sigma, cfg=cfg, space=space)



def bin_centroids_ratio(edges, cent_ratio=None):
    """Per-band ``E[raw/sigma | bin]`` for the categorical head, shape (n_band, K).

    Prefers the MEASURED table (``cent_ratio`` from ``helix.data.bins``, which
    is what the reference ships and reads back with). Falls back to the exact
    closed form for a uniform density inside each bin::

        E[sinh t | t in (a, b)] = (cosh b - cosh a) / (b - a)

    which agrees with the measured table to ~0.3%. The two OPEN
    outer edges (written as the +-1e18 sentinel) are closed by reflecting the
    neighbouring width; that under-reads the outermost bin by ~20-25% against the
    measured table, which is unavoidable without the table and is why the table
    is preferred.
    """
    if cent_ratio is not None:
        c = np.asarray(cent_ratio, np.float64)
        if np.isfinite(c).all():
            return c.astype(np.float32)   # measured beats derived; no closure needed
    e = _close_open_edges(edges)
    lo, hi = e[:, :-1], e[:, 1:]
    w = np.maximum(hi - lo, 1e-12)
    return ((np.cosh(hi) - np.cosh(lo)) / w).astype(np.float32)


def bin_centroids_asinh(edges):
    """Per-band bin midpoints in TOKEN space, shape (n_band, K).

    The estimator for ``var_expl``, which compares a posterior mean against the
    target in the space the target lives in — so no ``sinh`` is involved and the
    midpoint is the right summary. Contrast ``bin_centroids_ratio``, which is an
    expectation OF ``sinh`` and is the only correct thing for charge.
    """
    e = _close_open_edges(edges)
    return (0.5 * (e[:, :-1] + e[:, 1:])).astype(np.float32)


def _close_open_edges(edges):
    """Replace the +-1e18 open-bin sentinels with a reflected neighbouring width.

    ``helix.data.bins`` writes the outer edges as +-1e18 so a tail can never
    fall off the grid. Any centroid formula has to invent a finite edge for those
    two bins; reflecting the adjacent width is the convention, and it is why a
    DERIVED outer centroid runs ~24% below a measured one, which costs 2.7-3.0%
    of a band's sum|centroid| (measured on the R1 corpus).

    Needs three finite interior edges to reflect from. With fewer (K < 3) there
    is nothing to reflect, so it raises — reachable, because the derivation runs
    whenever a measured table is absent.
    """
    e = np.array(edges, dtype=np.float64, copy=True)
    BIG = 1e6                       # token space is arcsinh(); it stays single-digit
    if e.shape[1] < 4:
        raise ValueError(
            f"need at least 3 bins to close an open edge, got K={e.shape[1]-1}; "
            f"a table this small cannot have its outer centroids derived")
    for b in range(e.shape[0]):
        if abs(e[b, 0]) >= BIG:
            e[b, 0] = e[b, 1] - (e[b, 2] - e[b, 1])
        if abs(e[b, -1]) >= BIG:
            e[b, -1] = e[b, -2] + (e[b, -2] - e[b, -3])
    if np.abs(e).max() >= BIG:
        raise ValueError(f"edges still hold a sentinel: max|e| = {np.abs(e).max():.3e}")
    return e


def decode_categorical(logits, cell_band, centroids, readout="mean"):
    """Categorical head -> value in UNITS OF SIGMA (i.e. ``space="ratio"``).

    ``logits`` is ``(n_cells, n_slot, K)``; ``centroids`` is ``(n_band, K)`` from
    :func:`bin_centroids_ratio`. Returns ``(n_cells, n_slot)``.

    ``readout="mean"`` is ``sum_k p_k * centroid_k`` — the CHARGE-closing
    estimate. ``readout="mode"`` takes the modal bin's centroid (bright-pixel
    fidelity).

    Never apply ``sinh`` to the result: the centroids are already an expectation
    of ``sinh``, so ``sinh(mean)`` double-applies it and reads back ~31% of the
    charge. That is exactly the bug this function exists to make unavailable —
    feed the output to ``_rows_from_grid(..., space="ratio")``.
    """
    x = np.asarray(logits, np.float32)
    x = x - x.max(-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(-1, keepdims=True)
    c = np.asarray(centroids, np.float32)[np.asarray(cell_band, np.int64)]  # (n_cells, K)
    if readout == "mean":
        return np.einsum("csk,ck->cs", p, c).astype(np.float32)
    if readout == "mode":
        k = p.argmax(-1)                                   # (n_cells, n_slot)
        return np.take_along_axis(c[:, None, :], k[..., None], -1)[..., 0].astype(np.float32)
    raise ValueError(f"readout must be 'mean' or 'mode', got {readout!r}")


# ---- model-side key contract ----------------------------------------------

#: ``wire_pos`` normaliser of the FM's wire feature. The value is the max wire
#: count over the production wire planes, so it is derivable from a shard
#: (``_meta['n_wires'].max()``) — but a model trained against 1969 must keep
#: seeing 1969, or every FiLM wire feature shifts.
NW_MAX = 1969.0

#: ``assemble`` output name -> the name the FM gathers from its batch.
FM_RENAME = {"cell_band": "band_id", "cell_gid": "plane_id",
             "cell_t": "t_phys", "cell_wire": "wire_pos"}


def to_fm(tok: dict, *, nw_max: float = NW_MAX) -> dict:
    """Tokenizer output -> the key names the FM consumes.

    ``assemble`` speaks in cells (``cell_*``); the model gathers
    ``band_id``/``plane_id``/``t_phys``/``wire_pos`` plus a ``wirefeat`` derived
    here.

    Pass ``nw_max=meta['n_wires'].max()`` to derive the normaliser from the
    corpus instead of the trained-model constant.
    """
    out = {FM_RENAME.get(k, k): v for k, v in tok.items() if k != "_meta"}
    wp = np.asarray(out["wire_pos"], np.float32)
    out["wirefeat"] = (wp / np.float32(nw_max))[:, None].astype(np.float32)
    return out


# ---- transform wrapper (helix owns the whole tokenizer) --------------------


def _name_hash(name: str) -> int:
    """Stable 32-bit digest of an event name.

    Not ``hash(name)``: Python salts *string* hashing per process
    (PYTHONHASHSEED), so a seed derived from it would change on every fresh
    interpreter — a resumed run, a re-run, and each DDP rank would draw different
    augmentations for the same event. blake2b is stable across processes,
    machines, and versions.
    """
    import hashlib
    return int.from_bytes(
        hashlib.blake2b(name.encode(), digest_size=4).digest(), "big")


class CoeffTokenize:
    """Coeff rows -> patch tokens, as a pimm-data-compatible transform.

    Lives in helix, not pimm: helix owns the tokenizer's semantics (patch
    geometry, band selection, RoPE time coords, normalisation), so the class and
    its config belong with the logic — a change to the token layout should touch
    ONE repo. This needs no pimm-data import, because the transform protocol is
    duck-typed: any ``fn(data_dict) -> data_dict`` with a ``scope`` works.

    Registration is the consumer's business — pimm (or a recipe) does::

        from pimm_data.transform import TRANSFORMS
        from helix.model.tokenize import CoeffTokenize
        TRANSFORMS.register_module(module=CoeffTokenize)

    so ``dict(type='CoeffTokenize', ...)`` resolves in a config, while helix
    itself stays dependency-free.

    Runs per event in a DataLoader worker (``scope='sample'``), i.e. in the HEAD
    of the transform list before the terminal ``Collect``. That spends CPU, which
    is idle during training, rather than GPU, which is not — and the shard
    metadata it needs travels with the sample.
    """

    scope = "sample"

    def __init__(self, part="coeff", cfg=None, dead_frac=0.0, seed=None, gids=None,
                 n_wires=None, band_lengths=None, norm_sigma=None):
        cfg = _require_cfg(cfg)
        self.part = part
        self.cfg = patch_config(cfg)
        self.dead_frac = float(dead_frac)
        self.seed = seed
        self._override = dict(gids=gids, n_wires=n_wires,
                              band_lengths=band_lengths, norm_sigma=norm_sigma)

    def _meta(self, sub):
        """Shard metadata: explicit constructor args win, else the sample's
        ``_meta`` (surfaced by the dataset from the shard ``/config``)."""
        meta = dict(sub.get("_meta") or {})
        for k, v in self._override.items():
            if v is not None:
                meta[k] = v
        missing = [k for k in ("gids", "n_wires", "band_lengths", "norm_sigma")
                   if meta.get(k) is None]
        if missing:
            raise KeyError(
                f"CoeffTokenize needs {missing} — pass them to the constructor or "
                "have the dataset put them in sample['{}']['_meta']".format(self.part))
        return meta

    def __call__(self, data):
        sub = data[self.part]
        m = self._meta(sub)
        rng = None if self.seed is None else np.random.default_rng(
            self.seed ^ _name_hash(data.get("name", "")))
        tok = assemble(
            sub["band"], sub["plane_gid"], sub["wire"], sub["tau"],
            np.asarray(sub["value"]).reshape(-1),
            gids=m["gids"], n_wires=m["n_wires"], band_lengths=m["band_lengths"],
            norm_sigma=m["norm_sigma"], cfg=self.cfg,
            dead_frac=self.dead_frac, rng=rng, wire_valid=m.get("wire_valid"))
        n_cells = tok.pop("n_cells")
        # Emit the names the model gathers, not assemble()'s cell_* vocabulary.
        # It must run HERE, before the terminal Collect — Collect selects by
        # exact key name and (in its multi-part `parts=` form) prefixes to
        # `<part>_<key>`, so it can neither rename nor be renamed after. The
        # recipe must therefore use the single-part `Collect(part=...)` form,
        # which leaves these keys unprefixed and matching the model.
        out = to_fm(tok)
        # n_cells stays OUT of the batch on purpose: post-collate it is exactly
        # `B["plane_id"].shape[0]` (the cell row-space length), and a python int
        # would not survive collate as a per-event value anyway. The geometry and
        # the band lengths ride along for a later per-event transform that maps
        # cells to tokens (helix.data.denoise.DenoiseRows); `_meta` never reaches a batch.
        out["_meta"] = dict(n_cells=n_cells, n_slot=self.cfg.n_slot, cfg=self.cfg,
                            band_lengths=np.asarray(m["band_lengths"]))
        data[self.part] = out
        return data
