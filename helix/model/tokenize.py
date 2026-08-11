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
    cell_t: str = "centroid"                          # 'centroid' | 'grid_center'
    sigma_norm: float = 2.6                           # research SIGMA; only the
    # centroid weight sees it, and only through its 1e-6 floor (it cancels in the
    # weighted mean). Kept so the weight matches vit_tpc bit for bit.

    @property
    def n_slot(self) -> int:
        return self.pw * self.pt

    def __post_init__(self):
        if self.cell_t not in ("centroid", "grid_center"):
            raise ValueError(
                f"cell_t must be 'centroid' or 'grid_center', got {self.cell_t!r}")



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
    cfg = cfg or PatchConfig()
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

    # One sigma gather serves both the token value and the centroid weight. The
    # weight must be the PRE-arcsinh ratio: vit_tpc weighted by |val| where its
    # val was `raw * SIGMA/sigma` (the cache stored the scaled value and arcsinh'd
    # it later), so weighting by the arcsinh'd token instead would compress large
    # amplitudes and shift every centroid.
    sig_row = np.maximum(sigma_for_rows(plane_gid, band, gids, norm_sigma), 1e-6)
    ratio = (np.asarray(value, np.float32) / sig_row).astype(np.float32)
    val = np.arcsinh(ratio).astype(np.float32)          # == normalize_values(...)
    target = (np.arcsinh(np.asarray(clean, np.float32) / sig_row).astype(np.float32)
              if clean is not None else np.zeros_like(val))

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

    # RoPE time coord, TOFF-corrected so planes share a zero. Two modes:
    #
    # 'grid_center' — geometric centre of the patch's tick range. Depends only on
    #   WHICH cell it is, never on its contents. This replaced the original
    #   survivor-MAX, which was band-width biased (coarse patches span 128 ticks
    #   vs 32, so max pushed them ~100 ticks late), occupancy-dependent, and
    #   zero-clamped negative A4 times.
    # 'centroid'   — amplitude-weighted MEAN of the surviving coefficients' drift
    #   times. Debiased survivor-max: it keeps the sub-patch timing that
    #   cross-plane triangulation uses, without the band-width bias (the max was
    #   the biased part, not the use of survivors). Measurably better — the
    #   research probe scores 3D 0.60 vs 0.42 for grid-centre — and it is what the
    #   production runs train on (fm/configs/cent_*.yaml set cellt: centroid).
    dec = (1 << np.asarray(cfg.lev, np.int64)).astype(np.float32)
    toff = np.asarray(cfg.toff, np.float32)
    if cfg.cell_t == "centroid":
        tp = ((tau.astype(np.float32) + np.asarray(cfg.delta, np.float32)[band]) * dec[band]
              - toff[plane_gid % 3])
        w = (np.float32(cfg.sigma_norm) * np.abs(ratio)).astype(np.float32) + 1e-6
        ws = np.zeros(n_cells, np.float32); ts = np.zeros(n_cells, np.float32)
        np.add.at(ws, cell, w); np.add.at(ts, cell, w * tp)
        cell_t = (ts / np.maximum(ws, 1e-6)).astype(np.float32)
    else:
        center_tau = cell_tb.astype(np.float32) * pt + pt / 2.0
        cell_t = ((center_tau + np.asarray(cfg.delta, np.float32)[cell_band]) * dec[cell_band]
                  - toff[cell_gid % 3]).astype(np.float32)

    return dict(
        band=band, val=val, target=target, cell=cell.astype(np.int64),
        slot=slot.astype(np.int64), n_cells=n_cells,
        occ=occ.astype(np.float32), inp=inp, tgt=tgt, valid=valid,
        dead=dead.astype(np.float32), cell_band=cell_band, cell_gid=cell_gid,
        cell_t=cell_t, cell_wire=(cell_wb * pw).astype(np.float32),
        # Integer block indices. cell_wire carries wb*pw already, but as float32
        # for RoPE; cell_tb has no float twin at all, because `cell_t` is a
        # physical time that in the default 'centroid' mode is a WEIGHTED MEAN
        # and therefore not invertible. Without these two the token grid cannot
        # be mapped back to (wire, tau) — see `detokenize`.
        cell_wb=cell_wb, cell_tb=cell_tb,
    )


# ---- the inverse: tokens -> coefficient rows -------------------------------

def _rows_from_grid(occ_mask, values, cell_band, cell_gid, cell_wb, cell_tb, *,
                    gids, norm_sigma, cfg):
    """Shared core: an occupancy mask over the (cell, slot) grid -> coeff rows.

    ``(cell, slot)`` is a bijection with ``(plane_gid, band, wire, tau)`` given
    the per-cell block indices, so this inverts the scatter exactly. What differs
    between the two public entry points is only WHERE the mask comes from: known
    occupancy (``detokenize``) or predicted occupancy (``decode_prediction``).
    """
    pw, pt = cfg.pw, cfg.pt
    cell, slot = np.nonzero(np.asarray(occ_mask))
    band = np.asarray(cell_band, np.int64)[cell]
    plane_gid = np.asarray(cell_gid, np.int64)[cell]
    wire = np.asarray(cell_wb, np.int64)[cell] * pw + slot // pt
    tau = np.asarray(cell_tb, np.int64)[cell] * pt + slot % pt
    # undo arcsinh(raw / sigma): the token value is normalised per (plane, band)
    sig = np.maximum(sigma_for_rows(plane_gid, band, gids, norm_sigma), 1e-6)
    value = (np.sinh(np.asarray(values, np.float32)[cell, slot]) * sig).astype(np.float32)
    return dict(band=band.astype(np.uint8), plane_gid=plane_gid.astype(np.int32),
                wire=wire.astype(np.int32), tau=tau.astype(np.int32), value=value)


def detokenize(tok, *, gids, norm_sigma, cfg=PatchConfig(), values_key="inp"):
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
    return _rows_from_grid(np.asarray(tok["occ"]).astype(bool), tok[values_key],
                           tok["cell_band"], tok["cell_gid"],
                           tok["cell_wb"], tok["cell_tb"],
                           gids=gids, norm_sigma=norm_sigma, cfg=cfg)


def decode_prediction(occ_logit, values, tok, *, gids, norm_sigma,
                      cfg=PatchConfig(), threshold=0.0, respect_valid=True):
    """Model output -> coefficient rows.

    ``occ_logit`` and ``values`` are the FM's two heads over the ``(cell, slot)``
    grid. A coefficient is emitted where the occupancy logit exceeds
    ``threshold`` (0.0 == probability 0.5) and, unless ``respect_valid=False``,
    where the slot is geometrically real — a slot past the plane's wire count or
    the band's length cannot hold a coefficient no matter what the head says.

    For a categorical value head, pass the decoded bin centres as ``values``;
    this function does not know how a head parameterises its value.
    """
    occ = np.asarray(occ_logit) > threshold
    if respect_valid and "valid" in tok:
        occ = occ & np.asarray(tok["valid"]).astype(bool)
    return _rows_from_grid(occ, values, tok["cell_band"], tok["cell_gid"],
                           tok["cell_wb"], tok["cell_tb"],
                           gids=gids, norm_sigma=norm_sigma, cfg=cfg)


# ---- model-side key contract ----------------------------------------------

#: ``wire_pos`` normaliser used by the trained FM (research ``fm/data.py``). The
#: value is the max wire count over the production wire planes, so it is
#: derivable from a shard (``_meta['n_wires'].max()``) — but a model trained
#: against 1969 must keep seeing 1969, or every FiLM wire feature shifts.
NW_MAX = 1969.0

#: ``assemble`` output name -> the name ``fm/model.py`` gathers from its batch.
FM_RENAME = {"cell_band": "band_id", "cell_gid": "plane_id",
             "cell_t": "t_phys", "cell_wire": "wire_pos"}


def to_fm(tok: dict, *, nw_max: float = NW_MAX) -> dict:
    """Tokenizer output -> the key names ``fm/model.py`` consumes.

    The tokenizer mirrors ``vit_tpc.assemble_tpc_band``'s vocabulary
    (``cell_*``); the model gathers ``band_id``/``plane_id``/``t_phys``/
    ``wire_pos`` plus a ``wirefeat`` that exists in neither. The research
    pipeline bridged that in ``fm/data.py::_to_fm``; without an equivalent,
    ``FMModel.forward`` raises ``KeyError: 'band_id'`` on its first access.

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

    This used to be ``hash(name) & 0xFFFFFFFF``. Python salts *string* hashing
    per process (PYTHONHASHSEED), so the seed derived from it changed on every
    fresh interpreter: a resumed run, a re-run, and each DDP rank all drew
    different masks for the same event, while the code reads as though the seed
    is a deterministic function of the event. blake2b is stable across processes,
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

    def __init__(self, part="coeff", clean_part="coeff_clean", out_part=None,
                 cfg=None, dead_frac=0.0, seed=None, gids=None, n_wires=None,
                 band_lengths=None, norm_sigma=None, fm_names=True):
        self.part = part
        self.clean_part = clean_part
        self.out_part = out_part or part
        self.cfg = cfg if isinstance(cfg, PatchConfig) else PatchConfig(**(cfg or {}))
        self.dead_frac = float(dead_frac)
        self.seed = seed
        self.fm_names = bool(fm_names)
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
        clean = data.get(self.clean_part)
        rng = None if self.seed is None else np.random.default_rng(
            self.seed ^ _name_hash(data.get("name", "")))
        tok = assemble(
            sub["band"], sub["plane_gid"], sub["wire"], sub["tau"],
            np.asarray(sub["value"]).reshape(-1),
            gids=m["gids"], n_wires=m["n_wires"], band_lengths=m["band_lengths"],
            norm_sigma=m["norm_sigma"], cfg=self.cfg,
            value_clean=(None if clean is None
                         else np.asarray(clean["value"]).reshape(-1)),
            dead_frac=self.dead_frac, rng=rng)
        n_cells = tok.pop("n_cells")
        # Emit the names fm/model.py gathers, not assemble()'s cell_* vocabulary.
        # Without this the first batch dies on B["band_id"] (model.py:266), and
        # `to_fm` sat unused. It must run HERE, before the terminal Collect —
        # Collect selects by exact key name and (in its multi-part `parts=` form)
        # prefixes to `<part>_<key>`, so it can neither rename nor be renamed
        # after. The recipe must therefore use the single-part `Collect(part=...)`
        # form, which leaves these keys unprefixed and matching the model.
        out = to_fm(tok) if self.fm_names else dict(tok)
        # n_cells stays OUT of the batch on purpose: post-collate it is exactly
        # `B["plane_id"].shape[0]` (the cell row-space length), and a python int
        # would not survive collate as a per-event value anyway.
        out["_meta"] = dict(n_cells=n_cells, n_slot=self.cfg.n_slot)
        data[self.out_part] = out
        if self.clean_part in data and self.clean_part != self.out_part:
            del data[self.clean_part]        # its values are folded into tgt
        return data
