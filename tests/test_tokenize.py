"""helix.model.tokenize: rows to tokens and back, and the cell key and event-name hash it packs
(regressions that once produced a wrong number without failing)."""
import os
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from helix.model.tokenize import assemble, PatchConfig
from helix.model.tokenize import (assemble, detokenize, decode_prediction,
                            PatchConfig, sigma_for_rows)


# ---- rows to tokens ----------------------------------------------------------------------

# The patch tokenizer: geometry, normalisation, the transform and its keys.
#
# ``_expected`` below states the patch geometry independently of ``assemble``
# (its own key packing, slot numbering, validity and time coordinate), so a
# divergence in either the geometry or the normalisation shows up as a failed
# array comparison.

LENS_T = np.array([271, 271, 542, 1084])
LEV_T = np.array([4, 4, 3, 2])
DELTA_T = np.array([-2.38, 0.62, 0.75, 0.50])
TOFF = np.array([-17.4, 2.6, 5.5], np.float32)
PW, PT = 16, 8
N_SLOT = PW * PT


def _expected(band, gid, wire, tau, raw, sigma, nw_by_gid):
    """The tokens of PW x PT patches over (wire, tau), per (gid, band)."""
    wb, tb = wire // PW, tau // PT
    key = (gid.astype(np.int64) << 40) | (band.astype(np.int64) << 36) \
        | (wb.astype(np.int64) << 18) | tb
    uniq, cell = np.unique(key, return_inverse=True)
    n_cells = len(uniq)
    slot = (wire % PW) * PT + (tau % PT)
    cell_band = ((uniq >> 36) & 0xF).astype(np.int64)
    cell_gid = (uniq >> 40).astype(np.int64)
    cell_wb = ((uniq >> 18) & 0x3FFFF).astype(np.int64)
    cell_tb = (uniq & 0x3FFFF).astype(np.int64)

    occ = np.zeros((n_cells, N_SLOT), bool)
    inp = np.zeros((n_cells, N_SLOT), np.float32)
    occ[cell, slot] = True
    inp[cell, slot] = np.arcsinh(raw / sigma[gid, band])

    nw = nw_by_gid[cell_gid]
    Lb = LENS_T[cell_band]
    wi = np.arange(PW)[None, :]
    ti = np.arange(PT)[None, :]
    wok = (cell_wb[:, None] * PW + wi) < nw[:, None]
    tok = (cell_tb[:, None] * PT + ti) < Lb[:, None]
    valid = (wok[:, :, None] & tok[:, None, :]).reshape(n_cells, N_SLOT)

    _DEC = (1 << LEV_T).astype(np.float32)
    _center_tau = cell_tb.astype(np.float32) * PT + PT / 2.0
    cell_t = ((_center_tau + DELTA_T[cell_band]) * _DEC[cell_band]
              - TOFF[cell_gid % 3]).astype(np.float32)
    return dict(occ=occ, inp=inp, valid=valid, cell=cell, slot=slot,
                n_cells=n_cells, cell_band=cell_band, cell_gid=cell_gid,
                cell_t=cell_t, cell_wire=(cell_wb * PW).astype(np.float32))


def _unique(band, gid, wire, tau, *rest):
    """Drop rows sharing a ``(gid, band, wire, tau)``.

    A real CoeffEvent cannot contain duplicate coordinates — it is built by
    compacting the nonzeros of a dense band, which emits each position once — so
    a fixture that collides is unfaithful, not a bug to tolerate. ``assemble``
    rejects duplicates (they would be silently overwritten in the token grid).
    """
    key = (gid.astype(np.int64) << 44) | (band.astype(np.int64) << 40) \
        | (wire.astype(np.int64) << 20) | tau.astype(np.int64)
    _, keep = np.unique(key, return_index=True)
    keep = np.sort(keep)
    return (band[keep], gid[keep], wire[keep], tau[keep]) + tuple(r[keep] for r in rest)


def _rows(seed=0, n=4000, gids=(0, 1, 2, 3, 4, 5), nw=1969):
    """Random coefficient rows + the per-(gid,band) sigma table."""
    rng = np.random.default_rng(seed)
    band = rng.integers(0, 4, n)
    gid = rng.choice(np.asarray(gids), n)
    wire = rng.integers(0, nw, n)
    tau = np.array([rng.integers(0, LENS_T[b]) for b in band])
    raw = (rng.standard_normal(n) * 6.0).astype(np.float32)
    sigma = (rng.uniform(0.8, 4.0, (len(gids), 4))).astype(np.float32)
    band, gid, wire, tau, raw = _unique(band, gid, wire, tau, raw)
    return band, gid, wire, tau, raw, sigma


def test_tokens_match_an_independent_statement_of_the_geometry():
    gids = np.array([0, 1, 2, 3, 4, 5])
    band, gid, wire, tau, raw, sigma = _rows()
    nw = np.full(len(gids), 1969, np.int64)

    got = assemble(band, gid, wire, tau, raw, gids=gids, n_wires=nw,
                   band_lengths=LENS_T, norm_sigma=sigma, cfg=PatchConfig())
    ref = _expected(band, gid, wire, tau, raw, sigma, nw_by_gid=nw)

    assert got["n_cells"] == ref["n_cells"]
    for k in ("cell", "slot", "cell_band", "cell_gid"):
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)
    np.testing.assert_array_equal(got["valid"], ref["valid"])
    np.testing.assert_array_equal(got["occ"].astype(bool), ref["occ"])
    np.testing.assert_allclose(got["cell_t"], ref["cell_t"], rtol=1e-6, atol=1e-4)
    np.testing.assert_array_equal(got["cell_wire"], ref["cell_wire"])
    np.testing.assert_allclose(got["inp"], ref["inp"], rtol=1e-5, atol=1e-5)
    # noisy-only: the target is the input
    np.testing.assert_array_equal(got["tgt"], got["inp"])


def test_duplicate_coordinates_are_rejected_not_silently_overwritten():
    """Two rows at the same (gid, band, wire, tau) land in one (cell, slot). The
    scatter is last-write-wins, so one coefficient would vanish silently and
    ``inp[cell, slot] == val`` — the invariant the FM's loss gathers on — would
    stop holding for that row with no error anywhere."""
    gids = np.array([0])
    band = np.array([0, 0, 1]); gid = np.array([0, 0, 0])
    wire = np.array([3, 3, 7]); tau = np.array([5, 5, 9])       # rows 0 and 1 collide
    with pytest.raises(ValueError, match="duplicate coefficient coordinates"):
        assemble(band, gid, wire, tau, np.array([1., 2., 3.], np.float32),
                 gids=gids, n_wires=np.array([64]), band_lengths=LENS_T,
                 norm_sigma=np.ones((1, 4), np.float32), cfg=PatchConfig())
    # and the de-duplicated version tokenizes cleanly, with the gather holding
    out = assemble(band[1:], gid[1:], wire[1:], tau[1:],
                   np.array([2., 3.], np.float32), gids=gids,
                   n_wires=np.array([64]), band_lengths=LENS_T,
                   norm_sigma=np.ones((1, 4), np.float32), cfg=PatchConfig())
    np.testing.assert_allclose(out["inp"][out["cell"], out["slot"]], out["val"])


def test_time_coordinate_ignores_the_cell_contents():
    """`t_phys` is the masked tokens' RoPE position, computed before masking:
    it must depend only on WHICH cell, never on the amplitudes in it."""
    gids = np.array([0, 1])
    band, gid, wire, tau, raw, sigma = _rows(seed=17, gids=(0, 1), nw=256)
    kw = dict(gids=gids, n_wires=np.full(2, 256), band_lengths=LENS_T,
              norm_sigma=sigma, cfg=PatchConfig())
    a = assemble(band, gid, wire, tau, raw, **kw)
    d = assemble(band, gid, wire, tau, raw * np.linspace(0.1, 4.0, raw.size).astype(np.float32), **kw)
    np.testing.assert_array_equal(a["cell_t"], d["cell_t"])
    # the retired switch that chose it is ignored where a run recorded it, not refused
    from helix.model.tokenize import patch_config
    assert patch_config(dict(cell_t="grid_center", pw=16, lev=[4, 4, 3, 2])) == PatchConfig()
    with pytest.raises(TypeError):
        PatchConfig(cell_t="grid_center")


def test_drops_d1_like_the_fm():
    """The FM tokenized 4 bands; the corpus keeps 5. Band 4 must be dropped."""
    gids = np.array([0, 1])
    n = 500
    rng = np.random.default_rng(3)
    band = rng.integers(0, 5, n)                    # includes D1
    gid = rng.choice(gids, n)
    wire = rng.integers(0, 100, n)
    bl = np.array([271, 271, 542, 1084, 2168])
    tau = np.array([rng.integers(0, bl[b]) for b in band])
    out = assemble(band, gid, wire, tau, rng.standard_normal(n).astype(np.float32),
                   gids=gids, n_wires=np.full(2, 100), band_lengths=bl,
                   norm_sigma=np.ones((2, 5), np.float32), cfg=PatchConfig())
    assert out["band"].max() < 4 and (out["cell_band"] < 4).all()
    assert out["band"].size == int((band < 4).sum())


def test_shapes_and_slot_bounds():
    gids = np.array([0, 1, 2])
    band, gid, wire, tau, raw, sigma = _rows(seed=5, gids=(0, 1, 2))
    out = assemble(band, gid, wire, tau, raw, gids=gids,
                   n_wires=np.full(3, 1969), band_lengths=LENS_T, norm_sigma=sigma, cfg=PatchConfig())
    cfg = PatchConfig()
    assert out["inp"].shape == (out["n_cells"], cfg.n_slot)
    assert out["occ"].shape == out["inp"].shape == out["tgt"].shape
    assert out["valid"].shape == out["inp"].shape
    assert 0 <= out["slot"].min() and out["slot"].max() < cfg.n_slot
    assert out["cell"].max() < out["n_cells"]
    # every occupied slot must also be a valid slot
    assert bool((out["valid"] | ~out["occ"].astype(bool)).all())


def test_dead_wire_augmentation_zeroes_input_not_target():
    gids = np.array([0])
    band, gid, wire, tau, raw, sigma = _rows(seed=9, gids=(0,), nw=512)
    out = assemble(band, gid, wire, tau, raw, gids=gids, n_wires=np.array([512]),
                   band_lengths=LENS_T, norm_sigma=sigma,
                   dead_frac=0.5, rng=np.random.default_rng(0), cfg=PatchConfig())
    assert out["dead"].sum() > 0, "no wires were killed"
    ks = np.repeat(out["dead"].astype(bool), PT, axis=1)
    assert np.all(out["inp"][ks] == 0.0)          # input zeroed
    assert np.all(out["occ"][ks] == 0.0)          # and marked inactive
    assert out["tgt"][ks].any()                   # target preserved: must still predict


def test_non_contiguous_gids_use_row_lookup():
    """n_wires is looked up by gid ROW, not gid value: with a dead plane they differ."""
    gids = np.array([0, 1, 2, 4, 5])              # plane 3 absent
    nw = np.array([100, 200, 300, 400, 500], np.int64)
    n = 400
    rng = np.random.default_rng(11)
    gid = rng.choice(gids, n)
    band = rng.integers(0, 4, n)
    wire = np.array([rng.integers(0, nw[np.searchsorted(gids, g)]) for g in gid])
    tau = np.array([rng.integers(0, LENS_T[b]) for b in band])
    out = assemble(band, gid, wire, tau, rng.standard_normal(n).astype(np.float32),
                   gids=gids, n_wires=nw, band_lengths=LENS_T,
                   norm_sigma=np.ones((5, 4), np.float32), cfg=PatchConfig())
    # every occupied slot lies inside its plane's true wire count
    for c in range(out["n_cells"]):
        g = int(out["cell_gid"][c])
        limit = int(nw[np.searchsorted(gids, g)])
        wb = int(out["cell_wire"][c])
        occ_w = np.where(out["occ"][c].reshape(PW, PT).any(1))[0]
        assert all(wb + int(w) < limit for w in occ_w), f"cell {c} gid {g} exceeds n_wires"


def test_transform_is_pimm_data_compatible_without_importing_it():
    """helix owns the whole tokenizer: the transform protocol is duck-typed
    (callable(data)->data with a `scope`), so no pimm-data import is needed and
    registration is the consumer's one-liner."""
    # the real property: importing helix must not pull pimm-data in (a source
    # grep would false-positive on the registration example in the docstring)
    import subprocess, sys as _s
    r = subprocess.run(
        [_s.executable, "-c",
         "import sys, helix.model.tokenize; "
         "print('pimm_data' in sys.modules or 'torch' in sys.modules)"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False", "helix.model.tokenize pulled in pimm-data or torch"
    from helix.model.tokenize import CoeffTokenize
    assert CoeffTokenize.scope == "sample"

    gids = np.array([0, 1])
    bl = np.array([271, 271, 542, 1084])
    rng = np.random.default_rng(2)
    n = 800
    band = rng.integers(0, 4, n); gid = rng.choice(gids, n)
    wire = rng.integers(0, 64, n)
    tau = np.array([rng.integers(0, bl[b]) for b in band])
    band, gid, wire, tau = _unique(band, gid, wire, tau)
    n = band.size
    meta = dict(gids=gids, n_wires=np.array([64, 64]), band_lengths=bl,
                norm_sigma=np.ones((2, 4), np.float32))
    sample = {
        "name": "evt0",
        "coeff": dict(band=band, plane_gid=gid, wire=wire, tau=tau,
                      value=rng.standard_normal((n, 1)).astype(np.float32), _meta=meta),
    }
    out = CoeffTokenize(cfg=dict())(dict(sample))
    cfg = PatchConfig()
    assert out["coeff"]["inp"].shape[1] == cfg.n_slot
    assert out["coeff"]["_meta"]["n_slot"] == cfg.n_slot
    occ = out["coeff"]["occ"].astype(bool)
    np.testing.assert_array_equal(out["coeff"]["tgt"][occ], out["coeff"]["inp"][occ])
    # constructor overrides win over the sample's _meta
    o2 = CoeffTokenize(cfg=dict(pw=8, pt=4), gids=gids, n_wires=np.array([64, 64]),
                       band_lengths=bl, norm_sigma=np.ones((2, 4), np.float32))(dict(sample))
    assert o2["coeff"]["inp"].shape[1] == 32


def test_transform_reports_missing_metadata_clearly():
    from helix.model.tokenize import CoeffTokenize
    sample = {"coeff": dict(band=np.array([0]), plane_gid=np.array([0]),
                            wire=np.array([0]), tau=np.array([0]),
                            value=np.zeros((1, 1), np.float32))}
    with pytest.raises(KeyError, match="gids"):
        CoeffTokenize(cfg=dict())(sample)


def test_to_fm_supplies_every_key_the_model_gathers():
    """The model gathers band_id, plane_id, t_phys, wire_pos, wirefeat, inp, occ,
    valid, tgt; assemble speaks cell_* and emits no wirefeat."""
    from helix.model.tokenize import to_fm, NW_MAX

    gids = np.array([0, 1])
    band, gid, wire, tau, raw, sigma = _rows(seed=21, gids=(0, 1), nw=512)
    tok = assemble(band, gid, wire, tau, raw, gids=gids, n_wires=np.full(2, 512),
                   band_lengths=LENS_T, norm_sigma=sigma, cfg=PatchConfig())
    B = to_fm(tok)

    needed = {"band_id", "plane_id", "t_phys", "wire_pos", "wirefeat",
              "inp", "occ", "valid", "tgt"}
    assert needed <= set(B), f"missing {sorted(needed - set(B))}"
    # renames carry the values through unchanged
    np.testing.assert_array_equal(B["band_id"], tok["cell_band"])
    np.testing.assert_array_equal(B["plane_id"], tok["cell_gid"])
    np.testing.assert_array_equal(B["t_phys"], tok["cell_t"])
    np.testing.assert_array_equal(B["wire_pos"], tok["cell_wire"])
    # wirefeat is COMPUTED, not renamed: (wire_pos / NW_MAX)[:, None]
    assert B["wirefeat"].shape == (tok["n_cells"], 1)
    np.testing.assert_allclose(B["wirefeat"][:, 0], tok["cell_wire"] / NW_MAX, rtol=1e-6)
    # the old cell_* names are gone, so a stale consumer fails loudly
    assert not ({"cell_band", "cell_gid", "cell_t", "cell_wire"} & set(B))
    # and the normaliser is overridable from the corpus
    B2 = to_fm(tok, nw_max=512.0)
    np.testing.assert_allclose(B2["wirefeat"][:, 0], tok["cell_wire"] / 512.0, rtol=1e-6)


# ---- tokens back to rows -----------------------------------------------------------------

# The tokenizer must be invertible.
#
# `assemble` maps coefficient rows to a (cell, slot) grid. Until now there was no
# way back, so nothing could turn model output into coefficients — which is the
# first step of every reconstruction metric (tokens -> rows -> iDWT -> charge).
# helix owns both directions: splitting a transform from its inverse across repos
# would put half the representation in pimm.
#
# Two entry points, deliberately separate:
#
#   detokenize          exact inverse, occupancy KNOWN. Verifiable, so it is what
#                       pins the other one.
#   decode_prediction   model output, occupancy PREDICTED from a logit. Cannot be
#                       checked against ground truth, so it is built on the first.

GIDS = np.array([0, 1, 2, 4, 5], np.int32)          # deliberately NON-contiguous
# Deliberately NOT multiples of pw=16 / pt=8: partial blocks at the edges are
# what make `valid` non-trivial. With round dimensions every slot is valid and
# the respect_valid test silently proves nothing.
N_WIRES = np.array([70, 65, 50, 63, 45], np.int32)
BAND_LENGTHS = np.array([20, 18, 33, 70, 130], np.int64)
CFG = PatchConfig()   # what every helix config trains


def _inv_rows(seed=0, n=400):
    """Random rows with UNIQUE (band, gid, wire, tau) — assemble rejects dups."""
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    while len(out) < n:
        b = int(rng.integers(0, CFG.n_bands))
        gi = int(rng.integers(0, len(GIDS)))
        w = int(rng.integers(0, N_WIRES[gi]))
        t = int(rng.integers(0, BAND_LENGTHS[b]))
        k = (b, int(GIDS[gi]), w, t)
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    a = np.array(out, np.int64)
    val = rng.normal(0, 30, size=len(a)).astype(np.float32)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], val


def _norm_sigma():
    return np.linspace(1.0, 4.0, len(GIDS) * 5, dtype=np.float32).reshape(len(GIDS), 5)


def _sorted(d):
    k = np.stack([d["band"], d["plane_gid"], d["wire"], d["tau"]]).astype(np.int64)
    return np.lexsort(k[::-1])


def test_round_trip_recovers_every_coordinate():
    band, gid, wire, tau, val = _inv_rows()
    ns = _norm_sigma()
    tok = assemble(band, gid, wire, tau, val, gids=GIDS, n_wires=N_WIRES,
                   band_lengths=BAND_LENGTHS, norm_sigma=ns, cfg=CFG)
    back = detokenize(tok, gids=GIDS, norm_sigma=ns, cfg=CFG)

    assert len(back["band"]) == len(band)
    o = np.lexsort(np.stack([band, gid, wire, tau])[::-1])
    g = _sorted(back)
    np.testing.assert_array_equal(back["band"][g], band[o].astype(np.uint8))
    np.testing.assert_array_equal(back["plane_gid"][g], gid[o].astype(np.int32))
    np.testing.assert_array_equal(back["wire"][g], wire[o].astype(np.int32))
    np.testing.assert_array_equal(back["tau"][g], tau[o].astype(np.int32))


def test_round_trip_recovers_values():
    """arcsinh/sinh in float32 — exact to rounding, not bit-identical."""
    band, gid, wire, tau, val = _inv_rows(seed=3)
    ns = _norm_sigma()
    tok = assemble(band, gid, wire, tau, val, gids=GIDS, n_wires=N_WIRES,
                   band_lengths=BAND_LENGTHS, norm_sigma=ns, cfg=CFG)
    back = detokenize(tok, gids=GIDS, norm_sigma=ns, cfg=CFG)
    o = np.lexsort(np.stack([band, gid, wire, tau])[::-1])
    np.testing.assert_allclose(back["value"][_sorted(back)], val[o], rtol=1e-5, atol=1e-4)


def test_non_contiguous_gids_survive_the_round_trip():
    """gid 3 is absent. norm_sigma is row-indexed by POSITION in gids, so a
    naive norm_sigma[gid] would denormalise against the wrong plane."""
    band, gid, wire, tau, val = _inv_rows(seed=5)
    assert 3 not in set(gid.tolist())
    ns = _norm_sigma()
    tok = assemble(band, gid, wire, tau, val, gids=GIDS, n_wires=N_WIRES,
                   band_lengths=BAND_LENGTHS, norm_sigma=ns, cfg=CFG)
    back = detokenize(tok, gids=GIDS, norm_sigma=ns, cfg=CFG)
    assert set(back["plane_gid"].tolist()) <= set(GIDS.tolist())
    o = np.lexsort(np.stack([band, gid, wire, tau])[::-1])
    np.testing.assert_allclose(back["value"][_sorted(back)], val[o], rtol=1e-5, atol=1e-4)


def test_dropped_bands_are_not_resurrected():
    """assemble keeps band < n_bands; the inverse must not invent the rest."""
    band, gid, wire, tau, val = _inv_rows(seed=7)
    band = band.copy(); band[:50] = CFG.n_bands          # D1: dropped
    ns = _norm_sigma()
    tok = assemble(band, gid, wire, tau, val, gids=GIDS, n_wires=N_WIRES,
                   band_lengths=BAND_LENGTHS, norm_sigma=ns, cfg=CFG)
    back = detokenize(tok, gids=GIDS, norm_sigma=ns, cfg=CFG)
    assert int(back["band"].max()) < CFG.n_bands
    assert len(back["band"]) == int((band < CFG.n_bands).sum())


# ---- prediction decoding --------------------------------------------------

def test_decode_prediction_threshold_selects_slots():
    band, gid, wire, tau, val = _inv_rows(seed=11)
    ns = _norm_sigma()
    tok = assemble(band, gid, wire, tau, val, gids=GIDS, n_wires=N_WIRES,
                   band_lengths=BAND_LENGTHS, norm_sigma=ns, cfg=CFG)
    occ = np.asarray(tok["occ"]).astype(bool)
    logit = np.where(occ, 5.0, -5.0).astype(np.float32)   # a perfect predictor

    got = decode_prediction(logit, tok["inp"], tok, gids=GIDS, norm_sigma=ns, cfg=CFG)
    ref = detokenize(tok, gids=GIDS, norm_sigma=ns, cfg=CFG)
    for k in ("band", "plane_gid", "wire", "tau"):
        np.testing.assert_array_equal(got[k][_sorted(got)], ref[k][_sorted(ref)])

    # everything on -> more rows; everything off -> none
    allon = decode_prediction(np.full_like(logit, 9.0), tok["inp"], tok,
                              gids=GIDS, norm_sigma=ns, cfg=CFG)
    assert len(allon["band"]) > len(ref["band"])
    alloff = decode_prediction(np.full_like(logit, -9.0), tok["inp"], tok,
                               gids=GIDS, norm_sigma=ns, cfg=CFG)
    assert len(alloff["band"]) == 0


def test_decode_prediction_respects_geometric_validity():
    """A slot past the plane's wire count or the band's length cannot hold a
    coefficient however confident the head is."""
    band, gid, wire, tau, val = _inv_rows(seed=13)
    ns = _norm_sigma()
    tok = assemble(band, gid, wire, tau, val, gids=GIDS, n_wires=N_WIRES,
                   band_lengths=BAND_LENGTHS, norm_sigma=ns, cfg=CFG)
    logit = np.full(np.asarray(tok["occ"]).shape, 9.0, np.float32)

    kept = decode_prediction(logit, tok["inp"], tok, gids=GIDS, norm_sigma=ns, cfg=CFG)
    loose = decode_prediction(logit, tok["inp"], tok, gids=GIDS, norm_sigma=ns,
                              cfg=CFG, respect_valid=False)
    assert len(kept["band"]) == int(np.asarray(tok["valid"]).sum())
    assert len(loose["band"]) > len(kept["band"])
    # every kept row is inside its plane and band
    for i, g in enumerate(GIDS):
        sel = kept["plane_gid"] == g
        assert kept["wire"][sel].max(initial=-1) < N_WIRES[i]
    for b in range(CFG.n_bands):
        sel = kept["band"] == b
        assert kept["tau"][sel].max(initial=-1) < BAND_LENGTHS[b]


def test_dead_wire_augmentation_is_not_invertible():
    """dead_frac deliberately destroys input; the inverse must not pretend
    otherwise. Documented rather than silently lossy."""
    band, gid, wire, tau, val = _inv_rows(seed=17)
    ns = _norm_sigma()
    tok = assemble(band, gid, wire, tau, val, gids=GIDS, n_wires=N_WIRES,
                   band_lengths=BAND_LENGTHS, norm_sigma=ns, cfg=CFG,
                   dead_frac=0.5, rng=np.random.default_rng(0))
    back = detokenize(tok, gids=GIDS, norm_sigma=ns, cfg=CFG)
    assert len(back["band"]) < len(band)


def test_pixel_cells_agrees_with_assemble():
    """A raw pixel must land in the cell `assemble` put its coefficients in.

    This is the probe's join. If it drifts, the probe gathers the WRONG cell's
    features — which does not crash: it returns a number near the geometry null
    that reads as a scientific result.
    """
    import numpy as np
    from helix.model.tokenize import (PatchConfig, cell_key, pixel_cells,
                                      tick_of_tau, tau_of_tick)

    rng = np.random.default_rng(5)
    band_lengths = np.array([271, 271, 542, 1084], np.int64)
    n = 4000
    cfg = PatchConfig()
    gid = rng.integers(0, 6, n)
    band = rng.integers(0, cfg.n_bands, n)
    wire = rng.integers(0, 1969, n)
    tau = np.array([rng.integers(0, band_lengths[b]) for b in band])

    tick = tick_of_tau(tau, gid, band, cfg)
    np.testing.assert_array_equal(tau_of_tick(tick, gid, band, band_lengths, cfg), tau)

    cells = pixel_cells(gid, wire, tick, band_lengths, cfg)
    assert cells.shape == (n, cfg.n_bands)
    np.testing.assert_array_equal(cells[np.arange(n), band],
                                  cell_key(gid, band, wire, tau, cfg))


def test_tau_of_tick_clips_into_the_band():
    """Out-of-range ticks must land inside the band, not produce a bogus cell."""
    import numpy as np
    from helix.model.tokenize import tau_of_tick

    bl = np.array([271, 271, 542, 1084], np.int64)
    for b, hi in enumerate(bl):
        tau = tau_of_tick(np.array([-1e6, 1e6]), np.zeros(2, np.int64),
                          np.full(2, b), bl, cfg=CFG)
        assert tau.min() >= 0 and tau.max() <= hi - 1


# ---- the event-name hash and the cell key ------------------------------------------------

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Event-name seeding: hash() is salted per process
# --------------------------------------------------------------------------

def test_name_hash_is_stable_across_processes():
    """The tokenizer's per-event seed must be a function of the event.

    It was ``hash(name) & 0xFFFFFFFF``. Python salts string hashing per
    interpreter, so a resumed run, a re-run and each DDP rank drew different
    masks for the same event while the code read as though the seed were
    deterministic. Pinned against an independent digest so a future rewrite
    cannot quietly reintroduce process-dependence.
    """
    from helix.model.tokenize import _name_hash

    for name in ("", "event_0", "run_0027575715/event_41", "événement"):
        want = int.from_bytes(
            hashlib.blake2b(name.encode(), digest_size=4).digest(), "big")
        assert _name_hash(name) == want
        assert 0 <= _name_hash(name) < 2 ** 32

    # Distinct events must not collide onto one seed.
    names = [f"event_{i}" for i in range(2000)]
    assert len({_name_hash(n) for n in names}) == len(names)


def test_no_builtin_hash_used_for_seeding():
    """Catch the pattern, not just this one call site.

    Parsed rather than grepped: a text scan also matches the docstring that
    quotes the old expression, which is documentation worth keeping.
    """
    import ast

    src = (REPO / "helix" / "model" / "tokenize.py").read_text()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "hash"]
    assert not calls, (
        f"builtin hash() called at line(s) {[n.lineno for n in calls]} — it is "
        f"salted per process; use _name_hash() for anything feeding an RNG seed")


# --------------------------------------------------------------------------
# Version: stamped into every shard, and checked across shards by pimm-data
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# cell_key is public so probes can agree with the tokenizer on cell identity
# --------------------------------------------------------------------------

def test_cell_key_round_trips():
    from helix.model.tokenize import PatchConfig, cell_key, unpack_cell_key

    cfg = PatchConfig()
    rng = np.random.default_rng(3)
    gid = rng.integers(0, 6, 5000)
    band = rng.integers(0, cfg.n_bands, 5000)
    wire = rng.integers(0, 4000, 5000)
    tau = rng.integers(0, 6000, 5000)

    k = cell_key(gid, band, wire, tau, cfg)
    g2, b2, wb2, tb2 = unpack_cell_key(k)
    np.testing.assert_array_equal(g2, gid)
    np.testing.assert_array_equal(b2, band)
    np.testing.assert_array_equal(wb2, wire // cfg.pw)
    np.testing.assert_array_equal(tb2, tau // cfg.pt)


def test_cell_key_agrees_with_assemble():
    """The reason it is public: a probe must land on the tokenizer's cells.

    If these two ever disagree, per-cell labels silently attach to the wrong
    cells — every array still has the right length, so nothing raises.
    """
    from helix.model.tokenize import PatchConfig, assemble, cell_key

    cfg = PatchConfig()
    rng = np.random.default_rng(11)
    n = 4000
    gids = np.array([0, 1, 2], np.int64)
    gid = rng.choice(gids, n)
    band = rng.integers(0, cfg.n_bands, n)
    wire = rng.integers(0, 300, n)
    tau = rng.integers(0, 400, n)
    # assemble requires unique (gid, band, wire, tau)
    coords = np.unique(np.stack([gid, band, wire, tau], 1), axis=0)
    gid, band, wire, tau = (coords[:, i].astype(np.int64) for i in range(4))

    tok = assemble(band, gid, wire, tau,
                   rng.normal(size=len(gid)).astype(np.float32),
                   gids=gids, n_wires=np.full(len(gids), 320, np.int64),
                   band_lengths=np.full(cfg.n_bands, 512, np.int64),
                   norm_sigma=np.full((len(gids), cfg.n_bands), 2.6, np.float32),
                   cfg=cfg)

    # every row's key must resolve to the cell assemble put that row in
    keys = cell_key(gid, band, wire, tau, cfg)
    uniq = np.unique(keys)
    assert tok["n_cells"] == len(uniq)
    np.testing.assert_array_equal(
        np.searchsorted(uniq, keys), tok["cell"])


