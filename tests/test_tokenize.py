"""The packaged tokenizer must reproduce the research one EXACTLY.

``helix.model.tokenize.assemble`` is a port of
``research/coeff_foundation_model/vit_tpc.py::assemble_tpc_band`` — the tokenizer
the FM was actually trained with (``fm/data.py`` calls it; ``fm/model.py``
consumes ``inp``/``occ``). The reference below is a transcription of that
function operating on the OLD data convention (pre-scaled ``val``, packed
``idx``), so a divergence in either the patch geometry or the normalisation shows
up as a failed array comparison.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from helix.model.tokenize import assemble, PatchConfig

# the research constants (star_tpc / vit_tpc / star_model)
LENS_T = np.array([271, 271, 542, 1084])
LEV_T = np.array([4, 4, 3, 2])
DELTA_T = np.array([-2.38, 0.62, 0.75, 0.50])
TOFF = np.array([-17.4, 2.6, 5.5], np.float32)
SIGMA = 2.6
PW, PT = 16, 8
N_SLOT = PW * PT


def _reference(band, gid, wire, idx, val, target, nw_by_gid, cellt="grid_center"):
    """Transcription of vit_tpc.assemble_tpc_band (old convention).

    ``cellt`` mirrors the research ``FM_CELLT`` env var: the default grid-centre
    branch, and the ``centroid`` branch the production configs actually set.
    """
    tau = idx % LENS_T[band]
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
    tgt = np.zeros((n_cells, N_SLOT), np.float32)
    occ[cell, slot] = True
    inp[cell, slot] = np.arcsinh(val / SIGMA)
    tgt[cell, slot] = np.arcsinh(target / SIGMA)

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
    if cellt == "centroid":                       # vit_tpc.py FM_CELLT branch
        _tp = ((tau.astype(np.float32) + DELTA_T[band]) * _DEC[band] - TOFF[gid % 3])
        _w = np.abs(val).astype(np.float32) + 1e-6
        _ws = np.zeros(n_cells, np.float32); _ts = np.zeros(n_cells, np.float32)
        np.add.at(_ws, cell, _w); np.add.at(_ts, cell, _w * _tp)
        cell_t = (_ts / np.maximum(_ws, 1e-6)).astype(np.float32)
    return dict(occ=occ, inp=inp, tgt=tgt, valid=valid, cell=cell, slot=slot,
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
    raw_clean = (raw + rng.standard_normal(n) * 0.5).astype(np.float32)
    sigma = (rng.uniform(0.8, 4.0, (len(gids), 4))).astype(np.float32)
    band, gid, wire, tau, raw, raw_clean = _unique(band, gid, wire, tau, raw, raw_clean)
    return band, gid, wire, tau, raw, raw_clean, sigma


@pytest.mark.parametrize("cellt", ["grid_center", "centroid"])
def test_matches_research_tokenizer_exactly(cellt):
    """Both time-coordinate modes must reproduce the research tokenizer.

    ``centroid`` is not an ablation: fm/configs/cent_*.yaml set ``cellt:
    centroid`` and it is what the production runs train on, so a port that only
    implemented grid-centre could not reproduce the trained model's coordinate.
    """
    gids = np.array([0, 1, 2, 3, 4, 5])
    band, gid, wire, tau, raw, raw_clean, sigma = _rows()
    nw = np.full(len(gids), 1969, np.int64)

    got = assemble(band, gid, wire, tau, raw, gids=gids, n_wires=nw,
                   band_lengths=LENS_T, norm_sigma=sigma, value_clean=raw_clean,
                   cfg=PatchConfig(cell_t=cellt))

    # OLD convention: cache stored val = raw * SIGMA/sigma_tab, packed idx.
    row_sigma = sigma[gid, band]
    old_val = raw * (SIGMA / row_sigma)
    old_clean = raw_clean * (SIGMA / row_sigma)
    idx = wire * LENS_T[band] + tau
    ref = _reference(band, gid, wire, idx, old_val, old_clean,
                     nw_by_gid=np.full(len(gids), 1969, np.int64), cellt=cellt)

    assert got["n_cells"] == ref["n_cells"]
    for k in ("cell", "slot", "cell_band", "cell_gid"):
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)
    np.testing.assert_array_equal(got["valid"], ref["valid"])
    np.testing.assert_array_equal(got["occ"].astype(bool), ref["occ"])
    np.testing.assert_allclose(got["cell_t"], ref["cell_t"], rtol=1e-6, atol=1e-4)
    np.testing.assert_array_equal(got["cell_wire"], ref["cell_wire"])
    # the normalisation identity: arcsinh(raw/sigma) == arcsinh(val_scaled/SIGMA)
    np.testing.assert_allclose(got["inp"], ref["inp"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got["tgt"], ref["tgt"], rtol=1e-5, atol=1e-5)


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
                 norm_sigma=np.ones((1, 4), np.float32))
    # and the de-duplicated version tokenizes cleanly, with the gather holding
    out = assemble(band[1:], gid[1:], wire[1:], tau[1:],
                   np.array([2., 3.], np.float32), gids=gids,
                   n_wires=np.array([64]), band_lengths=LENS_T,
                   norm_sigma=np.ones((1, 4), np.float32))
    np.testing.assert_allclose(out["inp"][out["cell"], out["slot"]], out["val"])


def test_cell_t_modes_differ_and_centroid_is_occupancy_sensitive():
    """grid_center depends only on WHICH cell; centroid on what is in it. If the
    two ever agreed the mode switch would be doing nothing."""
    gids = np.array([0, 1])
    band, gid, wire, tau, raw, _, sigma = _rows(seed=17, gids=(0, 1), nw=256)
    kw = dict(gids=gids, n_wires=np.full(2, 256), band_lengths=LENS_T,
              norm_sigma=sigma)
    a = assemble(band, gid, wire, tau, raw, cfg=PatchConfig(cell_t="grid_center"), **kw)
    b = assemble(band, gid, wire, tau, raw, cfg=PatchConfig(cell_t="centroid"), **kw)
    assert not np.allclose(a["cell_t"], b["cell_t"])
    # same cells either way — only the coordinate changes
    np.testing.assert_array_equal(a["cell_band"], b["cell_band"])
    np.testing.assert_array_equal(a["cell_gid"], b["cell_gid"])
    # centroid moves when the amplitudes move; grid_center cannot
    c = assemble(band, gid, wire, tau, raw * np.linspace(0.1, 4.0, raw.size).astype(np.float32),
                 cfg=PatchConfig(cell_t="centroid"), **kw)
    assert not np.allclose(b["cell_t"], c["cell_t"])
    d = assemble(band, gid, wire, tau, raw * np.linspace(0.1, 4.0, raw.size).astype(np.float32),
                 cfg=PatchConfig(cell_t="grid_center"), **kw)
    np.testing.assert_array_equal(a["cell_t"], d["cell_t"])
    assert PatchConfig().cell_t == "centroid", "production default must match the live run"
    with pytest.raises(ValueError, match="cell_t must be"):
        PatchConfig(cell_t="survivor_max")


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
                   norm_sigma=np.ones((2, 5), np.float32))
    assert out["band"].max() < 4 and (out["cell_band"] < 4).all()
    assert out["band"].size == int((band < 4).sum())


def test_shapes_and_slot_bounds():
    gids = np.array([0, 1, 2])
    band, gid, wire, tau, raw, _, sigma = _rows(seed=5, gids=(0, 1, 2))
    out = assemble(band, gid, wire, tau, raw, gids=gids,
                   n_wires=np.full(3, 1969), band_lengths=LENS_T, norm_sigma=sigma)
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
    band, gid, wire, tau, raw, raw_clean, sigma = _rows(seed=9, gids=(0,), nw=512)
    out = assemble(band, gid, wire, tau, raw, gids=gids, n_wires=np.array([512]),
                   band_lengths=LENS_T, norm_sigma=sigma, value_clean=raw_clean,
                   dead_frac=0.5, rng=np.random.default_rng(0))
    assert out["dead"].sum() > 0, "no wires were killed"
    ks = np.repeat(out["dead"].astype(bool), PT, axis=1)
    assert np.all(out["inp"][ks] == 0.0)          # input zeroed
    assert np.all(out["occ"][ks] == 0.0)          # and marked inactive
    assert out["tgt"][ks].any()                   # target preserved: must still predict


def test_non_contiguous_gids_use_row_lookup():
    """The research code did n_wires[gid]; with a dead plane that is wrong."""
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
                   norm_sigma=np.ones((5, 4), np.float32))
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
        "coeff_clean": dict(band=band, plane_gid=gid, wire=wire, tau=tau,
                            value=rng.standard_normal((n, 1)).astype(np.float32)),
    }
    out = CoeffTokenize()(dict(sample))
    cfg = PatchConfig()
    assert out["coeff"]["inp"].shape[1] == cfg.n_slot
    assert out["coeff"]["_meta"]["n_slot"] == cfg.n_slot
    assert "coeff_clean" not in out          # folded into tgt
    assert out["coeff"]["tgt"].any()
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
        CoeffTokenize()(sample)


def test_to_fm_supplies_every_key_the_model_gathers():
    """`fm/model.py` gathers band_id, plane_id, t_phys, wire_pos, wirefeat, inp,
    occ, valid, cell, slot, target, tgt. The tokenizer mirrors vit_tpc's cell_*
    vocabulary and emits NO wirefeat, so without this adapter FMModel.forward
    raises KeyError('band_id') on its first access (model.py:266)."""
    from helix.model.tokenize import to_fm, NW_MAX

    gids = np.array([0, 1])
    band, gid, wire, tau, raw, raw_clean, sigma = _rows(seed=21, gids=(0, 1), nw=512)
    tok = assemble(band, gid, wire, tau, raw, gids=gids, n_wires=np.full(2, 512),
                   band_lengths=LENS_T, norm_sigma=sigma, value_clean=raw_clean)
    B = to_fm(tok)

    needed = {"band_id", "plane_id", "t_phys", "wire_pos", "wirefeat",
              "inp", "occ", "valid", "cell", "slot", "target", "tgt"}
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
