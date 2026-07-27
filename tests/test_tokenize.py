"""The packaged tokenizer must reproduce the research one EXACTLY.

``helix.tokenize.assemble`` is a port of
``research/coeff_foundation_model/vit_tpc.py::assemble_tpc_band`` — the tokenizer
the FM was actually trained with (``fm/data.py`` calls it; ``fm/model.py``
consumes ``inp``/``occ``). The reference below is a transcription of that
function operating on the OLD data convention (pre-scaled ``val``, packed
``idx``), so a divergence in either the patch geometry or the normalisation shows
up as a failed array comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

from helix.tokenize import assemble, PatchConfig

# the research constants (star_tpc / vit_tpc / star_model)
LENS_T = np.array([271, 271, 542, 1084])
LEV_T = np.array([4, 4, 3, 2])
DELTA_T = np.array([-2.38, 0.62, 0.75, 0.50])
TOFF = np.array([-17.4, 2.6, 5.5], np.float32)
SIGMA = 2.6
PW, PT = 16, 8
N_SLOT = PW * PT


def _reference(band, gid, wire, idx, val, target, nw_by_gid):
    """Transcription of vit_tpc.assemble_tpc_band (old convention)."""
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
    return dict(occ=occ, inp=inp, tgt=tgt, valid=valid, cell=cell, slot=slot,
                n_cells=n_cells, cell_band=cell_band, cell_gid=cell_gid,
                cell_t=cell_t, cell_wire=(cell_wb * PW).astype(np.float32))


def _rows(seed=0, n=4000, gids=(0, 1, 2, 3, 4, 5), nw=1969):
    """Random coefficient rows + the per-(gid,band) sigma table."""
    rng = np.random.default_rng(seed)
    band = rng.integers(0, 4, n)
    gid = rng.choice(np.asarray(gids), n)
    wire = rng.integers(0, nw, n)
    tau = np.array([rng.integers(0, LENS_T[b]) for b in band])
    sigma = (rng.uniform(0.8, 4.0, (len(gids), 4))).astype(np.float32)
    raw = (rng.standard_normal(n) * 6.0).astype(np.float32)
    raw_clean = (raw + rng.standard_normal(n) * 0.5).astype(np.float32)
    return band, gid, wire, tau, raw, raw_clean, sigma


def test_matches_research_tokenizer_exactly():
    gids = np.array([0, 1, 2, 3, 4, 5])
    band, gid, wire, tau, raw, raw_clean, sigma = _rows()
    nw = np.full(len(gids), 1969, np.int64)

    got = assemble(band, gid, wire, tau, raw, gids=gids, n_wires=nw,
                   band_lengths=LENS_T, norm_sigma=sigma, value_clean=raw_clean)

    # OLD convention: cache stored val = raw * SIGMA/sigma_tab, packed idx.
    row_sigma = sigma[gid, band]
    old_val = raw * (SIGMA / row_sigma)
    old_clean = raw_clean * (SIGMA / row_sigma)
    idx = wire * LENS_T[band] + tau
    ref = _reference(band, gid, wire, idx, old_val, old_clean,
                     nw_by_gid=np.full(len(gids), 1969, np.int64))

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
