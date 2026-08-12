"""The tokenizer must be invertible.

`assemble` maps coefficient rows to a (cell, slot) grid. Until now there was no
way back, so nothing could turn model output into coefficients — which is the
first step of every reconstruction metric (tokens -> rows -> iDWT -> charge).
helix owns both directions: splitting a transform from its inverse across repos
would put half the representation in pimm.

Two entry points, deliberately separate:

  detokenize          exact inverse, occupancy KNOWN. Verifiable, so it is what
                      pins the other one.
  decode_prediction   model output, occupancy PREDICTED from a logit. Cannot be
                      checked against ground truth, so it is built on the first.
"""

import numpy as np
import pytest

from helix.model.tokenize import (assemble, detokenize, decode_prediction,
                            PatchConfig, sigma_for_rows)

GIDS = np.array([0, 1, 2, 4, 5], np.int32)          # deliberately NON-contiguous
# Deliberately NOT multiples of pw=16 / pt=8: partial blocks at the edges are
# what make `valid` non-trivial. With round dimensions every slot is valid and
# the respect_valid test silently proves nothing.
N_WIRES = np.array([70, 65, 50, 63, 45], np.int32)
BAND_LENGTHS = np.array([20, 18, 33, 70, 130], np.int64)
CFG = PatchConfig()


def _rows(seed=0, n=400):
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
    band, gid, wire, tau, val = _rows()
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
    band, gid, wire, tau, val = _rows(seed=3)
    ns = _norm_sigma()
    tok = assemble(band, gid, wire, tau, val, gids=GIDS, n_wires=N_WIRES,
                   band_lengths=BAND_LENGTHS, norm_sigma=ns, cfg=CFG)
    back = detokenize(tok, gids=GIDS, norm_sigma=ns, cfg=CFG)
    o = np.lexsort(np.stack([band, gid, wire, tau])[::-1])
    np.testing.assert_allclose(back["value"][_sorted(back)], val[o], rtol=1e-5, atol=1e-4)


def test_non_contiguous_gids_survive_the_round_trip():
    """gid 3 is absent. norm_sigma is row-indexed by POSITION in gids, so a
    naive norm_sigma[gid] would denormalise against the wrong plane."""
    band, gid, wire, tau, val = _rows(seed=5)
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
    band, gid, wire, tau, val = _rows(seed=7)
    band = band.copy(); band[:50] = CFG.n_bands          # D1: dropped
    ns = _norm_sigma()
    tok = assemble(band, gid, wire, tau, val, gids=GIDS, n_wires=N_WIRES,
                   band_lengths=BAND_LENGTHS, norm_sigma=ns, cfg=CFG)
    back = detokenize(tok, gids=GIDS, norm_sigma=ns, cfg=CFG)
    assert int(back["band"].max()) < CFG.n_bands
    assert len(back["band"]) == int((band < CFG.n_bands).sum())


# ---- prediction decoding --------------------------------------------------

def test_decode_prediction_threshold_selects_slots():
    band, gid, wire, tau, val = _rows(seed=11)
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
    band, gid, wire, tau, val = _rows(seed=13)
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
    band, gid, wire, tau, val = _rows(seed=17)
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
    that reads as a scientific result. The research implementation kept its own
    copies of lev/delta/toff AND hard-coded LENS_T=[271,271,542,1084], a
    per-CORPUS table; they agreed with the shipped corpus by luck.

    Checked in both cell_t modes because m113 trained grid_center while the
    PatchConfig default is centroid — cell IDENTITY is the same either way, and
    that invariant is exactly what makes a per-pixel truth artifact reusable
    across checkpoints.
    """
    import numpy as np
    from helix.model.tokenize import (PatchConfig, cell_key, pixel_cells,
                                      tick_of_tau, tau_of_tick)

    rng = np.random.default_rng(5)
    band_lengths = np.array([271, 271, 542, 1084], np.int64)
    n = 4000
    for mode in ("centroid", "grid_center"):
        cfg = PatchConfig(cell_t=mode)
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
                          np.full(2, b), bl)
        assert tau.min() >= 0 and tau.max() <= hi - 1
