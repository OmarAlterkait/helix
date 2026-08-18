"""The two probe designs: arm construction and the slab match."""
import numpy as np
import pytest

from helix.probe.designs import mlp_designs, slab_context, triangulate_designs


def test_every_arm_carries_the_geo_floor():
    geo = np.ones((5, 3), np.float32)
    d = mlp_designs(geo, {"trained": np.zeros((5, 4), np.float32), "raw": None})
    assert set(d) == {"geo", "trained"}          # None arms are skipped
    assert d["geo"].shape == (5, 3)
    assert d["trained"].shape == (5, 7)
    np.testing.assert_array_equal(d["trained"][:, 4:], geo)


def test_arm_row_mismatch_is_refused():
    with pytest.raises(ValueError, match="rows"):
        mlp_designs(np.ones((5, 2), np.float32), {"x": np.ones((4, 2), np.float32)})


def test_slab_matches_the_other_planes_at_the_same_drift_bin():
    """Rows in planes 0,1,2 of one volume at the same tick must see each other."""
    plane = np.array([0, 1, 2])
    tick = np.array([100.0, 101.0, 102.0])          # same tbin=8 bucket
    feats = np.array([[1., 0.], [0., 1.], [1., 1.]], np.float32)
    wire = np.array([200.0, 400.0, 600.0])
    ctx, ctxw, hit = slab_context(plane, tick, feats, wire, np.zeros(3), tbin=8)
    assert hit.sum() == 6                            # every row finds both partners
    # plane 0's partners are planes 1 and 2
    np.testing.assert_allclose(ctx[0, :2], [0., 1.])
    np.testing.assert_allclose(ctx[0, 2:], [1., 1.])
    np.testing.assert_allclose(ctxw[0], [400 / 2000.0, 600 / 2000.0])


def test_slab_does_not_match_across_volumes_or_drift_bins():
    plane = np.array([0, 4])                         # different volumes
    ctx, _, hit = slab_context(plane, np.array([10.0, 10.0]),
                               np.ones((2, 2), np.float32), np.array([1.0, 2.0]),
                               np.zeros(2))
    assert hit.sum() == 0

    plane = np.array([0, 1])
    _, _, hit2 = slab_context(plane, np.array([10.0, 900.0]),   # far apart in time
                              np.ones((2, 2), np.float32), np.array([1.0, 2.0]),
                              np.zeros(2))
    assert hit2.sum() == 0


def test_triangulate_arms_nest_correctly():
    """solo must be exactly the single-plane design on the same rows, so the
    three arms are comparable."""
    n, n_bands, band_d, gd = 6, 4, 5, 3
    fd = n_bands * band_d
    geo = np.random.default_rng(0).normal(size=(n, gd)).astype(np.float32)
    own = np.random.default_rng(1).normal(size=(n, fd)).astype(np.float32)
    plane = np.array([0, 1, 2, 0, 1, 2])
    d = triangulate_designs(geo, own, plane, np.full(n, 50.0),
                            np.arange(n) * 100.0, np.zeros(n))
    # solo is exactly the single-plane design, so the arms are comparable
    np.testing.assert_array_equal(d["solo"], np.concatenate([own, geo], 1))
    np.testing.assert_array_equal(d["solo"][:, :fd], own)
    # Context is BAND-POOLED before the slab mean (as the reference does), so it
    # is 2 x band_d, not 2 x fd — the un-pooled form made the cross design 6162
    # dims and OOM-killed a 200 GB node at 2.5M rows.
    assert d["cross"].shape[1] == fd + 2 * band_d + 2 + gd
    # xwire carries NO own features — it is a model-INDEPENDENT ceiling
    # (reference probe_3d_triangulate.py:96 = concatenate([xw, geo])). This
    # assertion previously included `fd`, contradicting its own "wires, not
    # features" comment, and that made the ceiling model-dependent: it produced
    # an apparent k30-vs-R1 difference of 0.0248 for a design that cannot have
    # one. The reference measures literally 0.6979 in every row of every jsonl
    # across ~20 checkpoints, which is only possible with no model features.
    assert d["xwire"].shape[1] == 2 + 2 + gd         # 2 wires + 2 hit + geo
    assert d["xwire"].shape[1] < d["cross"].shape[1]
    # And the concrete property that matters: perturbing the model features must
    # not move xwire at all.
    own2 = own + 7.0
    d2 = triangulate_designs(geo, own2, plane, np.full(n, 50.0),
                             np.arange(n) * 100.0, np.zeros(n))
    np.testing.assert_array_equal(d["xwire"], d2["xwire"])
    assert not np.array_equal(d["solo"], d2["solo"])   # the control moved


def test_slabs_do_not_pool_across_events():
    """Without the event in the key, a row gets context averaged over UNRELATED
    events, which destroys the correspondence the epipolar match exploits.
    Measured on real data: partner-wire correlation with u went from +0.07 to
    +0.224 once the event was keyed.
    """
    plane = np.array([0, 1, 0, 1])
    tick = np.array([100.0, 100.0, 100.0, 100.0])     # same drift bin
    wire = np.array([10.0, 20.0, 10.0, 999.0])        # ev1's partner is far away
    feats = np.zeros((4, 1), np.float32)
    ev = np.array([0, 0, 1, 1])
    _, ctxw, hit = slab_context(plane, tick, feats, wire, ev, tbin=8)
    assert hit.sum() > 0
    # event 0's row must see 20, not the mean of 20 and 999
    np.testing.assert_allclose(ctxw[0, 0], 20.0 / 2000.0)
    np.testing.assert_allclose(ctxw[2, 0], 999.0 / 2000.0)
