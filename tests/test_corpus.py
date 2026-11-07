"""Corpus builder on the TPC chain — keeps the old star_tpc semantics:

- clean target is co-supported (clean coeffs at the NOISY support), as
  ``val_clean = clean[gid][b][noisy_mask]``; the clean transform is the chain's
  (``TPCChain.dwt_rows``: no gate, no threshold, the dropped finest band zeroed).
- norm_sigma = mean over the shard's events of the per-event threshold σ (old sigma_tab).
- padding to a 2**level multiple (old ``pad(g, (0,(-nt)%4))``) — tested on a
  non-power-of-2 length so pad>0.
- planes are grouped by TPC (gid // 3) and each TPC goes through the chain once.
"""
from __future__ import annotations

import numpy as np
import pytest
import h5py

from helix.tpc.chain import TPCChain, sim_config
from helix.tpc import chain as tpc_chain
from helix.build import normalization_table
from _build_fixture import build_corpus  # noqa: E402
from helix.core.coeff_io import read_coeff_event, read_coeff_shard

NW = [8, 6, 5]                             # U, V, Y wires of each volume
N_VOL = 2
GIDS = list(range(3 * N_VOL))              # gid = volume * 3 + plane
NT = 102                                   # not a multiple of 4 → pad to 104


def _chain(**overrides):
    o = {"dwt": {"wavelet": "db2", "level": 2}}
    o.update(overrides)
    return TPCChain(sim_config([NW] * N_VOL, o))


def _plane_fn(ev):
    rng = np.random.default_rng(1000 + ev)
    clean, noisy = {}, {}
    for gid in GIDS:
        w = NW[gid % 3]
        c = np.zeros((w, NT), np.float32)
        for _ in range(3):
            c[rng.integers(w), rng.integers(NT)] = rng.uniform(20, 40)
        clean[gid] = c
        nb = (w + 64 - 1) // 64
        coh = (rng.standard_normal((nb, NT)).astype(np.float32) * 2.0)[
            np.minimum(np.arange(w) // 64, nb - 1)]
        incoh = rng.standard_normal((w, NT)).astype(np.float32) * 0.5
        noisy[gid] = (c + coh + incoh).astype(np.float32)
    return noisy, clean


def test_build_corpus_matches_old_semantics(tmp_path):
    ch = _chain()
    noisy, clean, norm = build_corpus(range(4), _plane_fn, ch, tmp_path, dataset_name="cx", run="run_000")
    assert len(noisy) == 4 and len(clean) == 4

    # (1) padding: basis validates; padded length = 104, raw 102
    for ce in noisy:
        ce.basis.validate()
        assert ce.basis.n_ticks_raw == NT and ce.basis.padded_length == 104
        assert list(ce.gids) == GIDS and list(ce.n_wires) == NW * N_VOL
        assert (ce.band < ce.basis.n_bands - 1).all()        # D1 dropped (drop_finest 1)

    # (2) clean is CO-SUPPORTED (identical coords) with clean values at those coords
    for ce_n, ce_c in zip(noisy, clean):
        for k in ("band", "plane_gid", "wire", "tau"):
            np.testing.assert_array_equal(getattr(ce_c, k), getattr(ce_n, k))
        np.testing.assert_array_equal(ce_c.sigma_threshold, ce_n.sigma_threshold)
    # spot-check clean values == the chain's clean transform at the noisy coords, on the target's
    # value step
    ce_n, ce_c = noisy[0], clean[0]
    _, cl = _plane_fn(0)
    clean_bands = {g: ch.dwt_rows(cl[g]) for g in GIDS}
    exact = np.array([clean_bands[int(g)][int(b)][int(w), int(t)]
                      for g, b, w, t in zip(ce_n.plane_gid, ce_n.band, ce_n.wire, ce_n.tau)], np.float32)
    step = tpc_chain.value_step(ce_n.sigma_threshold, ch.cfg["store"]["step_target"])
    np.testing.assert_array_equal(ce_c.value, tpc_chain.quantize(exact, ce_n.band, ce_n.plane_gid, ce_n.gids, step))

    # (3) norm_sigma = mean of the events' sigma_threshold
    np.testing.assert_allclose(norm, normalization_table(noisy))
    np.testing.assert_allclose(norm, np.stack([e.sigma_threshold for e in noisy]).mean(0), rtol=1e-6)

    # (4) shards written + round-trip; /config carries norm_sigma
    for i, ce in enumerate(read_coeff_shard(tmp_path / "cx_coeff_0000.h5")):
        for k in ("band", "wire", "tau", "value"):
            np.testing.assert_array_equal(getattr(ce, k), getattr(noisy[i], k))
    with h5py.File(tmp_path / "cx_coeff_0000.h5", "r") as f:
        np.testing.assert_allclose(f["config"]["norm_sigma"][:], norm)
    # coeff_clean is a VALUES-ONLY shard: co-supported, so it stores no coords and
    # inherits the noisy shard's through the coord_digest-checked join.
    p_noisy = tmp_path / "cx_coeff_0000.h5"
    p_clean = tmp_path / "cx_coeff_clean_0000.h5"
    with h5py.File(p_clean, "r") as f:
        assert not bool(f["config"].attrs["has_coords"])
        assert not any(k in f["coord"] for k in ("band", "plane_gid", "wire", "tau"))
    cc = read_coeff_event(p_clean, 0, coords_from=p_noisy)
    np.testing.assert_array_equal(cc.band, noisy[0].band)
    np.testing.assert_array_equal(cc.value, clean[0].value)
    # reading it standalone must fail loudly, not yield coord-less garbage
    with pytest.raises(ValueError, match="values-only"):
        read_coeff_event(p_clean, 0)


def test_build_corpus_is_the_chain_per_tpc(tmp_path):
    """The builder adds nothing to the chain: per TPC, ``process_images`` on the planes in
    order, then ``coeff_event`` over every TPC's results."""
    ch = _chain()
    (ce,), _, _ = build_corpus([("f.h5", 7)], _plane_fn, ch, tmp_path, run="r", with_clean=False)
    noisy, _ = _plane_fn(7)
    res = [r for t in range(N_VOL) for r in ch.process_images([noisy[3 * t + p] for p in range(3)], t)]
    ref = tpc_chain.coeff_event(ch, res, run="r", source_file="f.h5", event=7)
    for k in ("band", "plane_gid", "wire", "tau", "value", "gids", "n_wires", "sigma_threshold"):
        np.testing.assert_array_equal(getattr(ce, k), getattr(ref, k), err_msg=k)
    assert ce.basis == ref.basis and (ce.run, ce.source_file, ce.event) == ("r", "f.h5", 7)


def test_build_corpus_no_clean(tmp_path):
    noisy, clean, norm = build_corpus(range(2), _plane_fn, _chain(), tmp_path, dataset_name="cx", with_clean=False)
    assert clean == []
    assert not (tmp_path / "cx_coeff_clean_0000.h5").exists()
    assert (tmp_path / "cx_coeff_0000.h5").exists()


# ---- regression: audit fixes ----------------------------------------------

def test_one_event_shard_has_a_table(tmp_path):
    n, c, norm = build_corpus(range(1), _plane_fn, _chain(), tmp_path, dataset_name="cx")
    assert norm.shape[0] == len(GIDS) and norm.max() > 0


def test_global_norm_sigma_is_frozen_across_shards(tmp_path):
    """A supplied norm_sigma must be written verbatim to EVERY shard, so the same
    coefficient normalises identically no matter which shard it landed in."""
    ch = _chain()
    # shard A derives its own table; shard B is built from different events but
    # must reuse A's table when it is passed in.
    a, _, norm_a = build_corpus(range(3), _plane_fn, ch, tmp_path / "a", dataset_name="cx")
    b, _, norm_b = build_corpus(range(3, 6), _plane_fn, ch, tmp_path / "b",
                                dataset_name="cx", norm_sigma=norm_a)
    np.testing.assert_array_equal(norm_b, norm_a)
    with h5py.File(tmp_path / "b" / "cx_coeff_0000.h5") as f:
        np.testing.assert_allclose(f["config"]["norm_sigma"][:], norm_a)
    # a self-derived table for different events would NOT have matched
    _, _, norm_own = build_corpus(range(3, 6), _plane_fn, ch, tmp_path / "c",
                                  dataset_name="cx")
    assert not np.array_equal(norm_own, norm_a)
    with pytest.raises(ValueError, match="norm_sigma: shape"):
        build_corpus(range(2), _plane_fn, ch, tmp_path / "d", dataset_name="cx",
                     norm_sigma=np.ones((99, 3), np.float32))


def test_clean_geometry_mismatch_raises():
    ch = _chain()
    noisy, clean = _plane_fn(0)
    ce = tpc_chain.coeff_event(ch, ch.process_planes(noisy))
    bad_clean = {g: v[: v.shape[0] // 2] for g, v in clean.items()}    # half the wires
    with pytest.raises(ValueError, match="wires"):
        tpc_chain.clean_coeff_event(ch, ce, bad_clean)
    missing = {g: clean[g] for g in GIDS[1:]}                           # drop a gid
    with pytest.raises(ValueError, match="missing gid"):
        tpc_chain.clean_coeff_event(ch, ce, missing)


def test_noisy_planes_must_fill_each_tpc():
    """The chain processes a TPC's planes together: a TPC with a plane missing, a plane
    with the wrong wire count, or a gid past the readout must raise, not misalign."""
    ch = _chain()
    noisy, _ = _plane_fn(0)
    with pytest.raises(ValueError, match="the chain needs all 3"):
        ch.process_planes({g: v for g, v in noisy.items() if g != 4})
    with pytest.raises(ValueError, match="geometry must match"):
        ch.process_planes({**noisy, 1: noisy[1][:-1]})
    with pytest.raises(ValueError, match="tpc 2"):
        ch.process_planes({**noisy, 6: noisy[0], 7: noisy[1], 8: noisy[2]})
    # a whole TPC absent is fine: its gids are simply not in the event
    ce = tpc_chain.coeff_event(ch, ch.process_planes({g: noisy[g] for g in (3, 4, 5)}))
    assert list(ce.gids) == [3, 4, 5]


def test_sigma_threshold_and_norm_are_populated(tmp_path):
    """A shard whose sigma_threshold is zero has a meaningless norm_sigma — the
    tokenizer's normalization table. This regression guards the path where a
    flat-bands branch skipped the sigma assignment entirely. The dropped finest
    band's column is 0 by construction (no coefficients there); every kept band
    must be populated.
    """
    noisy, clean, norm = build_corpus(range(3), _plane_fn, _chain(), tmp_path, dataset_name="cx")
    for ce in noisy:
        assert ce.sigma_threshold.shape[0] == len(GIDS)
        assert np.isfinite(ce.sigma_threshold).all()
        assert (ce.sigma_threshold[:, :-1] > 0).all(), "a kept band's sigma_threshold is zero"
        assert (ce.sigma_threshold[:, -1] == 0).all(), "D1 is dropped: its sigma is 0"
    assert norm.max() > 0, "norm_sigma is all zero"
    with h5py.File(tmp_path / "cx_coeff_0000.h5", "r") as f:
        assert f["config"]["norm_sigma"][:].max() > 0
        assert f["coord"]["sigma_threshold"][:].max() > 0


def test_clean_target_is_written_only_when_asked(tmp_path):
    """A source with noise-free planes writes no clean shard by default (nothing trains on it)."""
    from _build_fixture import PlanesSource
    from helix.build import build
    src = PlanesSource(range(2), _plane_fn)
    build(src, str(tmp_path / "cx"), indices=[0, 1], chain=_chain(), events_per_shard=2)
    assert (tmp_path / "cx_coeff_0000.h5").exists()
    assert not (tmp_path / "cx_coeff_clean_0000.h5").exists()
