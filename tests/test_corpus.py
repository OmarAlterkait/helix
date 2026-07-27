"""Corpus builder — matches the old star_tpc semantics (compared line-by-line):

- clean target is co-supported (clean coeffs at the NOISY support), as
  ``val_clean = clean[gid][b][noisy_mask]``.
- norm_sigma = mean over cal events of the per-event threshold σ (old sigma_tab).
- padding to a 2**level multiple (old ``pad(g, (0,(-nt)%16))``) — tested on a
  non-power-of-2 length so pad>0.
"""
from __future__ import annotations

import numpy as np
import h5py

from helix.core.backend import set_backend
from helix.core.wavelet import wavedec
from helix.tpc.config import DetectorConfig
from helix.tpc.pipeline import _pad_time
from helix.tpc.corpus import build_corpus, normalization_table
from helix.core.coeff_io import read_coeff_event, read_coeff_shard

GIDS = [0, 1]
NW = [8, 6]
NT = 102                                   # not a multiple of 4 → pad to 104


def _config():
    return DetectorConfig(num_time_steps=NT, wavelet="db2", dwt_level=2,
                          group_size=64, threshold_kappa=1.0)


def _plane_fn(ev):
    rng = np.random.default_rng(1000 + ev)
    clean, noisy = {}, {}
    for gid, w in zip(GIDS, NW):
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
    set_backend("numpy")
    cfg = _config()
    noisy, clean, norm = build_corpus(range(4), _plane_fn, cfg, tmp_path,
                                      dataset_name="cx", run="run_000", cal_events=(0, 1))
    assert len(noisy) == 4 and len(clean) == 4

    # (1) padding: basis validates; padded length = 104, raw 102
    for ce in noisy:
        ce.basis.validate()
        assert ce.basis.n_ticks_raw == NT and ce.basis.padded_length == 104

    # (2) clean is CO-SUPPORTED (identical coords) with clean values at those coords
    for ce_n, ce_c in zip(noisy, clean):
        for k in ("band", "plane_gid", "wire", "tau"):
            np.testing.assert_array_equal(getattr(ce_c, k), getattr(ce_n, k))
    # spot-check clean values == clean wavedec at the noisy coords (old val_clean)
    ce_n, ce_c = noisy[0], clean[0]
    _, cl = _plane_fn(0)
    clean_bands = {g: wavedec(_pad_time(cl[g], cfg.dwt_level), wavelet=cfg.wavelet,
                              level=cfg.dwt_level, mode=cfg.dwt_mode)[0] for g in GIDS}
    for i in range(ce_n.n_coeff):
        gid, b, w, t = int(ce_n.plane_gid[i]), int(ce_n.band[i]), int(ce_n.wire[i]), int(ce_n.tau[i])
        assert ce_c.value[i] == np.float32(clean_bands[gid][b][w, t])

    # (3) norm_sigma = mean of the cal events' sigma_threshold
    np.testing.assert_allclose(norm, normalization_table([noisy[0], noisy[1]]))
    np.testing.assert_allclose(
        norm, np.stack([noisy[0].sigma_threshold, noisy[1].sigma_threshold]).mean(0), rtol=1e-6)

    # (4) shards written + round-trip; /config carries norm_sigma
    for i, ce in enumerate(read_coeff_shard(tmp_path / "cx_coeff_0000.h5")):
        for k in ("band", "wire", "tau", "value"):
            np.testing.assert_array_equal(getattr(ce, k), getattr(noisy[i], k))
    with h5py.File(tmp_path / "cx_coeff_0000.h5", "r") as f:
        np.testing.assert_allclose(f["config"]["norm_sigma"][:], norm)
    # coeff_clean shard exists and is co-supported with coeff
    cc = read_coeff_event(tmp_path / "cx_coeff_clean_0000.h5", 0)
    np.testing.assert_array_equal(cc.band, noisy[0].band)
    np.testing.assert_array_equal(cc.value, clean[0].value)


def test_build_corpus_no_clean(tmp_path):
    set_backend("numpy")
    noisy, clean, norm = build_corpus(range(2), _plane_fn, _config(), tmp_path,
                                      dataset_name="cx", with_clean=False, cal_events=(0, 1))
    assert clean == []
    assert not (tmp_path / "cx_coeff_clean_0000.h5").exists()
    assert (tmp_path / "cx_coeff_0000.h5").exists()


# ---- regression: audit fixes ----------------------------------------------

def test_cal_events_out_of_range_raises(tmp_path):
    set_backend("numpy")
    import pytest
    # explicit cal_events beyond what was built still raises
    with pytest.raises(ValueError, match="cal_events"):
        build_corpus(range(2), _plane_fn, _config(), tmp_path, dataset_name="cx",
                     cal_events=(0, 5))
    # the DEFAULT (None) averages every event, so a 1-event build is fine
    n, c, norm = build_corpus(range(1), _plane_fn, _config(), tmp_path, dataset_name="cx")
    assert norm.shape[0] == len(GIDS) and norm.max() > 0


def test_global_norm_sigma_is_frozen_across_shards(tmp_path):
    """A supplied norm_sigma must be written verbatim to EVERY shard, so the same
    coefficient normalises identically no matter which shard it landed in."""
    import pytest, h5py
    set_backend("numpy")
    cfg = _config()
    # shard A derives its own table; shard B is built from different events but
    # must reuse A's table when it is passed in.
    a, _, norm_a = build_corpus(range(3), _plane_fn, cfg, tmp_path / "a", dataset_name="cx")
    b, _, norm_b = build_corpus(range(3, 6), _plane_fn, cfg, tmp_path / "b",
                                dataset_name="cx", norm_sigma=norm_a)
    np.testing.assert_array_equal(norm_b, norm_a)
    with h5py.File(tmp_path / "b" / "cx_coeff_0000.h5") as f:
        np.testing.assert_allclose(f["config"]["norm_sigma"][:], norm_a)
    # a self-derived table for different events would NOT have matched
    _, _, norm_own = build_corpus(range(3, 6), _plane_fn, cfg, tmp_path / "c",
                                  dataset_name="cx")
    assert not np.array_equal(norm_own, norm_a)
    with pytest.raises(ValueError, match="norm_sigma shape"):
        build_corpus(range(2), _plane_fn, cfg, tmp_path / "d", dataset_name="cx",
                     norm_sigma=np.ones((99, 3), np.float32))


def test_clean_geometry_mismatch_raises():
    import pytest
    from helix.tpc.pipeline import process_plane, event_coeff_event
    from helix.tpc.corpus import clean_coeff_event
    set_backend("numpy")
    cfg = _config()
    noisy, clean = _plane_fn(0)
    results = {g: process_plane(img, cfg, removal="gate") for g, img in noisy.items()}
    ce = event_coeff_event(results, cfg)
    bad_clean = {g: v[: v.shape[0] // 2] for g, v in clean.items()}    # half the wires
    with pytest.raises(ValueError, match="wires"):
        clean_coeff_event(ce, bad_clean, cfg)
    missing = {GIDS[0]: clean[GIDS[0]]}                                 # drop a gid
    with pytest.raises(ValueError, match="missing gid"):
        clean_coeff_event(ce, missing, cfg)


def test_multipass_now_padded():
    from helix.tpc.pipeline import process_plane
    set_backend("numpy")
    cfg = _config()                                          # NT=102 -> pad to 104
    img = np.random.default_rng(0).standard_normal((8, NT)).astype(np.float32)
    gate = process_plane(img, cfg, removal="gate").sparse
    mp = process_plane(img, cfg, removal="multipass").sparse
    assert [c.shape[-1] for c in gate.coeffs] == [c.shape[-1] for c in mp.coeffs]


def test_sigma_threshold_and_norm_are_populated(tmp_path):
    """A shard whose sigma_threshold is zero has a meaningless norm_sigma — the
    tokenizer's normalization table. This regression guards the path where the
    flat (jax) branch skipped the sigma assignment entirely.
    """
    import h5py
    set_backend("numpy")
    noisy, clean, norm = build_corpus(range(3), _plane_fn, _config(), tmp_path,
                                      dataset_name="cx", cal_events=(0, 1))
    for ce in noisy:
        assert ce.sigma_threshold.shape[0] == len(GIDS)
        assert np.isfinite(ce.sigma_threshold).all()
        assert ce.sigma_threshold.max() > 0, "sigma_threshold is all zero"
    assert norm.max() > 0, "norm_sigma is all zero"
    with h5py.File(tmp_path / "cx_coeff_0000.h5", "r") as f:
        assert f["config"]["norm_sigma"][:].max() > 0
        assert f["coord"]["sigma_threshold"][:].max() > 0
