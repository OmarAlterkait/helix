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
