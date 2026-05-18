"""Tests for the end-to-end pipeline."""

import numpy as np
from helix.config import DetectorConfig
from helix.pipeline import process_plane, process_event


def test_process_plane(config, synthetic_plane):
    result = process_plane(synthetic_plane["dig"], config, synthetic_plane["sigma_w"])
    assert result.cleaned.shape == synthetic_plane["dig"].shape
    assert result.reconstructed.shape == synthetic_plane["dig"].shape
    assert result.sparse.n_kept > 0


def test_process_plane_f0(config, synthetic_plane):
    result = process_plane(synthetic_plane["dig"], config, synthetic_plane["sigma_w"])
    clean = synthetic_plane["clean"]
    sig = np.abs(clean) > 0
    if not sig.any():
        return
    f0 = 1.0 - np.abs(result.reconstructed[sig] - clean[sig]).sum() / np.abs(clean[sig]).sum()
    assert f0 > 0.6


def test_process_event(config, synthetic_plane):
    planes = {"plane_U": synthetic_plane["dig"], "plane_V": synthetic_plane["dig"]}
    results = process_event(planes, config)
    assert len(results) == 2
    assert "plane_U" in results
    assert "plane_V" in results


def test_config_presets():
    for factory in [DetectorConfig.sbnd, DetectorConfig.microboone, DetectorConfig.icarus]:
        cfg = factory()
        assert cfg.group_size > 0
        assert cfg.beta >= 0
        assert cfg.dwt_level > 0
