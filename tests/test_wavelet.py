"""Tests for wavelet sparsification."""

import numpy as np
from helix.config import DetectorConfig
from helix.wavelet import sparsify, reconstruct


def test_sparsify_reduces_coefficients(config, synthetic_plane):
    result = sparsify(synthetic_plane["clean"], config)
    assert result.sparsity > 0.5
    assert result.n_kept < result.n_total


def test_reconstruct_roundtrip(config, synthetic_plane):
    clean = synthetic_plane["clean"]
    result = sparsify(clean, config)
    recon = reconstruct(result, config, clean.shape[1])
    assert recon.shape == clean.shape

    sig = np.abs(clean) > 0
    if sig.any():
        f0 = 1.0 - np.abs(recon[sig] - clean[sig]).sum() / np.abs(clean[sig]).sum()
        assert f0 > 0.8


def test_threshold_values(config):
    """Verify threshold computation for known input."""
    rng = np.random.default_rng(99)
    nw, nt = 64, config.num_time_steps
    noise = rng.standard_normal((nw, nt)).astype(np.float32) * 2.0
    result = sparsify(noise, config)
    assert result.sparsity > 0.8


def test_sigma_per_band_shape(config, synthetic_plane):
    result = sparsify(synthetic_plane["clean"], config)
    assert len(result.sigma_per_band) == config.dwt_level + 1
