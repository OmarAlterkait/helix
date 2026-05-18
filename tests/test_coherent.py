"""Tests for coherent noise removal."""

import numpy as np
from helix.config import DetectorConfig
from helix.coherent import remove_coherent


def test_removes_coherent_noise(config, synthetic_plane):
    cleaned = remove_coherent(synthetic_plane["dig"], config, synthetic_plane["sigma_w"])
    coh_before = np.sqrt((synthetic_plane["coh"] ** 2).mean())
    residual_coh = cleaned - synthetic_plane["clean"] - (
        synthetic_plane["dig"] - synthetic_plane["coh"] - synthetic_plane["clean"])
    coh_after = np.sqrt((residual_coh ** 2).mean())
    assert coh_after < coh_before * 0.5


def test_preserves_signal(config, synthetic_plane):
    cleaned = remove_coherent(synthetic_plane["dig"], config, synthetic_plane["sigma_w"])
    clean = synthetic_plane["clean"]
    sig = np.abs(clean) > 0
    if not sig.any():
        return
    err = np.abs(cleaned[sig] - clean[sig])
    ac = np.abs(clean[sig])
    f0 = 1.0 - err.sum() / ac.sum()
    assert f0 > 0.7


def test_output_shape(config, synthetic_plane):
    cleaned = remove_coherent(synthetic_plane["dig"], config, synthetic_plane["sigma_w"])
    assert cleaned.shape == synthetic_plane["dig"].shape


def test_no_signal_passthrough(config):
    """With no signal, coherent removal should mostly cancel coh without creating artifacts."""
    rng = np.random.default_rng(123)
    nw, nt = 128, config.num_time_steps
    gs = config.group_size
    ng = nw // gs
    sigma = np.full(nw, 1.5, dtype=np.float32)
    intrinsic = (sigma[:, None] * rng.standard_normal((nw, nt))).astype(np.float32)
    coh_w = rng.standard_normal((ng, nt)).astype(np.float32) * 2.5
    coh = np.zeros((nw, nt), dtype=np.float32)
    for w in range(nw):
        coh[w] = coh_w[w // gs]
    dig = intrinsic + coh

    cleaned = remove_coherent(dig, config, sigma)
    rms_before = np.sqrt((dig ** 2).mean())
    rms_after = np.sqrt((cleaned ** 2).mean())
    assert rms_after < rms_before


def test_sigma_estimation_fallback(config, synthetic_plane):
    """When sigma is not provided, it should be estimated from data."""
    cleaned = remove_coherent(synthetic_plane["dig"], config, sigma_per_wire=None)
    assert cleaned.shape == synthetic_plane["dig"].shape
