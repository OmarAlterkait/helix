"""Shared test fixtures: synthetic TPC data generation."""

import numpy as np
import pytest

from helix.config import DetectorConfig


@pytest.fixture
def config():
    return DetectorConfig(
        group_size=64,
        beta=0.15,
        num_time_steps=512,
    )


@pytest.fixture
def synthetic_plane(config):
    """Generate a synthetic plane with known signal, intrinsic noise, and coherent noise."""
    rng = np.random.default_rng(42)
    nw, nt = 256, config.num_time_steps
    gs = config.group_size
    ng = nw // gs

    clean = np.zeros((nw, nt), dtype=np.float32)
    for w in range(80, 140):
        t_center = 200 + (w - 80) * 2
        for dt in range(-15, 16):
            t = t_center + dt
            if 0 <= t < nt:
                clean[w, t] = 40.0 * np.exp(-0.5 * (dt / 5.0) ** 2)

    sigma_w = np.full(nw, 1.5, dtype=np.float32)
    intrinsic = (sigma_w[:, None] * rng.standard_normal((nw, nt))).astype(np.float32)

    coh_waveforms = rng.standard_normal((ng, nt)).astype(np.float32) * 2.5
    coh = np.zeros_like(clean)
    for w in range(nw):
        coh[w] = coh_waveforms[w // gs]

    dig = (clean + intrinsic + coh).astype(np.float32)

    return {
        "clean": clean,
        "dig": dig,
        "coh": coh,
        "sigma_w": sigma_w,
        "nw": nw,
        "nt": nt,
    }
