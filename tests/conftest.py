"""Shared test fixtures: synthetic TPC data generation."""

import importlib.util
import os

import numpy as np
import pytest

from helix.tpc.config import DetectorConfig


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


# Pin torch's intra-op threads for the whole suite.
#
# Two reasons, both measured. Speed: these tests run many tiny models, where 20
# threads cost far more in coordination than they save — 381s -> 66s across
# test_model_fm.py + test_training_parity.py. Determinism: CPU float reductions
# split across threads, so the sum order (and the last bits) depend on the thread
# count, which depends on machine load. That is what made the goldens fail
# intermittently before they pinned it themselves.
try:
    import torch
    torch.set_num_threads(4)
except ImportError:
    pass


def pytest_configure(config):
    """Refuse to pretend the pimm seam was checked when pimm is absent.

    Seven test modules gate themselves on `_pimm_importable()` and skip when it
    is False. That is correct for a DSP-only environment -- but the skip is
    silent, and it hid FORTY tests, including every test of the pimm-facing
    evaluator, launcher and WeightEMA code. They had never run: the containers
    that carry the DSP dependencies do not have pimm installed, and every
    invocation used a PYTHONPATH without a pimm checkout on it, so the guard
    returned False every time and the suite reported a clean 287 passed.

    `HELIX_REQUIRE_PIMM=1` turns that into a hard error at startup. Use it for
    any run whose result is meant to mean "the integration is good", and put a
    pimm checkout on PYTHONPATH:

        PYTHONPATH=<pimm-checkout>:<helix> HELIX_REQUIRE_PIMM=1 pytest

    See TESTING.md for the full incantation per container.
    """
    if os.environ.get("HELIX_REQUIRE_PIMM") != "1":
        return
    try:
        if importlib.util.find_spec("pimm") is None:
            raise ImportError("no pimm on sys.path")
        import pimm.datasets.builder  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise pytest.UsageError(
            f"HELIX_REQUIRE_PIMM=1 but pimm is not importable ({exc}). "
            f"The pimm-facing tests would have SKIPPED silently. Put a pimm "
            f"checkout on PYTHONPATH, or unset HELIX_REQUIRE_PIMM to accept a "
            f"DSP-only run."
        ) from exc


# ---------------------------------------------------------------------------
# jaxtpc fixtures for the dense-chain tests
#
# tests/test_dense_chain.py exercises Densify (pimm-data) -> AddNoise/Digitize
# (helix) end to end, so it needs a jaxtpc sample. These are SYNTHETIC --
# pimm_data.testing builds a minimal cross-modality-consistent v3 dataset from
# numpy + h5py -- so this is a code dependency helix already has (helix.data
# imports pimm_data), not a data dependency on the simulator.
#
# The *_DATA_ROOT env vars point at real shards when set, matching pimm-data's
# own conftest so the same command means the same thing in both repos.
# ---------------------------------------------------------------------------

def _jaxtpc_root(env, factory, maker, name):
    import os
    real = os.environ.get(env)
    if real:
        return real
    root = factory.mktemp(name)
    maker(str(root))
    return str(root)


@pytest.fixture(scope="session")
def jaxtpc_data_root(tmp_path_factory):
    testing = pytest.importorskip("pimm_data.testing")
    return _jaxtpc_root("JAXTPC_DATA_ROOT", tmp_path_factory,
                        testing.make_jaxtpc_sample, "jaxtpc_synth")


@pytest.fixture(scope="session")
def jaxtpc_pixel_data_root(tmp_path_factory):
    testing = pytest.importorskip("pimm_data.testing")
    maker = getattr(testing, "make_jaxtpc_pixel_sample", None)
    if maker is None:
        pytest.skip("pimm_data.testing has no pixel sample builder")
    return _jaxtpc_root("JAXTPC_PIXEL_DATA_ROOT", tmp_path_factory,
                        maker, "jaxtpc_pixel_synth")
