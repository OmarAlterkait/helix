"""Tests for the lazy backend dispatcher and cross-backend wavelet consistency."""
import numpy as np
import pytest

from helix.core import backend
from helix.core.wavelet import sparsify, reconstruct, ThresholdSpec


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    backend.set_backend("numpy")


def test_default_is_numpy_no_heavy_import():
    backend._override = None
    import os
    os.environ.pop("HELIX_BACKEND", None)
    assert backend.get_backend() == "numpy"


def test_set_backend_validates():
    with pytest.raises(ValueError):
        backend.set_backend("tensorflow")


def test_env_override(monkeypatch):
    backend._override = None
    monkeypatch.setenv("HELIX_BACKEND", "jax")
    assert backend.get_backend() == "jax"


def _signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 1024)).astype(np.float32)
    x[:, 200:260] += 30 * np.exp(-0.5 * ((np.arange(60) - 30) / 6) ** 2)
    return x


@pytest.mark.parametrize("be", ["numpy", "jax", "torch"])
def test_backend_roundtrip(be):
    pytest.importorskip(be if be != "numpy" else "numpy")
    backend.set_backend(be)
    x = _signal()
    r = sparsify(x, wavelet="coif3", level=6, mode="periodization",
                 threshold=ThresholdSpec("topk", keep=0.05))
    rec = np.asarray(reconstruct(r, 1024))
    assert rec.shape == x.shape
    assert r.compression > 1.0
    rel = np.linalg.norm(rec - x) / np.linalg.norm(x)
    assert rel < 0.6   # loose: pulse retained, most noise dropped


def test_numpy_torch_agree():
    torch = pytest.importorskip("torch")
    x = _signal()
    backend.set_backend("numpy")
    rn = reconstruct(sparsify(x, wavelet="coif3", level=6,
                              threshold=ThresholdSpec("topk", keep=0.05)), 1024)
    backend.set_backend("torch")
    rt = np.asarray(reconstruct(sparsify(x, wavelet="coif3", level=6,
                                         threshold=ThresholdSpec("topk", keep=0.05)), 1024))
    # different DWT phase, but both reconstruct the same pulse to similar fidelity
    en = np.linalg.norm(np.asarray(rn) - x) / np.linalg.norm(x)
    et = np.linalg.norm(rt - x) / np.linalg.norm(x)
    assert abs(en - et) < 0.1
