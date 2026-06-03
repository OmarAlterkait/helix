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


def test_numpy_torch_coeffs_identical():
    """torch FFT periodization DWT is now pywt-exact (== numpy/jax), not just
    a self-consistent PR pair. Coeffs, n_kept and reconstruction all agree to
    float32 precision."""
    torch = pytest.importorskip("torch")
    x = _signal()
    spec = ThresholdSpec("universal", "hard", scale=1.2)
    backend.set_backend("numpy")
    rn = sparsify(x, wavelet="coif3", level=6, mode="periodization", threshold=spec)
    recn = np.asarray(reconstruct(rn, 1024))
    backend.set_backend("torch")
    rt = sparsify(x, wavelet="coif3", level=6, mode="periodization", threshold=spec)
    rect = np.asarray(reconstruct(rt, 1024))
    assert rt.n_kept == rn.n_kept                       # same survivors, not just similar
    assert len(rt.coeffs) == len(rn.coeffs)
    for a, b in zip(rn.coeffs, rt.coeffs):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape
        assert np.abs(a - b).max() < 1e-3               # float32 rounding only
    assert np.abs(recn - rect).max() < 1e-3


def test_torch_dwt_matches_pywt_long():
    """Raw torch _wavedec reproduces pywt.wavedec on an optical-scale signal
    (length a multiple of 2^level)."""
    torch = pytest.importorskip("torch")
    import pywt
    from helix.core import wavelet_ops_torch as wt
    n = 1 << 14                                         # 16384 = optical-scale, even at every level
    level = pywt.dwt_max_level(n, pywt.Wavelet("coif3").dec_len)  # same cap helix applies
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, n)).astype(np.float32)
    tc = wt._wavedec(torch.as_tensor(x), "coif3", level)
    pc = pywt.wavedec(x, "coif3", level=level, mode="periodization", axis=-1)
    assert len(tc) == len(pc)
    for a, b in zip(tc, pc):
        assert np.abs(a.numpy() - b).max() < 1e-3
