"""Cross-backend parity for the coherent gate and the wavelet seam: torch must match numpy.

The gate's backends agree to ~1e-4 ADC, not exactly (float32 reduction order; see
:mod:`helix.tpc.coherent_gate_ops_numpy`), so these pin a tolerance, not equality.
Runs on CUDA when available, otherwise on CPU torch; skips without torch.
"""
from __future__ import annotations

import numpy as np
import pytest

from helix.core import backend
from helix.tpc import coherent_gate_ops_numpy as gate_np

torch = pytest.importorskip("torch")
from helix.tpc import coherent_gate_ops_torch as gate_t  # noqa: E402

GS = 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _restore_backend():
    """``set_backend`` writes a module global, and tests here flip it. A bare reset
    on the last line is SKIPPED when the test fails, leaking 'torch' into everything
    that runs after. test_backend.py guards the same way."""
    yield
    backend.set_backend("numpy")


def _bands(rng, W=200, lbs=(24, 24, 48)):
    """Coherent (rank-1 per block) + sparse signal + incoherent, per band."""
    out = []
    for Lb in lbs:
        nb = (W + GS - 1) // GS
        cm = rng.standard_normal((nb, Lb)).astype(np.float32) * 2.5
        coherent = cm[np.minimum(np.arange(W) // GS, nb - 1)]
        incoh = rng.standard_normal((W, Lb)).astype(np.float32) * 0.3
        sig = np.zeros((W, Lb), np.float32)
        for _ in range(6):
            sig[rng.integers(W), rng.integers(Lb)] = rng.uniform(20, 40)
        out.append((coherent + incoh + sig).astype(np.float32))
    return out


def _t(bands):
    return [torch.as_tensor(b).to(DEV) for b in bands]


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _close(ref, got, tol=1e-4, what=""):
    assert len(ref) == len(got)
    for i, (a, b) in enumerate(zip(ref, got)):
        a, b = _np(a), _np(b)
        assert a.shape == b.shape
        np.testing.assert_allclose(b, a, rtol=tol, atol=tol, err_msg=f"band {i} {what}")


@pytest.mark.parametrize("npass", [1, 2])
@pytest.mark.parametrize("gs", [64, 48])
def test_torch_gate_matches_numpy(npass, gs):
    """The torch gate must reproduce numpy, including the partial trailing block
    (W=200 is not a multiple of either group size) and the 2-pass signal
    re-detection. The gate is the removal step, so a divergence here changes every
    coefficient downstream."""
    bands = _bands(np.random.default_rng(2))
    a = gate_np.gate_bands(bands, group_size=gs, npass=npass)
    t = gate_t.gate_bands(_t(bands), group_size=gs, npass=npass)
    _close(a, t, what=f"(npass={npass}, gs={gs})")


@pytest.mark.parametrize("npass", [1, 2])
@pytest.mark.parametrize("kgate", [3.0, 4.0])
def test_torch_gate_matches_numpy_kgate(npass, kgate):
    bands = _bands(np.random.default_rng(0))
    ref = gate_np.gate_bands(bands, group_size=GS, kgate=kgate, ksig=3.0, npass=npass)
    got = gate_t.gate_bands(_t(bands), group_size=GS, kgate=kgate, ksig=3.0, npass=npass)
    _close(ref, got, what=f"(npass={npass}, kgate={kgate})")


def test_partial_trailing_block():
    """W not a multiple of group_size exercises the partial trailing block."""
    bands = _bands(np.random.default_rng(1), W=173)          # 173 = 2*64 + 45
    ref = gate_np.gate_bands(bands, group_size=GS, npass=2)
    got = gate_t.gate_bands(_t(bands), group_size=GS, npass=2)
    _close(ref, got)


def test_gate_approx_false():
    bands = _bands(np.random.default_rng(2))
    out = gate_t.gate_bands(_t(bands), gate_approx=False)
    np.testing.assert_array_equal(_np(out[0]), bands[0])   # cA untouched
    assert not np.array_equal(_np(out[1]), bands[1])       # details gated


# ---- the shared wavelet seam: torch must match numpy for EVERY method -------

def test_threshold_bands_honours_method_and_sigma_across_backends():
    """`helix.core.wavelet.threshold_bands` is the DETECTOR-AGNOSTIC seam — the
    two-step (wavedec, then gate, then threshold) entry point. A backend that
    dispatched every call to a universal-only kernel once returned
    universal-thresholded coefficients for `method='topk'` and ignored an explicit
    `sigma=`, with nothing raising. Every method and the explicit sigma are pinned.
    """
    from helix.core.wavelet import threshold_bands, ThresholdSpec

    rng = np.random.default_rng(0)
    bands = [rng.standard_normal((16, L)).astype(np.float32) for L in (64, 64, 128, 256)]

    def run(bk, th, sigma=None):
        backend.set_backend(bk)
        out, n_kept, _, _ = threshold_bands(bands if bk == "numpy" else _t(bands), th, sigma=sigma)
        return np.concatenate([_np(c) for c in out], axis=-1), int(n_kept)

    for th in (ThresholdSpec(method="universal"),
               ThresholdSpec(method="topk", keep=0.02),
               ThresholdSpec(method="energy", energy=0.99)):
        a, ka = run("numpy", th)
        b, kb = run("torch", th)
        assert ka == kb, f"{th.method}: n_kept numpy {ka} vs torch {kb}"
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-4, err_msg=th.method)

    # an explicit sigma must actually be used, not silently recomputed
    th = ThresholdSpec(method="universal")
    a, ka = run("numpy", th, sigma=5.0)
    b, kb = run("torch", th, sigma=5.0)
    assert ka == kb, f"explicit sigma: n_kept numpy {ka} vs torch {kb}"
    np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-4)
    _, k_free = run("torch", th)
    assert kb != k_free, "sigma= made no difference — it is being ignored"


def test_torch_q50_is_not_torch_median():
    """The convention that decides which coefficients survive.

    `torch.median` returns the LOWER of two middles; `np.median` averages them.
    Every sigma in this library is a MAD, and a MAD feeds a threshold — so on
    even-length inputs the tie-break systematically shifts the threshold. It did:
    the torch backend kept 1-2 extra coefficients per event, always more, never
    fewer, until band_sigma switched to q50.
    """
    from helix.core.backend import torch_q50
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert float(torch_q50(x, 0)) == 2.5            # average of the two middles
    assert float(torch.median(x)) == 2.0            # the lower one
    np.testing.assert_allclose(float(torch_q50(x, 0)), float(np.median(x.numpy())))
    odd = torch.tensor([1.0, 5.0, 3.0])
    np.testing.assert_allclose(float(torch_q50(odd, 0)), float(np.median(odd.numpy())))
    # and along a dim, against numpy, on an even-length axis
    m = torch.randn(7, 64)
    np.testing.assert_allclose(torch_q50(m, -1).numpy(),
                               np.median(m.numpy(), axis=-1), rtol=1e-6, atol=1e-6)


def test_torch_threshold_bands_matches_numpy():
    """The other half of the two-step seam: wavedec -> threshold on torch equals numpy."""
    from helix.core.wavelet import wavedec, threshold_bands, ThresholdSpec
    rng = np.random.default_rng(5)
    img = rng.standard_normal((64, 256)).astype(np.float32) * 2.0
    th = ThresholdSpec(method="universal")
    backend.set_backend("numpy")
    bn, ln = wavedec(img, wavelet="db2", level=3, mode="periodization")
    on, kn, tn, sn = threshold_bands(bn, th)
    backend.set_backend("torch")
    bt, lt = wavedec(torch.as_tensor(img).to(DEV), wavelet="db2", level=3, mode="periodization")
    ot, kt, tt, st = threshold_bands(bt, th)
    assert ln == lt and kn == kt and tn == tt, f"n_kept {kn} vs {kt}"
    np.testing.assert_allclose(_np(st), sn, rtol=1e-5, atol=1e-6)
    _close(on, ot)


def test_torch_wavedec_refuses_non_periodization():
    """The FFT filter bank IS periodization; silently accepting another mode
    would return coefficients that disagree with numpy for the same call."""
    from helix.core.wavelet import wavedec
    backend.set_backend("torch")
    with pytest.raises(ValueError, match="periodization"):
        wavedec(torch.zeros(8, 64).to(DEV), wavelet="db2", level=2, mode="symmetric")
