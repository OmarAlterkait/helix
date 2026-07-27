"""Cross-backend parity for the coherent gate: jax (GPU) must match numpy.

``jnp.quantile`` and ``np.quantile`` share semantics (both average the two middle
values), so the JAX backend reproduces the numpy A-parity ``sigc`` essentially
exactly — measured max abs 1e-13…1e-16, median relative diff 0.0 on a real
1969x4336 plane. This test pins that: a drift in either backend breaks it.
"""
from __future__ import annotations

import numpy as np
import pytest

from helix.core import backend
from helix.tpc.coherent_gate import coherent_gate

GS = 64

jax = pytest.importorskip("jax", reason="jax not installed")


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


@pytest.mark.parametrize("npass", [1, 2])
@pytest.mark.parametrize("kgate", [3.0, 4.0])
def test_jax_matches_numpy(npass, kgate):
    bands = _bands(np.random.default_rng(0))

    backend.set_backend("numpy")
    ref = coherent_gate(bands, group_size=GS, kgate=kgate, ksig=3.0, npass=npass)
    backend.set_backend("jax")
    got = coherent_gate(bands, group_size=GS, kgate=kgate, ksig=3.0, npass=npass)
    backend.set_backend("numpy")

    for i, (a, b) in enumerate(zip(ref, got)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape
        np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-5,
                                   err_msg=f"band {i} (npass={npass}, kgate={kgate})")


def test_partial_trailing_block():
    """W not a multiple of group_size exercises the NaN-padded partial block."""
    bands = _bands(np.random.default_rng(1), W=173)          # 173 = 2*64 + 45
    backend.set_backend("numpy")
    ref = coherent_gate(bands, group_size=GS, npass=2)
    backend.set_backend("jax")
    got = coherent_gate(bands, group_size=GS, npass=2)
    backend.set_backend("numpy")
    for i, (a, b) in enumerate(zip(ref, got)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-5, atol=1e-5,
                                   err_msg=f"band {i}")


def test_gate_approx_false_and_dispatch():
    bands = _bands(np.random.default_rng(2))
    backend.set_backend("jax")
    out = coherent_gate(bands, gate_approx=False)
    np.testing.assert_array_equal(np.asarray(out[0]), bands[0])   # cA untouched
    assert not np.array_equal(np.asarray(out[1]), bands[1])       # details gated
    backend.set_backend("numpy")


def test_jax_rejects_legacy_median_sigc():
    bands = _bands(np.random.default_rng(3))
    backend.set_backend("jax")
    with pytest.raises(ValueError, match="quantile"):
        coherent_gate(bands, sigc_mode="median")
    backend.set_backend("numpy")
