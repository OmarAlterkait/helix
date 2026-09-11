"""JAX noise must reproduce the numpy forward model statistically.

The RNG streams differ (JAX counter-based keys vs numpy Generator), so these are
distributional checks against the same analytic targets the numpy path hits, plus
the structural invariants that are easy to break in a port: the coherent waveform
must be *exactly* shared within a group, anti-correlated between adjacent groups,
and renormalised by the pooled-global RMS AFTER the coupling.
"""
import numpy as np
import pytest

from helix.tpc.noise import (coherent_noise, incoherent_noise, generate_noise,
                             DEFAULT_ENC)

jax = pytest.importorskip("jax", reason="jax not installed")


def _jax_device_ok():
    """jax importable is not the same as jax USABLE.

    On a node with no visible GPU this build raises from `cuInit` instead of
    falling back to CPU, so every test here failed with a RuntimeError rather
    than skipping. Probe once, defensively, and skip the module if the backend
    cannot produce a device (set JAX_PLATFORMS=cpu to run them on CPU)."""
    try:
        return bool(jax.devices())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _jax_device_ok(),
    reason="jax has no usable device (set JAX_PLATFORMS=cpu to run on CPU)")
from helix.tpc.noise_jax import (  # noqa: E402
    coherent_noise_jax, incoherent_noise_jax, generate_noise_jax)

NW, NT, GS = 256, 512, 64
# No module-level PRNGKey: constructing one initialises the JAX backend at
# COLLECTION time, so a node with an absent or broken GPU driver fails to collect
# (RuntimeError: Unable to initialize backend 'cuda') instead of skipping — real
# breakage then looks like infrastructure noise. Keys are built inside tests.


def _key():
    return jax.random.PRNGKey(0)


def test_coherent_shared_within_group_and_anticorrelated():
    a = np.asarray(coherent_noise_jax(_key(), NW, NT, group_size=GS))
    g = a.reshape(NW // GS, GS, NT)
    # every wire in a group carries the identical waveform
    assert np.abs(g - g[:, :1]).max() == 0.0
    # adjacent groups are anti-correlated (beta=0.15 coupling)
    gm = g.mean(axis=1)
    lag1 = np.corrcoef(gm[:-1].ravel(), gm[1:].ravel())[0, 1]
    assert -0.45 < lag1 < -0.15, f"adjacent-group corr {lag1:+.3f} outside expected band"


def test_coherent_rms_matches_target_and_numpy():
    ref = coherent_noise(NW, NT, np.random.default_rng(0), group_size=GS)
    got = np.asarray(coherent_noise_jax(_key(), NW, NT, group_size=GS))
    assert got.shape == ref.shape
    # pooled-global renormalisation to rms_adc=2.5, applied AFTER the coupling
    assert abs(got.std() - 2.5) < 0.05
    assert abs(got.std() - ref.std()) < 0.05


def test_incoherent_rms_matches_analytic():
    x, y, z = DEFAULT_ENC
    L = np.full(NW, 2.33)
    expect = np.sqrt(x ** 2 + (y + z * 2.33) ** 2)
    ref = incoherent_noise((NW, NT), L, np.random.default_rng(0))
    got = np.asarray(incoherent_noise_jax(_key(), NW, NT, L))
    assert abs(got.std() - expect) < 0.02
    assert abs(got.std() - ref.std()) < 0.02


def test_colored_spectrum_is_plumbed():
    """A colored series spectrum must change the spectral shape vs white."""
    L = np.full(NW, 2.33)
    freqs = np.fft.rfftfreq(NT, d=1.0 / 2e6)
    amps = 1.0 / (1.0 + freqs / 5e4)            # a decaying (colored) shape
    white = np.asarray(incoherent_noise_jax(_key(), NW, NT, L, series_spectrum=None))
    color = np.asarray(incoherent_noise_jax(_key(), NW, NT, L,
                                            series_spectrum=(freqs, amps)))
    pw = np.abs(np.fft.rfft(white, axis=1)).mean(0)
    pc = np.abs(np.fft.rfft(color, axis=1)).mean(0)
    # colored puts relatively more power at low frequency than white
    assert (pc[:20].sum() / pc.sum()) > (pw[:20].sum() / pw.sum())


def test_generate_noise_combined_matches_numpy():
    L = np.full(NW, 2.33)
    ref = generate_noise((NW, NT), rng=np.random.default_rng(0), wire_lengths_m=L,
                         incoherent=True, coherent=True, group_size=GS)
    got = np.asarray(generate_noise_jax(_key(), (NW, NT), wire_lengths_m=L,
                                        incoherent=True, coherent=True, group_size=GS))
    assert got.shape == ref.shape == (NW, NT)
    assert abs(got.std() - ref.std()) < 0.06        # incoherent + coherent in quadrature
    assert got.dtype == np.float32


def test_incoherent_requires_wire_lengths():
    with pytest.raises(ValueError, match="wire_lengths_m"):
        generate_noise_jax(_key(), (NW, NT), incoherent=True, coherent=False)
