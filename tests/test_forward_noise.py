"""The LArTPC forward noise model: ENC, coherent grouping, digitisation.

Moved from pimm-data with the model itself. The 20 tests left behind there
exercise `Densify` and the load-time pipeline against JAXTPCDataset fixtures --
those are data-layer integration and stay with the data layer. These 11 are the
physics: they assert the ENC formula, that a coherent draw is shared within a
group and independent of group size, and that digitisation clips/rounds/offsets
as the production formula does. None of them needs simulator output; they run on
synthetic arrays, so helix's physics tests stand alone.
"""
import numpy as np
import pytest

from helix.tpc.noise import (generate_noise, digitize, incoherent_noise,
                             coherent_noise, DEFAULT_ENC)


def test_generate_noise_returns_noise():
    """generate_noise returns the noise array (caller adds); shape/dtype OK."""
    rng = np.random.default_rng(0)
    shape = (128, 256)
    noise = generate_noise(shape, rng=rng, wire_lengths_m=2.3, incoherent=True,
                           coherent=True)
    assert noise.shape == shape
    assert noise.dtype == np.float32
    assert np.any(noise != 0.0)
    # caller adds it; generation does not touch any image
    img = np.zeros(shape, dtype=np.float32)
    noisy = img + noise
    assert np.all(img == 0.0) and np.any(noisy != 0.0)


def test_incoherent_requires_wire_lengths():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_noise((8, 16), rng=rng, incoherent=True, coherent=False)


def test_generate_noise_rejects_non_2d():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_noise((10,), rng=rng, wire_lengths_m=2.3)


def test_incoherent_rms_matches_enc_model():
    """Per-channel RMS ~ sqrt(white^2 + (y + z*L)^2) with a flat series shape."""
    rng = np.random.default_rng(1)
    n_ch, n_ticks = 64, 4096
    L = 2.0
    x, y, z = DEFAULT_ENC
    noise = incoherent_noise((n_ch, n_ticks), L, rng)
    rms = noise.std(axis=1)
    expected = np.sqrt(x**2 + (y + z * L) ** 2)
    assert abs(rms.mean() - expected) < 0.1 * expected


def test_incoherent_per_length_array():
    rng = np.random.default_rng(2)
    n_ch, n_ticks = 32, 4096
    lengths = np.linspace(0.5, 4.0, n_ch)
    noise = incoherent_noise((n_ch, n_ticks), lengths, rng)
    # Longer wires -> larger series term -> larger RMS (monotone trend).
    rms = noise.std(axis=1)
    # correlation between length and rms should be strongly positive
    c = np.corrcoef(lengths, rms)[0, 1]
    assert c > 0.8


def test_incoherent_length_mismatch_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        incoherent_noise((10, 64), np.ones(7), rng)


# --------------------------------------------------------------------------
# coherent noise: shared-within-group + the no-1/sqrt(N) invariant
# --------------------------------------------------------------------------

def test_coherent_shared_within_group():
    rng = np.random.default_rng(3)
    gs = 64
    noise = coherent_noise(256, 512, rng, group_size=gs)
    # every channel in a group carries the identical waveform
    for g0 in range(0, 256, gs):
        block = noise[g0:g0 + gs]
        assert np.allclose(block, block[0][None, :])
    # different groups differ
    assert not np.allclose(noise[0], noise[gs])


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_coherent_rms_independent_of_group_size(group_size):
    """The shared waveform survives per-channel averaging — its per-channel RMS
    is ~rms_adc regardless of group_size, NOT rms_adc/sqrt(group_size)."""
    rng = np.random.default_rng(4)
    rms_adc = 2.5
    noise = coherent_noise(512, 8192, rng, group_size=group_size,
                           rms_adc=rms_adc)
    per_channel_rms = noise.std(axis=1).mean()
    # within ~25% of the per-group target (neighbor coupling adds a little);
    # the 1/sqrt(N) bug would give 2.5/4=0.6 .. 2.5/sqrt(128)=0.22 — far below.
    assert 0.75 * rms_adc < per_channel_rms < 1.5 * rms_adc


def test_digitize_clips_rounds_and_is_integer_valued():
    ped = 410
    x = np.array([[-1000.0, -410.3, 0.4, 0.6, 3684.4, 5000.0]], dtype=np.float32)
    out = digitize(x, ped)  # valid pedestal-subtracted range [-410, 3685]
    # clipped to [-ped, adc_max-ped]
    assert out.min() >= -ped and out.max() <= 4095 - ped
    assert np.isclose(out[0, 0], -ped)        # -1000 -> code 0 -> -410
    assert np.isclose(out[0, -1], 4095 - ped)  # 5000 -> code 4095 -> 3685
    # integer-valued (codes), rounding to nearest
    assert np.allclose(out, np.round(out))


def test_digitize_gain_and_nbits():
    # n_bits sets adc_max; a value above the cap is clipped (10-bit -> 1023)
    out10 = digitize(np.array([[2000.0]], np.float32), 0, n_bits=10)
    assert out10[0, 0] == 1023.0
    # explicit adc_max overrides n_bits
    out = digitize(np.array([[1000.0]], np.float32), 0, n_bits=12, adc_max=500)
    assert out[0, 0] == 500.0
    # gain scales before pedestal/clip
    outg = digitize(np.array([[100.0]], np.float32), 0, gain=2.0)
    assert outg[0, 0] == 200.0

def test_torch_digitize_matches_the_numpy_formula():
    """dense_ops.digitize (torch) == round -> clip -> unpedestal.

    Came from pimm-data's test_dense_collapse when digitize moved here with
    the forward model. It asserts the FORMULA rather than importing an
    oracle across the boundary.
    """
    import torch
    from helix.tpc import dense_ops

    g = {0: torch.randn(2, 4, 8) * 30}
    ped = {0: 400}
    out = dense_ops.digitize(g, ped, n_bits=12)[0].numpy()
    want = np.clip(np.rint(g[0].numpy() + 400), 0, (1 << 12) - 1) - 400
    np.testing.assert_allclose(out, want)


# ─────────────────────────────────────────────────────────────────────────────
# Recovered from tests/test_forward_mirror.py, which was deleted with the
# forward-model move. Its SUBJECT did not move: DetectorConfig carries a third
# independent copy of the ENC constants and the coherent beta, and the only test
# that pinned them against helix/tpc/noise.py went with the mirror file. From
# then until this was restored, the two copies could drift with nothing failing.
#
# This is intra-helix. It has nothing to do with pimm-data and should not have
# been deleted alongside the cross-repo tests.
# ─────────────────────────────────────────────────────────────────────────────

def test_detector_config_matches_the_noise_module_constants():
    """DetectorConfig's ENC triple IS helix.tpc.noise.DEFAULT_ENC.

    config.py:75-77 restates (0.90, 0.79, 0.22) as three separate fields. If
    someone retunes the forward model in noise.py, wire_sigma_intrinsic keeps
    computing the old sigma and the corpus builder and the config disagree
    silently.
    """
    from helix.tpc.config import DetectorConfig
    cfg = DetectorConfig()
    assert (cfg.noise_enc_x, cfg.noise_enc_y, cfg.noise_enc_z) == DEFAULT_ENC


def test_detector_config_sigma_matches_the_forward_model_formula():
    """The two copies must also AGREE NUMERICALLY, not merely hold equal floats."""
    from helix.tpc.config import DetectorConfig
    cfg = DetectorConfig()
    x, y, z = DEFAULT_ENC
    for L in (0.0, 2.33, 4.7):
        assert np.isclose(cfg.wire_sigma_intrinsic(L),
                          np.sqrt(x**2 + (y + z * L) ** 2), rtol=1e-6)


def test_detector_config_beta_matches_the_noise_module():
    """xblock_kernel is (-beta, 1, -beta) built from a beta restated in config.py."""
    from helix.tpc.config import DetectorConfig
    from helix.tpc.noise import DEFAULT_COH_BETA
    k = DetectorConfig().xblock_kernel
    assert k == (-DEFAULT_COH_BETA, 1.0, -DEFAULT_COH_BETA)
