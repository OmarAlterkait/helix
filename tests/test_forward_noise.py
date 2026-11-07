"""The LArTPC forward noise model: ENC, coherent grouping, digitisation.

The physics, on synthetic arrays: they assert the ENC formula, that a coherent draw is shared within a
group and independent of group size, and that digitisation clips/rounds/offsets
as the production formula does; then the torch model (what the build uses)
against the numpy reference and, where a JAXTPC checkout is declared, against JAXTPC.
"""
import numpy as np
import pytest

from helix.tpc.noise import DEFAULT_ENC
from _noise_reference import generate_noise, digitize, incoherent_noise, coherent_noise


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
    """helix.tpc.noise.digitize (torch) == round -> clip -> unpedestal.

    Came from pimm-data's test_dense_collapse when digitize moved here with
    the forward model. It asserts the FORMULA rather than importing an
    oracle across the boundary.
    """
    torch = pytest.importorskip("torch")
    from helix.tpc import noise

    g = {0: torch.randn(2, 4, 8) * 30}
    ped = {0: 400}
    out = noise.digitize(g, ped, n_bits=12)[0].numpy()
    want = np.clip(np.rint(g[0].numpy() + 400), 0, (1 << 12) - 1) - 400
    np.testing.assert_allclose(out, want)


# ---------------------------------------------------------------------------
# helix.tpc.noise (torch) against the numpy reference and JAXTPC. These came with the
# dense-chain tests of the retired load-time transforms; they test the noise model itself.
# ---------------------------------------------------------------------------

import os  # noqa: E402
import sys  # noqa: E402

torch = pytest.importorskip("torch")
from helix.tpc import noise as helix_noise  # noqa: E402
from helix.tpc.noise import DEFAULT_SAMPLING_RATE_HZ  # noqa: E402


def _import_jaxtpc():
    """Import JAXTPC's `tools.{coherent_noise,noise}` WITHOUT leaving a trace.

    This used to insert JAXTPC's root at sys.path[0] and evict every cached
    `tools.*` module, then return. Both effects were permanent for the rest of
    the session, so `tools` stopped meaning helix's own `tools/` package and
    three tests in test_model_fm.py failed with
    `ModuleNotFoundError: No module named 'tools.convert_fm_ckpt'` -- but only
    when the whole suite ran, and only in file order. In isolation they passed.
    (helix's own `tools/` is gone now, retired with that converter, so this
    particular symptom cannot recur -- but the mechanism is unchanged.)

    The namespace collision it guards against is real (pimm also ships a `tools`),
    so the eviction stays; it is the FAILURE TO RESTORE that was the bug. The
    returned module objects stay valid after being removed from sys.modules.
    """
    # HELIX_JAXTPC_ROOT, the name helix.paths declares and every other consumer
    # uses (scripts/build_sim.py, tests/test_corpus_identity.py). This
    # read a bare JAXTPC_ROOT with an S3DF literal beside it, so setting the
    # DOCUMENTED variable had no effect on this test and the literal was the only
    # thing that ever matched.
    from helix.paths import resolve as _resolve
    _jx, _ = _resolve('HELIX_JAXTPC_ROOT')
    for root in (str(_jx) if _jx else None,):
        if not (root and os.path.isdir(os.path.join(root, 'tools'))):
            continue
        saved_path = list(sys.path)
        saved_mods = {k: v for k, v in sys.modules.items()
                      if k == 'tools' or k.startswith('tools.')}
        sys.path.insert(0, root)
        for m in list(saved_mods):
            del sys.modules[m]
        try:
            import tools.coherent_noise as cn
            import tools.noise as nz
            if not (getattr(cn, '__file__', '') or '').startswith(root):
                return None          # got a foreign `tools`
            return cn, nz, root
        except Exception:
            return None
        finally:
            for m in [k for k in list(sys.modules)
                      if k == 'tools' or k.startswith('tools.')]:
                del sys.modules[m]
            sys.modules.update(saved_mods)
            sys.path[:] = saved_path
    return None



def test_digitize_matches_numpy():
    g = (torch.randn(2, 16, 32) * 1000.0)
    out = helix_noise.digitize({0: g}, pedestal=410, n_bits=12)
    exp = digitize(g.numpy(), 410, n_bits=12)
    assert np.array_equal(out[0].numpy(), exp)


def test_coherent_adjacent_group_anticorrelation():
    rng = np.random.default_rng(5)
    gs = 32
    noise = coherent_noise(8 * gs, 8192, rng, group_size=gs, beta=0.15)
    # one representative channel per group
    reps = noise[::gs]
    cc = np.corrcoef(reps)
    # adjacent groups are anti-correlated (beta>0); non-adjacent ~0
    adj = np.array([cc[i, i + 1] for i in range(len(reps) - 1)])
    assert adj.mean() < -0.05


def test_reconcile_enc_params_with_npz():
    """helix.tpc.noise.DEFAULT_ENC must equal JAXTPC's noise_spectrum.npz."""
    jx = _import_jaxtpc()
    if jx is None:
        pytest.skip("JAXTPC repo not importable")
    _, nz, root = jx
    npz = os.path.join(root, 'config', 'noise_spectrum.npz')
    if not os.path.exists(npz):
        pytest.skip("noise_spectrum.npz not found")
    x, y, z, _, _ = nz.load_noise_params(npz)
    assert np.allclose(DEFAULT_ENC, (x, y, z)), \
        f"helix DEFAULT_ENC {DEFAULT_ENC} != JAXTPC npz ({x}, {y}, {z})"


def test_reconcile_coherent_bitexact_with_jaxtpc():
    """Identical params + identical Generator -> bit-identical coherent noise.

    pimm-data.coherent_noise is a faithful numpy port of JAXTPC
    generate_coherent_noise, so with the same np.random.Generator they must
    agree exactly (the grouping + spectrum + beta coupling are shared)."""
    jx = _import_jaxtpc()
    if jx is None:
        pytest.skip("JAXTPC repo not importable")
    cn, _, _ = jx
    n_wires, n_ticks, gs = 200, 2048, 64
    ours = coherent_noise(n_wires, n_ticks, np.random.default_rng(123),
                          group_size=gs, rms_adc=2.5, corner_freq_hz=20000.0,
                          spectral_slope=1.5, beta=0.15,
                          sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ)
    theirs = cn.generate_coherent_noise(
        n_wires, n_ticks, group_size=gs, beta=0.15, rms_adc=2.5,
        corner_freq_hz=20000.0, spectral_slope=1.5, sampling_rate_hz=2e6,
        rng=np.random.default_rng(123))
    assert np.allclose(ours, theirs, atol=1e-5)


def test_coherent_torch_default_statistical():
    """Default (on-device torch) coherent: not bit-exact to numpy, but matches the
    forward model's structure — within-group identical, per-group RMS == coh_rms,
    adjacent-group anti-correlation ~ -2β/(1+2β²)."""
    W, T, gid, gs, rms, beta = 256, 4096, 0, 64, 2.5, 0.15
    geom = {gid: {'n_wires': W, 'n_ticks': T}}
    grids = {gid: torch.zeros(1, W, T)}
    helix_noise.add_intrinsic_noise(grids, geom, seeds=[3], coherent=True,
                                  incoherent=False, group_size=gs, coh_rms=rms,
                                  beta=beta)
    x = grids[gid][0].numpy()
    g0 = x[:gs]                                      # within-group identical (common mode)
    assert np.abs(g0 - g0[0]).max() < 1e-5
    reps = x[::gs]                                   # one wire per group
    assert abs(reps.std(axis=1).mean() - rms) < 0.2  # per-group RMS ~ coh_rms
    lag1 = np.corrcoef(reps[:-1].ravel(), reps[1:].ravel())[0, 1]
    assert abs(lag1 - (-2 * beta / (1 + 2 * beta ** 2))) < 0.05


def test_incoherent_rms_matches_enc_statistical():
    W, T, gid = 64, 8192, 0
    L = np.linspace(0.42, 4.63, W).astype(np.float32)
    geom = {gid: {'n_wires': W, 'n_ticks': T, 'wire_lengths': L}}
    grids = {gid: torch.zeros(1, W, T)}
    x, y, z = 0.90, 0.79, 0.22
    helix_noise.add_intrinsic_noise(grids, geom, seeds=[7], coherent=False,
                                  incoherent=True, enc=(x, y, z))
    rms = grids[gid][0].std(dim=1, unbiased=False).numpy()
    expected = np.sqrt(x**2 + (y + z * L) ** 2)
    assert abs((rms / expected).mean() - 1.0) < 0.02       # mean tight
    assert float(np.abs(rms / expected - 1.0).max()) < 0.15  # per-channel band


def test_add_intrinsic_noise_requires_wire_lengths():
    geom = {0: {'n_wires': 8, 'n_ticks': 64}}  # no wire_lengths
    grids = {0: torch.zeros(1, 8, 64)}
    with pytest.raises(ValueError):
        helix_noise.add_intrinsic_noise(grids, geom, seeds=[0], coherent=False,
                                      incoherent=True)


def test_noise_reproducible_by_seed():
    W, T, gid = 32, 1024, 0
    L = np.full(W, 2.33, np.float32)
    geom = {gid: {'n_wires': W, 'n_ticks': T, 'wire_lengths': L}}

    def run(seed):
        g = {gid: torch.zeros(1, W, T)}
        helix_noise.add_intrinsic_noise(g, geom, seeds=[seed], coherent=True,
                                      incoherent=True)
        return g[gid].clone()

    assert torch.equal(run(99), run(99))      # same seed -> identical
    assert not torch.equal(run(1), run(2))    # different seed -> different


@pytest.mark.parametrize("T", [4096, 4097])
def test_incoherent_rms_at_odd_and_even_T(T):
    W, gid = 48, 0
    L = np.linspace(0.42, 4.63, W).astype(np.float32)
    geom = {gid: {'n_wires': W, 'n_ticks': T, 'wire_lengths': L}}
    grids = {gid: torch.zeros(1, W, T)}
    x, y, z = 0.90, 0.79, 0.22
    helix_noise.add_intrinsic_noise(grids, geom, seeds=[3], coherent=False,
                                  incoherent=True, enc=(x, y, z))
    rms = grids[gid][0].std(dim=1, unbiased=False).numpy()
    expected = np.sqrt(x**2 + (y + z * L) ** 2)
    assert abs((rms / expected).mean() - 1.0) < 0.03


def test_within_batch_events_differ_by_seed():
    W, T, gid = 32, 512, 0
    geom = {gid: {'n_wires': W, 'n_ticks': T}}
    g = {gid: torch.zeros(2, W, T)}
    helix_noise.add_intrinsic_noise(g, geom, seeds=[11, 22], coherent=True, incoherent=False)
    assert not torch.equal(g[gid][0], g[gid][1])          # distinct seeds -> distinct noise
    g2 = {gid: torch.zeros(2, W, T)}
    helix_noise.add_intrinsic_noise(g2, geom, seeds=[11, 11], coherent=True, incoherent=False)
    assert torch.equal(g2[gid][0], g2[gid][1])            # equal seeds -> equal noise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_incoherent_rms_on_cuda():
    W, T, gid = 48, 4096, 0
    L = np.linspace(0.42, 4.63, W).astype(np.float32)
    geom = {gid: {'n_wires': W, 'n_ticks': T, 'wire_lengths': L}}
    g = {gid: torch.zeros(1, W, T, device='cuda')}
    x, y, z = 0.90, 0.79, 0.22
    helix_noise.add_intrinsic_noise(g, geom, seeds=[3], coherent=False, incoherent=True,
                                  enc=(x, y, z))
    rms = g[gid][0].std(dim=1, unbiased=False).cpu().numpy()
    expected = np.sqrt(x**2 + (y + z * L) ** 2)
    assert abs((rms / expected).mean() - 1.0) < 0.05  # statistical (cuFFT/RNG != numpy)


def test_digitize_matches_production_formula():
    """digitize == round(x + ped).clip(0, 4095) - ped (doraemon/JAXTPC path)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-2000, 2000, size=(40, 128)).astype(np.float32)
    ped = 410
    out = digitize(x, ped)  # default n_bits=12 -> adc_max=4095, gain=1
    expected = np.round(x + ped).clip(0, 4095).astype(np.float32) - ped
    assert np.array_equal(out, expected)
