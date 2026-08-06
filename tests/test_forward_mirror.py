"""helix.tpc's forward model must stay identical to pimm-data's.

``helix.tpc.{geometry,noise,dense_ops}`` are mirrors of ``pimm_data.*``. The
duplication is deliberate — pimm-data must run with no helix installed and helix
must build a corpus with no pimm-data installed, so neither may depend on the
other — but a mirror with nothing holding it together is just a fork waiting to
happen. This is what holds it together.

The same forward model is used in two roles, which is why both copies exist:

  pimm-data   training-time augmentation (AddNoise / Digitize in the dense path)
  helix       build-time detector forward model (the corpus builder)

If they drift, a corpus built by helix stops matching the noise a training run
augments with, and nothing else would report it.

Skips when pimm-data is not importable.
"""

import numpy as np
import pytest

pimm_data = pytest.importorskip("pimm_data", reason="pimm-data not installed")

from helix.tpc import geometry as h_geom          # noqa: E402
from helix.tpc import noise as h_noise            # noqa: E402

from pimm_data import geometry as p_geom          # noqa: E402
from pimm_data import noise as p_noise            # noqa: E402

GEOM = "cubic_wireplane_geometry.json"
NW, NT, GS = 128, 256, 64


def test_defaults_match():
    """The inline physics constants are the model. If these drift, everything
    downstream drifts silently."""
    assert h_noise.DEFAULT_ENC == p_noise.DEFAULT_ENC
    for name in ("DEFAULT_GROUP_SIZE", "DEFAULT_COHERENT_RMS_ADC",
                 "DEFAULT_SAMPLING_RATE_HZ"):
        if hasattr(p_noise, name):
            assert getattr(h_noise, name) == getattr(p_noise, name), name


def test_plane_registry_matches():
    h = h_geom.load_plane_registry(GEOM)
    p = p_geom.load_plane_registry(GEOM)
    assert sorted(h) == sorted(p)
    for gid in h:
        assert h[gid]["label"] == p[gid]["label"]
        for k in ("n_wires", "n_ticks", "pedestal"):
            assert h[gid][k] == p[gid][k], f"gid {gid} field {k}"
        np.testing.assert_array_equal(h[gid]["wire_lengths"],
                                      p[gid]["wire_lengths"])


@pytest.mark.parametrize("coherent,incoherent", [(True, True), (True, False),
                                                 (False, True)])
def test_generate_noise_bit_identical(coherent, incoherent):
    """Same seed, same numbers — the RNG draw order is part of the model."""
    L = np.full(NW, 2.33)
    kw = dict(wire_lengths_m=L, incoherent=incoherent, coherent=coherent,
              group_size=GS)
    a = h_noise.generate_noise((NW, NT), rng=np.random.default_rng(7), **kw)
    b = p_noise.generate_noise((NW, NT), rng=np.random.default_rng(7), **kw)
    np.testing.assert_array_equal(a, b)


def test_colored_spectrum_bit_identical():
    """The colored series spectrum is the piece an earlier build got wrong (it
    defaulted to white), so it is pinned explicitly."""
    L = np.full(NW, 2.33)
    freqs = np.fft.rfftfreq(NT, d=1.0 / 2e6)
    amps = 1.0 / (1.0 + freqs / 5e4)
    a = h_noise.generate_noise((NW, NT), rng=np.random.default_rng(3),
                               wire_lengths_m=L, incoherent=True, coherent=True,
                               series_spectrum=(freqs, amps), group_size=GS)
    b = p_noise.generate_noise((NW, NT), rng=np.random.default_rng(3),
                               wire_lengths_m=L, incoherent=True, coherent=True,
                               series_spectrum=(freqs, amps), group_size=GS)
    np.testing.assert_array_equal(a, b)


def test_digitize_bit_identical():
    rng = np.random.default_rng(11)
    sig = rng.normal(0, 40, size=(NW, NT)).astype(np.float32)
    for ped, bits in ((1843, 12), (410, 12), (0, 10)):
        np.testing.assert_array_equal(h_noise.digitize(sig, ped, n_bits=bits),
                                      p_noise.digitize(sig, ped, n_bits=bits))


def test_torch_dense_ops_bit_identical():
    """The corpus was built through the TORCH path, so that is the one that
    actually has to match."""
    torch = pytest.importorskip("torch")
    from helix.tpc import dense_ops as h_d
    from pimm_data import dense_ops as p_d

    g = torch.Generator().manual_seed(5)
    n = 500
    wire = torch.randint(0, NW, (n,), generator=g)
    time = torch.randint(0, NT, (n,), generator=g)
    value = torch.randn(n, generator=g)
    pid = torch.zeros(n, dtype=torch.long)
    offset = torch.tensor([n])
    geom = {0: dict(label="volume_0_U", n_wires=NW, n_ticks=NT, pedestal=1843,
                    wire_lengths=np.full(NW, 2.33, np.float32))}

    a = h_d.densify(wire, time, value, pid, offset, geom)
    b = p_d.densify(wire, time, value, pid, offset, geom)
    assert sorted(a) == sorted(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"densify differs on plane {k}"

    seeds = torch.tensor([12345])
    na = h_d.add_intrinsic_noise({k: v.clone() for k, v in a.items()}, geom, seeds=seeds)
    nb = p_d.add_intrinsic_noise({k: v.clone() for k, v in b.items()}, geom, seeds=seeds)
    for k in na:
        assert torch.equal(na[k], nb[k]), f"add_intrinsic_noise differs on plane {k}"

    peds = {k: geom[k]["pedestal"] for k in na}
    da = h_d.digitize({k: v.clone() for k, v in na.items()}, peds, n_bits=12)
    db = p_d.digitize({k: v.clone() for k, v in nb.items()}, peds, n_bits=12)
    for k in da:
        assert torch.equal(da[k], db[k]), f"digitize differs on plane {k}"
