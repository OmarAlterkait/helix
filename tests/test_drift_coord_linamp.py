"""SerialFMModel options for the encoder ablations (m7_tests_results.md):
drift_cathode_tick -- signed distance from the cathode replaces t: relative positions inside a volume keep their
  size, and the drift groups mix volumes only next to the cathode;
lin_amp_scale -- an amplitude channel beside arcsinh, its embedding columns zero: a converted model is unchanged."""
import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm                                   # noqa: E402
from test_serial import SMALL, make_batch                          # noqa: E402


def test_cathode_coordinate_keeps_in_volume_distances_and_mixes_volumes_only_near_the_cathode():
    B = make_batch(n_cells=600)
    B["t_phys"] = torch.rand(600, generator=torch.Generator().manual_seed(1)) * 100.0
    m = build_fm({**SMALL, "drift_planes_per_volume": 3, "drift_cathode_tick": 100.0})
    x = m._tcoord(B)
    vol = B["plane_id"] // 3
    for v in (0, 1):
        i = (vol == v).nonzero().flatten()[:20]
        dt = B["t_phys"][i][:, None] - B["t_phys"][i][None]
        dx = x[i][:, None] - x[i][None]
        assert torch.allclose(dx.abs(), dt.abs(), atol=1e-4)            # same size of relative positions
    assert (x[vol == 0] <= 1e-6).all() and (x[vol == 1] >= -1e-6).all()  # volume 0 at x < 0, volume 1 at x > 0
    o, g, _ = m._layouts(B["plane_id"], x, B["wire_pos"])[1]           # the drift layout, sorted by x
    for s in range(0, len(o), g):
        grp = o[s:s + g]
        if len(set((vol[grp]).tolist())) > 1:                          # a group spanning both volumes ...
            assert x[grp].abs().min() < 10.0                           # ... sits at the cathode (x ~ 0)


def test_amplitude_channel_starts_as_a_no_op():
    B = make_batch(n_cells=120)
    base = build_fm(dict(SMALL)).eval()
    lin = build_fm({**SMALL, "lin_amp_scale": 30.0}).eval()
    sd = base.state_dict()
    w = lin.state_dict()["embed.weight"].clone(); w.zero_(); w[:, :sd["embed.weight"].shape[1]] = sd["embed.weight"]
    sd["embed.weight"] = w
    lin.load_state_dict(sd)
    with torch.no_grad():
        assert torch.allclose(base.encode(B), lin.encode(B), atol=1e-5)
    assert lin.embed.in_features == base.embed.in_features + SMALL["n_slot"]


def test_build_denoise_loads_a_pretrained_fm_into_an_encoder_with_the_amplitude_channel():
    from helix.model.denoise import build_denoise
    B = make_batch(n_cells=120)
    base = build_fm(dict(SMALL)).eval()
    sd = base.state_dict()
    m = build_denoise(dict(SMALL), sd, overrides={"lin_amp_scale": 30.0}).eval()
    with torch.no_grad():
        assert torch.allclose(base.encode(B), m.fm.encode(B), atol=1e-5)    # starts exactly as the pretrained FM
    assert m.fm.embed.weight.shape[1] == sd["embed.weight"].shape[1] + SMALL["n_slot"]
