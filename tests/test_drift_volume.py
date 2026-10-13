"""SerialFMModel(drift_planes_per_volume=k): the drift layouts (time-sorted, attention across planes at equal
drift time) stop mixing drift volumes. The original sorts by time alone, so a group holds tokens of both volumes
and the denoiser's charge depends on the other volume's amplitude (m7_tests_results.md, C2-C6)."""
import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm                                   # noqa: E402
from test_serial import SMALL, make_batch                          # noqa: E402


def _drift_groups(m, B):
    lay = m._layouts(B["plane_id"], B["t_phys"], B["wire_pos"])
    out = []
    for i in (1, 3):                                               # the two drift layouts
        o, g, _ = lay[i]
        vol = (B["plane_id"][o] // 3)
        out.append([set(vol[s:s + g].tolist()) for s in range(0, len(o), g)])
    return lay, out


def test_drift_groups_hold_one_volume_except_the_boundary_group():
    B = make_batch(n_cells=400)
    _, mixed0 = _drift_groups(build_fm(dict(SMALL)), B)
    _, mixed3 = _drift_groups(build_fm({**SMALL, "drift_planes_per_volume": 3}), B)
    assert sum(len(s) > 1 for s in mixed0[0]) > 1                  # the original mixes volumes in many groups
    for groups in mixed3:                                          # the sorted run of volume 0 then volume 1:
        assert sum(len(s) > 1 for s in groups) <= 2                # only the boundary (and the roll's wrap) mixes


def test_the_option_adds_no_parameters_and_leaves_the_plane_layouts_alone():
    B = make_batch(n_cells=200)
    m0, m3 = build_fm(dict(SMALL)), build_fm({**SMALL, "drift_planes_per_volume": 3})
    assert list(m0.state_dict()) == list(m3.state_dict())
    l0 = m0._layouts(B["plane_id"], B["t_phys"], B["wire_pos"])
    l3 = m3._layouts(B["plane_id"], B["t_phys"], B["wire_pos"])
    for i in (0, 2):
        assert torch.equal(l0[i][0], l3[i][0])
