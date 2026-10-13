"""DenoiseEvents charge-scaling augmentation: off by default without touching the random stream, regions
are (plane gid, wire block) for coefficients and truth cells alike, and it refuses to run without clean shards."""
import numpy as np
import pytest

from helix.data.denoise import DenoiseEvents
from helix.probe.resolution import FW


def test_off_by_default_and_the_random_stream_is_untouched():
    ds = DenoiseEvents("", [], "", items=[])
    rng = np.random.default_rng(3)
    assert ds.draw_gain(rng) is None
    assert rng.random() == np.random.default_rng(3).random()


def test_a_coefficient_and_the_truth_cell_on_its_wires_share_a_region():
    ds = DenoiseEvents("", [], "", items=[], gain_sigma_plane=0.1, gain_sigma_block=0.05, clean_root="/x")
    g = ds.draw_gain(np.random.default_rng(0))
    gid, fw = np.array([0, 2, 5, 5]), np.array([0, 31, 32, 900])
    for w_off in range(FW):                                    # both wires of a fine cell
        assert np.allclose(ds.region_gain(g, gid, fw * FW + w_off), ds.region_gain(g, gid, fw * FW))
    assert ds.region_gain(g, [5], [64])[0] != ds.region_gain(g, [5], [63])[0]   # 64-wire blocks
    assert np.all(g > 0) and abs(np.log(g).std() - np.hypot(0.1, 0.05)) < 0.02


def test_gains_need_the_clean_shards():
    with pytest.raises(ValueError):
        DenoiseEvents("", [], "", items=[], gain_sigma_plane=0.1)
