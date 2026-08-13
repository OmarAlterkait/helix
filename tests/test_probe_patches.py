"""Patch reduction and feature gather."""
import numpy as np
import pytest

from helix.probe.patches import patch_rows


def test_pixels_sharing_a_cell_tuple_become_one_row():
    """Pixels with the same 4 cells share every feature the probe sees, so they
    are one row — otherwise the metric is weighted by patch occupancy."""
    cells = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 9, 9, 9]])
    out = patch_rows(cells, np.array([0, 0, 1]), np.array([10, 12, 50]),
                     np.array([5, 7, 90]), np.array([100.0, 300.0, 200.0]),
                     np.array([1.0, 1.0, 1.0]), np.array([2.0, 4.0, -1.0]))
    assert len(out["y"]) == 2
    i = np.argsort(out["plane"])
    # charge-weighted: (100*2 + 300*4)/400 = 3.5
    np.testing.assert_allclose(out["y"][i][0], 3.5, rtol=1e-6)
    np.testing.assert_array_equal(out["n_pixels"][i], [2, 1])


def test_only_dominant_pixels_define_the_label():
    """f_top below threshold contributes no charge and no u."""
    cells = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    out = patch_rows(cells, np.zeros(2, int), np.array([1, 1]), np.array([1, 1]),
                     np.array([100.0, 900.0]), np.array([0.9, 0.1]),
                     np.array([5.0, -100.0]), dom_threshold=0.5)
    np.testing.assert_allclose(out["y"], [5.0])       # the f_top=0.1 pixel ignored
    np.testing.assert_array_equal(out["n_dom"], [1])
    np.testing.assert_array_equal(out["n_pixels"], [2])


def test_patches_with_no_dominant_pixel_are_dropped():
    cells = np.array([[7, 7, 7, 7]])
    out = patch_rows(cells, np.zeros(1, int), np.array([1]), np.array([1]),
                     np.array([500.0]), np.array([0.2]), np.array([3.0]),
                     dom_threshold=0.5)
    assert out["y"].size == 0


def test_geo_carries_presence_and_position_but_not_u():
    cells = np.array([[1, -1, 3, -1]])
    out = patch_rows(cells, np.array([2]), np.array([400]), np.array([2000]),
                     np.array([900.0]), np.array([1.0]), np.array([1.234]))
    geo = out["geo"][0]
    assert geo.shape == (6 + 1 + 1 + 4,)
    np.testing.assert_array_equal(geo[:6], [0, 0, 1, 0, 0, 0])     # plane one-hot
    np.testing.assert_allclose(geo[6], 400 / 2000.0)
    np.testing.assert_allclose(geo[7], 2000 / 4321.0)
    np.testing.assert_array_equal(geo[8:], [1, 0, 1, 0])           # presence bits
    assert not np.isclose(geo, 1.234).any(), "u must not leak into the geo arm"


def test_gather_leaves_missing_bands_zero():
    torch = pytest.importorskip("torch")
    from helix.probe.features import gather_cell_features
    feats = torch.arange(12, dtype=torch.float32).reshape(4, 3)   # 4 cells, d=3
    X = gather_cell_features(feats, np.array([[0, -1], [2, 3]]))
    assert X.shape == (2, 6)
    np.testing.assert_allclose(X[0].numpy(), [0, 1, 2, 0, 0, 0])
    np.testing.assert_allclose(X[1].numpy(), [6, 7, 8, 9, 10, 11])
