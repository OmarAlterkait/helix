"""Per-pixel truth extraction: the hits decode and the pixel reduction."""
import numpy as np
import pytest

from helix.probe.truth import decode_hits_plane, group_centroids, pixel_truth


class _Grp(dict):
    """Minimal stand-in for an h5py plane group."""
    def __getitem__(self, k):
        return np.asarray(super().__getitem__(k))


def test_decode_hits_expands_groups_to_samples():
    """CSR decode: each group owns group_sizes[i] samples, offset from its centre,
    with charge quantised against that group's own peak."""
    g = _Grp(center_wires=[100, 200], center_times=[10, 20],
             delta_wires=[0, 1, -1], delta_times=[0, 2, -2],
             charges_u16=[65535, 32767, 65535], group_ids=[7, 9],
             group_sizes=[2, 1], peak_charges=[1000.0, 500.0])
    out = decode_hits_plane(g)
    np.testing.assert_array_equal(out["wire"], [100, 101, 199])
    np.testing.assert_array_equal(out["tick"], [10, 12, 18])
    np.testing.assert_array_equal(out["group"], [7, 7, 9])
    np.testing.assert_allclose(out["q"], [1000.0, 499.99, 500.0], rtol=1e-3)


def test_decode_hits_rejects_an_inconsistent_plane():
    g = _Grp(center_wires=[1], center_times=[1], delta_wires=[0], delta_times=[0],
             charges_u16=[1, 2], group_ids=[0], group_sizes=[1],
             peak_charges=[1.0])
    with pytest.raises(ValueError, match="sum\\(group_sizes\\)"):
        decode_hits_plane(g)


def test_group_centroids_are_charge_weighted():
    pos = np.array([[0., 0., 0.], [10., 0., 0.], [0., 4., 0.]])
    q = np.array([1.0, 3.0, 1.0])
    cen = group_centroids(pos, q, np.array([0, 0, 1]))
    np.testing.assert_allclose(cen[0], [7.5, 0, 0])      # (0*1 + 10*3)/4
    np.testing.assert_allclose(cen[1], [0, 4, 0])


def test_group_centroids_survive_zero_charge():
    cen = group_centroids(np.array([[2., 0., 0.], [4., 0., 0.]]),
                          np.zeros(2), np.array([5, 5]))
    np.testing.assert_allclose(cen[5], [3., 0., 0.])     # unweighted fallback


def test_group_centroids_reject_a_length_mismatch():
    with pytest.raises(ValueError, match="deposit count mismatch"):
        group_centroids(np.zeros((3, 3)), np.zeros(3), np.zeros(2))


def test_pixel_truth_reduces_and_labels_by_the_dominant_group():
    """f_top is the largest group's SHARE of the pixel, and b1 is that group's
    centroid — which is what makes b1 a meaningful label for the pixel."""
    samples = dict(wire=np.array([5, 5, 5, 6]), tick=np.array([3, 3, 3, 9]),
                   q=np.array([70.0, 30.0, 100.0, 500.0]),
                   group=np.array([1, 2, 1, 2]))
    cen = {1: np.array([1., 1., 1.]), 2: np.array([2., 2., 2.])}
    w, t, qtot, ftop, b1 = pixel_truth(samples, cen, qtot_min=0.0)

    order = np.lexsort((t, w))
    w, t, qtot, ftop, b1 = (a[order] for a in (w, t, qtot, ftop, b1))
    np.testing.assert_array_equal(w, [5, 6])
    np.testing.assert_allclose(qtot, [200.0, 500.0])
    np.testing.assert_allclose(ftop, [170.0 / 200.0, 1.0])   # group 1 dominates (5,3)
    np.testing.assert_allclose(b1[0], [1., 1., 1.])
    np.testing.assert_allclose(b1[1], [2., 2., 2.])


def test_pixel_truth_applies_the_charge_cut():
    """qtot_min is a TARGET parameter — it changes which pixels define y."""
    samples = dict(wire=np.array([1, 2]), tick=np.array([1, 2]),
                   q=np.array([10.0, 900.0]), group=np.array([0, 0]))
    cen = {0: np.array([0., 0., 0.])}
    w, *_ = pixel_truth(samples, cen, qtot_min=250.0)
    np.testing.assert_array_equal(w, [2])
    assert pixel_truth(samples, cen, qtot_min=1e9)[0].size == 0


def test_pixel_truth_drops_pixels_whose_group_has_no_centroid():
    samples = dict(wire=np.array([1]), tick=np.array([1]),
                   q=np.array([900.0]), group=np.array([42]))
    assert pixel_truth(samples, {}, qtot_min=0.0)[0].size == 0
