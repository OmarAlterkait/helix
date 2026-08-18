"""The categorical head's read-back: centroids, decode, and the space it lives in.

These pin the fix for a bug that shipped: `viz_mask_recon.py` fed a posterior
mean over *arcsinh-space* bin centres into `_rows_from_grid`, which applies
`sinh(v) * sigma` -- i.e. the GAUSSIAN head's inverse applied to a CATEGORICAL
head. Because `sinh(E[t]) != E[sinh t]` and the posterior spans many bins, it
read back 68% of the charge (0.674 against the reference's 0.918).

The reference never does this: `fm/train.py:125`, `fm/e7_cat_eval.py:47` and
`fm/viz_cat.py:25` all read back as `sum_k p_k * centroid_k` where the centroid
table is measured in LINEAR space. So the estimator is an expectation of sinh,
never a sinh of an expectation.

`test_jensen_gap_is_real` is the one that would have caught it: it asserts the
two estimators genuinely differ on a spread posterior, so a future "simplification"
back to `sinh(mean)` fails here rather than in a figure nobody checks.
"""
import numpy as np
import pytest

from helix.model.tokenize import (PatchConfig, bin_centroids_ratio,
                                  decode_categorical, _rows_from_grid)

K, N_BAND = 64, 4


def _edges(lo=-6.84699, hi=7.10671, open_ends=True):
    """A table shaped like a real one: uniform in arcsinh, open outer edges."""
    inner = np.linspace(lo, hi, K - 1)
    row = np.concatenate([[-1e18 if open_ends else lo - 1.0], inner,
                          [1e18 if open_ends else hi + 1.0]])
    return np.repeat(row[None], N_BAND, 0).astype(np.float32)


def test_closed_form_matches_numeric_integral():
    """E[sinh t | t ~ U(a,b)] == (cosh b - cosh a)/(b - a), to quadrature."""
    e = _edges(open_ends=False)
    cent = bin_centroids_ratio(e)
    b = 0
    for k in (0, 1, K // 3, K // 2, K - 2, K - 1):
        a_, b_ = float(e[b, k]), float(e[b, k + 1])
        grid = np.linspace(a_, b_, 20001)
        want = np.trapezoid(np.sinh(grid), grid) / (b_ - a_)
        got = float(cent[b, k])
        assert abs(got - want) / max(abs(want), 1e-9) < 1e-5, (
            f"bin {k}: closed form {got:.6g} != integral {want:.6g}")


def test_open_edges_are_reflected_not_taken_literally():
    """The +-1e18 sentinels must not become centroids of ~5e17."""
    cent = bin_centroids_ratio(_edges())
    assert np.isfinite(cent).all()
    # Bounded by sinh of a plausible token value, not by the sentinel.
    assert np.abs(cent).max() < np.sinh(10.0), \
        f"outer centroid {np.abs(cent).max():.3g} looks like a sentinel"
    # And the outer bins still exceed their neighbours in magnitude.
    assert abs(cent[0, -1]) > abs(cent[0, -2]) > abs(cent[0, -3])


def test_measured_table_is_preferred_over_the_closed_form():
    """An empirical cent_ratio wins; a NaN one falls back."""
    e = _edges()
    fake = np.full((N_BAND, K), 3.25, np.float32)
    assert np.allclose(bin_centroids_ratio(e, fake), 3.25)
    nan = np.full((N_BAND, K), np.nan, np.float32)
    assert np.allclose(bin_centroids_ratio(e, nan), bin_centroids_ratio(e))


def test_onehot_posterior_returns_that_bins_centroid():
    """A delta posterior must decode to exactly the bin's centroid."""
    cent = bin_centroids_ratio(_edges())
    band = np.array([0, 1, 2, 3], np.int64)
    want_k = np.array([[3, 17], [K - 1, 0], [8, 8], [40, 2]])
    logits = np.full((4, 2, K), -30.0, np.float32)
    for c in range(4):
        for s in range(2):
            logits[c, s, want_k[c, s]] = 30.0
    got = decode_categorical(logits, band, cent, readout="mean")
    exp = cent[band[:, None], want_k]
    np.testing.assert_allclose(got, exp, rtol=1e-5)
    np.testing.assert_allclose(
        decode_categorical(logits, band, cent, readout="mode"), exp, rtol=1e-5)


def test_jensen_gap_is_real():
    """A SPREAD posterior must not decode to sinh(mean-of-asinh-centres).

    This is the shipped bug, in miniature. If someone "simplifies"
    decode_categorical back to that form, this fails.
    """
    e = _edges()
    cent = bin_centroids_ratio(e)
    # asinh-space midpoints, the thing the buggy path averaged
    ee = e.copy()
    for b in range(N_BAND):
        ee[b, 0] = ee[b, 1] - (ee[b, 2] - ee[b, 1])
        ee[b, -1] = ee[b, -2] + (ee[b, -2] - ee[b, -3])
    mid = 0.5 * (ee[:, :-1] + ee[:, 1:])

    band = np.zeros(1, np.int64)
    x = np.linspace(-3, 3, K).astype(np.float32)      # broad posterior
    logits = (-0.5 * x[None, None, :] ** 2).repeat(1, 0).reshape(1, 1, K)
    p = np.exp(logits - logits.max()); p /= p.sum()

    correct = float(decode_categorical(logits, band, cent, "mean")[0, 0])
    buggy = float(np.sinh((p[0, 0] * mid[0]).sum()))
    assert abs(correct) > 1e-6
    assert abs(buggy - correct) / abs(correct) > 0.05, (
        f"sinh(mean)={buggy:.4g} vs E[sinh]={correct:.4g} — the estimators "
        f"agree here, so this test cannot catch the bug it exists for")


def test_rows_from_grid_space_argument():
    """`ratio` multiplies by sigma; `asinh` sinh-es first. They must differ."""
    cfg = PatchConfig(cell_t="grid_center")
    n_cells, n_slot = 3, cfg.n_slot
    occ = np.zeros((n_cells, n_slot), bool); occ[:, :4] = True
    vals = np.full((n_cells, n_slot), 1.5, np.float32)
    kw = dict(cell_band=np.zeros(n_cells, np.int64),
              cell_gid=np.zeros(n_cells, np.int64),
              cell_wb=np.arange(n_cells, dtype=np.int64),
              cell_tb=np.zeros(n_cells, np.int64),
              gids=np.array([0]), norm_sigma=np.array([[2.0, 2.0, 2.0, 2.0]]),
              cfg=cfg)
    ratio = _rows_from_grid(occ, vals, space="ratio", **kw)["value"]
    asinh = _rows_from_grid(occ, vals, space="asinh", **kw)["value"]
    np.testing.assert_allclose(ratio, 1.5 * 2.0, rtol=1e-6)
    np.testing.assert_allclose(asinh, np.sinh(1.5) * 2.0, rtol=1e-6)
    assert not np.allclose(ratio, asinh)
    with pytest.raises(ValueError):
        _rows_from_grid(occ, vals, space="nonsense", **kw)


def test_decode_rejects_unknown_readout():
    cent = bin_centroids_ratio(_edges())
    with pytest.raises(ValueError):
        decode_categorical(np.zeros((1, 1, K), np.float32),
                           np.zeros(1, np.int64), cent, readout="median")

def test_mean_and_mode_differ_on_a_skewed_posterior():
    """`readout` must actually select an estimator.

    A one-hot posterior makes mean and mode identical, so the delta test above
    cannot tell them apart -- a mutation that ignores `readout` survives it. The
    reference reports BOTH for a reason: mean closes charge, mode is the
    bright-signal estimate, and on a skewed posterior they are far apart.
    """
    cent = bin_centroids_ratio(_edges())
    band = np.zeros(1, np.int64)
    # skewed: modal bin near the middle, a long heavy tail toward the top
    w = np.zeros(K, np.float32)
    w[K // 2] = 3.0
    w[K - 8:] = 1.0
    logits = np.log(w + 1e-9)[None, None, :].astype(np.float32)

    mean = float(decode_categorical(logits, band, cent, "mean")[0, 0])
    mode = float(decode_categorical(logits, band, cent, "mode")[0, 0])
    assert mode == pytest.approx(float(cent[0, K // 2]), rel=1e-5)
    assert abs(mean - mode) / max(abs(mean), 1e-9) > 0.5, (
        f"mean {mean:.4g} and mode {mode:.4g} are too close for this test to "
        f"distinguish the two read-outs")
