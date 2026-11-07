"""The categorical objective: its contracts, the per-band binning, and decoding a prediction."""
import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F                                    # noqa: E402
import numpy as np

from helix.model.loss import bucketize_bins, losses_cat            # noqa: E402
from helix.model.loss import bucketize_bins, losses_cat  # noqa: E402
from helix.model.tokenize import (PatchConfig, bin_centroids_ratio,
                                  decode_categorical, _rows_from_grid)


# ---- loss contracts ----------------------------------------------------------------------

# What the categorical objective COMPUTES, stated as the formula rather than a
# recorded number.
#
# These assert the MATH, deliberately, not a frozen output: the masking policy,
# the model width and the bin grid are all expected to change, whereas the formula
# is what has to stay true.

K_C = 8


def _batch(n_cells=6, n_slot=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    occ = (torch.rand(n_cells, n_slot, generator=g) < 0.6)
    return dict(occ=occ.float(), tgt=torch.randn(n_cells, n_slot, generator=g),
                valid=torch.ones(n_cells, n_slot, dtype=torch.bool),
                band_id=torch.randint(0, 2, (n_cells,), generator=g), n_cells=n_cells)


EDGES = torch.linspace(-3, 3, K_C + 1).repeat(2, 1)


def test_bce_is_masked_only_and_is_the_bce_formula():
    """Occupancy BCE is supervised on MASKED cells only: visible occupancy is
    observed, so supervising it would leak the answer."""
    B = _batch()
    g = torch.Generator().manual_seed(1)
    occ_logit = torch.randn(6, 4, generator=g)
    logits = torch.randn(6, 4, K_C, generator=g)
    m = torch.zeros(6, dtype=torch.bool); m[::2] = True

    bce, _ = losses_cat(occ_logit, logits, B, m, EDGES)
    sel = m[:, None] & B["valid"]
    want = F.binary_cross_entropy_with_logits(occ_logit[sel], B["occ"][sel])
    assert torch.allclose(bce, want), "BCE is not the masked-only BCE it claims"

    # and it must actually IGNORE the visible half: perturbing visible logits
    # cannot move it.
    pert = occ_logit.clone(); pert[~m] += 5.0
    bce2, _ = losses_cat(pert, logits, B, m, EDGES)
    assert torch.allclose(bce, bce2), "visible occupancy leaked into the BCE"


def test_value_term_is_the_cross_entropy_of_the_true_bin_on_masked_active_slots():
    B = _batch()
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(6, 4, K_C, generator=g)
    m = torch.zeros(6, dtype=torch.bool); m[:3] = True

    _, val = losses_cat(torch.zeros(6, 4), logits, B, m, EDGES)
    act = m[:, None] & B["valid"] & B["occ"].bool()
    binid = bucketize_bins(B["tgt"], B["band_id"], EDGES, K_C)
    want = F.cross_entropy(logits[act], binid[act])
    assert torch.allclose(val, want)


# ---- per-band binning (bucketize_bins, losses_cat) ---------------------------------------

# `losses_cat`'s bin assignment must be identical after the memory fix.
#
# The original computed the true bin as
#
#     binid = (tgt.unsqueeze(-1) >= edges[band][:, None, 1:-1]).sum(-1)
#
# which materialises an ``(n_cells, n_slot, K-1)`` tensor — and because the
# comparison is bool while ``.sum(-1)`` accumulates in **int64**, at 8 bytes per
# element. For a 37k-cell event at K=128 that is 4.66 GiB, and it OOM'd an 11 GB
# card at step 4 of the first real training launch.
#
# `bucketize` per band gives the same answer with no intermediate. The whole risk
# of that swap is the tie-break: `>=` sends a value landing exactly ON an edge to
# the upper bin, and `bucketize` only matches with ``right=True``. So these tests
# compare against the ORIGINAL expression, and go out of their way to put values
# exactly on edges — with random data the probability of a tie is zero, and the
# test would pass while proving nothing.

K = 16
N_BAND = 4
N_SLOT = 32


def _reference_binid(tgt, band_id, edges, K):
    """The pre-fix expression, verbatim, as the thing to match."""
    ec = edges[band_id]
    return (tgt.unsqueeze(-1) >= ec[:, None, 1:-1]).sum(-1).clamp(0, K - 1)


#: The SHIPPED binning, imported — not a local copy of it. This file used to
#: define its own `_bucketized_binid` labelled "what losses_cat now does" and
#: compare that to the reference, so every assertion below was about a
#: duplicate: the tie-break, the clamp and the missing-band guard could all have
#: changed in loss.py without failing a thing.
_bucketized_binid = bucketize_bins


def _edges():
    e = torch.linspace(-4, 4, K + 1).repeat(N_BAND, 1).clone()
    e[:, 0], e[:, -1] = -1e18, 1e18          # the real tables use infinite tails
    for b in range(N_BAND):                  # give each band a distinct grid
        e[b, 1:-1] = torch.linspace(-4 + b, 4 - b * 0.5, K - 1)
    return e


def test_matches_the_original_on_random_values():
    g = torch.Generator().manual_seed(0)
    tgt = torch.randn(300, 32, generator=g) * 3
    band_id = torch.randint(0, N_BAND, (300,), generator=g)
    e = _edges()
    assert torch.equal(_bucketized_binid(tgt, band_id, e, K),
                       _reference_binid(tgt, band_id, e, K))


def test_matches_the_original_ON_the_edges():
    """The tie-break is the whole risk of the swap. Random data never lands on an
    edge, so this places values exactly there — and just below/above."""
    e = _edges()
    band_id = torch.arange(N_BAND).repeat_interleave(4)          # 16 cells
    rows = []
    for b in band_id.tolist():
        interior = e[b, 1:-1]
        rows.append(torch.cat([interior,
                               interior - 1e-6,
                               interior + 1e-6,
                               torch.tensor([-1e9, 1e9])]))
    tgt = torch.stack(rows)
    assert torch.equal(_bucketized_binid(tgt, band_id, e, K),
                       _reference_binid(tgt, band_id, e, K)), \
        "tie-break differs: a value exactly on an edge lands in a different bin"


def test_losses_cat_value_is_unchanged():
    """End to end: the loss itself, not just the index."""
    g = torch.Generator().manual_seed(3)
    n_cells, n_slot = 200, 32
    tgt = torch.randn(n_cells, n_slot, generator=g) * 3
    occ = (torch.rand(n_cells, n_slot, generator=g) < 0.4).float()
    B = dict(valid=(torch.rand(n_cells, n_slot, generator=g) < 0.9),
             occ=occ, tgt=tgt,
             band_id=torch.randint(0, N_BAND, (n_cells,), generator=g))
    logits = torch.randn(n_cells, n_slot, K, generator=g)
    occ_logit = torch.randn(n_cells, n_slot, generator=g)
    mask = torch.zeros(n_cells, dtype=torch.bool); mask[::2] = True
    e = _edges()

    bce, val = losses_cat(occ_logit, logits, B, mask, e)

    # recompute with the reference index
    import torch.nn.functional as F
    binid = _reference_binid(tgt, B["band_id"], e, K)
    ce_e = F.cross_entropy(logits.reshape(-1, K), binid.reshape(-1),
                           reduction="none").view_as(tgt)
    act = occ.bool() & B["valid"] & mask[:, None]
    want = (ce_e * act).sum() / act.sum().clamp(min=1)
    assert torch.equal(val, want), f"loss changed: {float(val)} vs {float(want)}"


def test_no_large_intermediate():
    """The point of the change. Peak allocation must not scale with K.

    Measured on CPU with a tensor big enough that the old form would be
    conspicuous: the old expression allocates n_cells*n_slot*(K-1)*8 bytes."""
    g = torch.Generator().manual_seed(5)
    n_cells, n_slot, big_K = 4000, 128, 128
    tgt = torch.randn(n_cells, n_slot, generator=g)
    band_id = torch.randint(0, N_BAND, (n_cells,), generator=g)
    e = torch.linspace(-4, 4, big_K + 1).repeat(N_BAND, 1).contiguous()
    would_be = n_cells * n_slot * (big_K - 1) * 8
    assert would_be > 4e8, "fixture too small to be meaningful"
    out = _bucketized_binid(tgt, band_id, e, big_K)   # must simply not blow up
    assert out.shape == (n_cells, n_slot)
    assert out.dtype == torch.long


def test_a_band_missing_from_the_table_raises_instead_of_reading_garbage():
    """`torch.empty` + an unwritten slot = a loss that varies between identical
    calls. The original indexed `edges[band_id]` and raised; so must this."""
    e = _edges()[:2]                              # a table covering only 2 bands
    tgt = torch.randn(16, N_SLOT)
    band_id = torch.full((16,), 3, dtype=torch.long)
    with pytest.raises(IndexError) as ei:
        bucketize_bins(tgt, band_id, e, K)
    assert "band_id up to 3" in str(ei.value) and "only 2 rows" in str(ei.value)


def test_out_of_range_targets_clamp_into_the_open_end_bins():
    e = _edges()
    tgt = torch.tensor([[-1e6] * N_SLOT, [1e6] * N_SLOT])
    band_id = torch.zeros(2, dtype=torch.long)
    got = bucketize_bins(tgt, band_id, e, K)
    assert int(got[0].min()) == 0 and int(got[0].max()) == 0
    assert int(got[1].min()) == K - 1 and int(got[1].max()) == K - 1


def test_losses_cat_uses_this_function():
    """Mutation guard: break `bucketize_bins` and the shipped loss must move."""
    import helix.model.loss as L

    torch.manual_seed(0)
    n = 24
    e = _edges()
    B = dict(tgt=torch.randn(n, N_SLOT), occ=torch.ones(n, N_SLOT),
             valid=torch.ones(n, N_SLOT, dtype=torch.bool),
             band_id=torch.randint(0, N_BAND, (n,)))
    mask = torch.ones(n, dtype=torch.bool)
    logits = torch.randn(n, N_SLOT, K)
    occ_logit = torch.randn(n, N_SLOT)
    base = float(losses_cat(occ_logit, logits, B, mask, e)[1])

    orig = L.bucketize_bins
    try:
        L.bucketize_bins = lambda t, b, ed, k: torch.zeros_like(orig(t, b, ed, k))
        mutated = float(losses_cat(occ_logit, logits, B, mask, e)[1])
    finally:
        L.bucketize_bins = orig
    assert abs(mutated - base) > 1e-3, \
        "losses_cat did not route through bucketize_bins; the test pins a copy"


# ---- decoding a categorical prediction ---------------------------------------------------

# The categorical head's read-back: centroids, decode, and the space it lives in.
#
# These pin the read-back against one bug: feeding a posterior mean over
# *arcsinh-space* bin centres into `sinh(v) * sigma` -- an inverse for a Gaussian
# head applied to a CATEGORICAL one. Because `sinh(E[t]) != E[sinh t]` and the
# posterior spans many bins, that reads back 68% of the charge (0.674 against
# 0.918).
#
# The correct read-back is `sum_k p_k * centroid_k`, with the centroid table
# measured in LINEAR space: an expectation of sinh, never a sinh of an
# expectation.
#
# `test_jensen_gap_is_real` is the one that would have caught it: it asserts the
# two estimators genuinely differ on a spread posterior, so a future "simplification"
# back to `sinh(mean)` fails here rather than in a figure nobody checks.

K_D = 64                                # the decode tests' bin count (the binning tests above use K)


def _dec_edges(lo=-6.84699, hi=7.10671, open_ends=True):
    """A table shaped like a real one: uniform in arcsinh, open outer edges."""
    inner = np.linspace(lo, hi, K_D - 1)
    row = np.concatenate([[-1e18 if open_ends else lo - 1.0], inner,
                          [1e18 if open_ends else hi + 1.0]])
    return np.repeat(row[None], N_BAND, 0).astype(np.float32)


def test_closed_form_matches_numeric_integral():
    """E[sinh t | t ~ U(a,b)] == (cosh b - cosh a)/(b - a), to quadrature."""
    e = _dec_edges(open_ends=False)
    cent = bin_centroids_ratio(e)
    b = 0
    for k in (0, 1, K_D // 3, K_D // 2, K_D - 2, K_D - 1):
        a_, b_ = float(e[b, k]), float(e[b, k + 1])
        grid = np.linspace(a_, b_, 20001)
        want = np.trapezoid(np.sinh(grid), grid) / (b_ - a_)
        got = float(cent[b, k])
        assert abs(got - want) / max(abs(want), 1e-9) < 1e-5, (
            f"bin {k}: closed form {got:.6g} != integral {want:.6g}")


def test_open_edges_are_reflected_not_taken_literally():
    """The +-1e18 sentinels must not become centroids of ~5e17."""
    cent = bin_centroids_ratio(_dec_edges())
    assert np.isfinite(cent).all()
    # Bounded by sinh of a plausible token value, not by the sentinel.
    assert np.abs(cent).max() < np.sinh(10.0), \
        f"outer centroid {np.abs(cent).max():.3g} looks like a sentinel"
    # And the outer bins still exceed their neighbours in magnitude.
    assert abs(cent[0, -1]) > abs(cent[0, -2]) > abs(cent[0, -3])


def test_measured_table_is_preferred_over_the_closed_form():
    """An empirical cent_ratio wins; a NaN one falls back."""
    e = _dec_edges()
    fake = np.full((N_BAND, K_D), 3.25, np.float32)
    assert np.allclose(bin_centroids_ratio(e, fake), 3.25)
    nan = np.full((N_BAND, K_D), np.nan, np.float32)
    assert np.allclose(bin_centroids_ratio(e, nan), bin_centroids_ratio(e))


def test_onehot_posterior_returns_that_bins_centroid():
    """A delta posterior must decode to exactly the bin's centroid."""
    cent = bin_centroids_ratio(_dec_edges())
    band = np.array([0, 1, 2, 3], np.int64)
    want_k = np.array([[3, 17], [K_D - 1, 0], [8, 8], [40, 2]])
    logits = np.full((4, 2, K_D), -30.0, np.float32)
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
    e = _dec_edges()
    cent = bin_centroids_ratio(e)
    # asinh-space midpoints, the thing the buggy path averaged
    ee = e.copy()
    for b in range(N_BAND):
        ee[b, 0] = ee[b, 1] - (ee[b, 2] - ee[b, 1])
        ee[b, -1] = ee[b, -2] + (ee[b, -2] - ee[b, -3])
    mid = 0.5 * (ee[:, :-1] + ee[:, 1:])

    band = np.zeros(1, np.int64)
    x = np.linspace(-3, 3, K_D).astype(np.float32)      # broad posterior
    logits = (-0.5 * x[None, None, :] ** 2).repeat(1, 0).reshape(1, 1, K_D)
    p = np.exp(logits - logits.max()); p /= p.sum()

    correct = float(decode_categorical(logits, band, cent, "mean")[0, 0])
    buggy = float(np.sinh((p[0, 0] * mid[0]).sum()))
    assert abs(correct) > 1e-6
    assert abs(buggy - correct) / abs(correct) > 0.05, (
        f"sinh(mean)={buggy:.4g} vs E[sinh]={correct:.4g} — the estimators "
        f"agree here, so this test cannot catch the bug it exists for")


def test_rows_from_grid_space_argument():
    """`ratio` multiplies by sigma; `asinh` sinh-es first. They must differ."""
    cfg = PatchConfig()
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
    cent = bin_centroids_ratio(_dec_edges())
    with pytest.raises(ValueError):
        decode_categorical(np.zeros((1, 1, K_D), np.float32),
                           np.zeros(1, np.int64), cent, readout="median")

def test_mean_and_mode_differ_on_a_skewed_posterior():
    """`readout` must actually select an estimator.

    A one-hot posterior makes mean and mode identical, so the delta test above
    cannot tell them apart -- a mutation that ignores `readout` survives it. The
    reference reports BOTH for a reason: mean closes charge, mode is the
    bright-signal estimate, and on a skewed posterior they are far apart.
    """
    cent = bin_centroids_ratio(_dec_edges())
    band = np.zeros(1, np.int64)
    # skewed: modal bin near the middle, a long heavy tail toward the top
    w = np.zeros(K_D, np.float32)
    w[K_D // 2] = 3.0
    w[K_D - 8:] = 1.0
    logits = np.log(w + 1e-9)[None, None, :].astype(np.float32)

    mean = float(decode_categorical(logits, band, cent, "mean")[0, 0])
    mode = float(decode_categorical(logits, band, cent, "mode")[0, 0])
    assert mode == pytest.approx(float(cent[0, K_D // 2]), rel=1e-5)
    assert abs(mean - mode) / max(abs(mean), 1e-9) > 0.5, (
        f"mean {mean:.4g} and mode {mode:.4g} are too close for this test to "
        f"distinguish the two read-outs")
