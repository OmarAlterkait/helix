"""RankMe: the Gram shortcut must equal the reference's svdvals, and the metric
must behave like an effective rank.

The reference (``fm/feats_rank.py``) centres the feature matrix and calls
``torch.linalg.svdvals`` directly. At corpus scale N is ~1.5M rows, where that is
O(N d^2) and intractable, so ``scripts/feats_rank.py`` goes through the Gram
matrix instead. That is the same quantity mathematically; these tests hold it to
that, because "mathematically the same" is exactly the kind of claim that turns
out to be off by a sinh.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from feats_rank import rankme  # noqa: E402


def test_gram_matches_svdvals():
    """The shortcut and the reference formula agree to float tolerance."""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(4000, 64, generator=g)
    a = rankme(X, exact=True)          # reference: svdvals on centred X
    b = rankme(X, exact=False)         # ours: sqrt(eigvalsh(X^T X))
    assert abs(a - b) / a < 1e-4, f"gram {b:.6f} != svdvals {a:.6f}"


def test_isotropic_features_are_full_rank():
    """Independent unit-variance columns => RankMe near d."""
    g = torch.Generator().manual_seed(1)
    d = 32
    X = torch.randn(20000, d, generator=g)
    assert rankme(X) > 0.9 * d, "isotropic features should be near full rank"


def test_collapsed_features_have_rank_one():
    """A rank-1 feature matrix must score ~1, whatever its scale."""
    g = torch.Generator().manual_seed(2)
    v = torch.randn(48, generator=g)
    X = torch.randn(5000, 1, generator=g) @ v[None, :]
    assert rankme(X) < 1.05, f"rank-1 input scored {rankme(X):.3f}"


def test_scale_invariant():
    """RankMe normalises the singular values, so a global rescale cannot move it."""
    g = torch.Generator().manual_seed(3)
    X = torch.randn(3000, 24, generator=g)
    assert abs(rankme(X) - rankme(X * 137.0)) < 1e-3


def test_centering_matters_and_is_applied():
    """A large constant offset is not signal; RankMe must be blind to it.

    Without centring, a shared offset appears as one dominant singular value and
    drags the effective rank toward 1 — the metric would then report "collapsed"
    for a perfectly healthy representation that merely has a mean.
    """
    g = torch.Generator().manual_seed(4)
    X = torch.randn(8000, 32, generator=g)
    shifted = X + 50.0
    assert abs(rankme(X) - rankme(shifted)) / rankme(X) < 1e-3


def test_degenerate_input_does_not_raise():
    """Zero features are a real failure mode; report, don't crash."""
    X = torch.zeros(100, 16)
    r = rankme(X)
    assert np.isfinite(r), f"zero features gave {r}"


def test_streaming_the_gram_matches_one_pass():
    """`main` accumulates the Gram batch by batch; it must land on the same
    number as a single pass over the whole matrix.

    `main` had its own inline copy of the centre/eigen/entropy tail, so THIS —
    the path that actually produced every RankMe number reported in this project
    — was never the thing under test. `rankme` was.
    """
    import torch
    from feats_rank import rank_from_gram, rankme

    g_ = torch.Generator().manual_seed(11)
    X = torch.randn(4000, 24, generator=g_)
    X[:, 3:] *= 0.05                              # a real spectrum, not white

    d = X.shape[1]
    gram = torch.zeros(d, d, dtype=torch.float64)
    colsum = torch.zeros(d, dtype=torch.float64)
    n = 0
    for chunk in X.split(137):                    # deliberately uneven batches
        gram += (chunk.T @ chunk).double()
        colsum += chunk.sum(0).double()
        n += chunk.shape[0]

    assert n == X.shape[0]
    streamed = rank_from_gram(gram, colsum, n)
    assert streamed == pytest.approx(rankme(X), rel=1e-6)
    assert streamed == pytest.approx(rankme(X, exact=True), rel=1e-3), \
        "the streamed Gram must agree with svdvals, not just with itself"


def test_gram_centring_is_not_a_no_op():
    """`cov = X^T X - n mu mu^T`. Drop the correction and an offset column
    dominates the spectrum, which is the whole reason RankMe centres."""
    import torch
    from feats_rank import rank_from_gram

    g_ = torch.Generator().manual_seed(3)
    X = torch.randn(2000, 12, generator=g_)
    X += 50.0                                     # a large common offset
    gram = (X.T @ X).double()
    colsum = X.sum(0).double()
    centred = rank_from_gram(gram, colsum, X.shape[0])
    uncentred = rank_from_gram(gram, torch.zeros_like(colsum), X.shape[0])
    assert uncentred < 2.0, "an uncentred offset collapses the rank to ~1"
    assert centred > 10.0, f"centred rank should be near full, got {centred}"
