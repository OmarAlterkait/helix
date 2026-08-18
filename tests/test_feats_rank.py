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
