"""The evaluator's grid-free metrics: var_expl and charge closure.

These are the reference's actual headline numbers (fm/train.py::perband_mse_cat,
printed every eval by mae_ddp.py:216-220). They were never ported, and that gap
is why a 31%-low charge read-back and a cross-bin-table CE comparison both
survived a long time: nothing in the run reported a quantity that would have
moved when either was wrong.

var_expl is grid-free — it compares a posterior-mean reconstruction against the
target's own measured variance — so unlike cross-entropy it stays meaningful
across a change of bin table or corpus.
"""
import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm  # noqa: E402


def _pimm_importable():
    try:
        if importlib.util.find_spec("pimm") is None:
            return False
        import pimm.datasets.builder  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pimm_importable(), reason="pimm not importable")

N_BAND, K, N_SLOT = 4, 16, 8
ARCH = dict(n_slot=N_SLOT, n_band=N_BAND, n_plane=6, d=32, blocks=1, dec_blocks=1,
            heads=4, dec_mode="cross", n_bins=K)


def _tables():
    """Edges + the two centroid tables, consistent with each other."""
    inner = np.linspace(-3.0, 3.0, K - 1)
    edges = np.concatenate([[-1e18], inner, [1e18]])[None].repeat(N_BAND, 0)
    mid = np.empty((N_BAND, K), np.float64)
    for b in range(N_BAND):
        e = edges[b].copy()
        e[0] = e[1] - (e[2] - e[1])
        e[-1] = e[-2] + (e[-2] - e[-3])
        mid[b] = 0.5 * (e[:-1] + e[1:])
    return (edges.astype(np.float32), mid.astype(np.float32),
            np.sinh(mid).astype(np.float32))       # cent_ratio == E[sinh] here


def _batch(n_cells=24, seed=3):
    g = np.random.default_rng(seed)
    occ = (g.random((n_cells, N_SLOT)) < 0.5)
    tgt = g.normal(0, 1.2, (n_cells, N_SLOT)).astype(np.float32)
    return dict(
        tgt=torch.from_numpy(tgt),
        occ=torch.from_numpy(occ.astype(np.float32)),
        valid=torch.ones(n_cells, N_SLOT, dtype=torch.float32),
        band_id=torch.from_numpy(g.integers(0, N_BAND, n_cells)),
    )


def _run(logits, B, mask):
    from helix.integrations.pimm import _acc_grid_free
    edges, ca, cr = _tables()
    core = build_fm(dict(ARCH))
    core.set_bins(edges, cent_asinh=ca, cent_lin=ca, cent_ratio=cr)
    gf = {k: 0.0 for k in ("sse", "sy", "syy", "nv", "chg_pred", "chg_true",
                           "chg_pred_s", "chg_true_s")}
    _acc_grid_free(core, B, mask, logits, gf)
    mean_y = gf["sy"] / gf["nv"]
    var_y = max(gf["syy"] / gf["nv"] - mean_y ** 2, 1e-12)
    return (1.0 - (gf["sse"] / gf["nv"]) / var_y,
            gf["chg_pred"] / gf["chg_true"], gf)


def test_perfect_prediction_scores_one():
    """One-hot on the true bin => var_expl ~ 1 and closure ~ 1.

    Not exact: the centroid of the true bin is not the true value, so the
    residual is the quantisation error. That floor is the point — it says how
    much of the metric a 128-bin head can ever recover.
    """
    B, mask = _batch(), None
    n = B["tgt"].shape[0]
    mask = torch.ones(n, dtype=torch.bool)
    edges, ca, _ = _tables()
    band = B["band_id"].numpy()
    # bucketize the target onto the same grid the metric decodes from
    binid = np.stack([np.digitize(B["tgt"].numpy()[i], edges[band[i], 1:-1])
                      for i in range(n)])
    logits = torch.full((n, N_SLOT, K), -20.0)
    for i in range(n):
        for j in range(N_SLOT):
            logits[i, j, int(binid[i, j])] = 20.0

    ve, cc, _ = _run(logits, B, mask)
    assert ve > 0.97, f"perfect binning should explain ~all variance, got {ve:.4f}"
    # UNSIGNED closure. The signed ratio is meaningless on a symmetric target:
    # Sum(sinh(y)) is a small difference of large numbers, and a perfectly-binned
    # predictor scores ~0.008 on it. That instability is why the metric reports
    # unsigned as the closure and signed only as a bias indicator.
    assert 0.9 < cc < 1.1, f"perfect binning should close charge, got {cc:.4f}"


def test_uninformative_prediction_scores_zero():
    """A constant posterior explains ~none of the variance."""
    B = _batch()
    n = B["tgt"].shape[0]
    mask = torch.ones(n, dtype=torch.bool)
    logits = torch.zeros(n, N_SLOT, K)          # uniform over bins
    ve, _, _ = _run(logits, B, mask)
    assert ve < 0.05, f"uniform posterior should explain ~0 variance, got {ve:.4f}"


def test_scored_only_on_masked_and_occupied():
    """The support must match the value loss: masked & valid & occupied."""
    B = _batch()
    n = B["tgt"].shape[0]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[: n // 2] = True
    logits = torch.zeros(n, N_SLOT, K)
    _, _, gf = _run(logits, B, mask)
    want = float((mask[:, None] & B["valid"].bool() & B["occ"].bool()).sum())
    assert gf["nv"] == want, f"scored {gf['nv']} slots, support is {want}"


def test_absent_centroids_are_skipped_not_guessed():
    """Without the centroid tables the metric must emit nothing at all."""
    from helix.integrations.pimm import _acc_grid_free
    edges, _, _ = _tables()
    core = build_fm(dict(ARCH))
    core.set_bins(edges)                         # edges only; centroids stay NaN
    B = _batch()
    mask = torch.ones(B["tgt"].shape[0], dtype=torch.bool)
    gf = {k: 0.0 for k in ("sse", "sy", "syy", "nv", "chg_pred", "chg_true",
                           "chg_pred_s", "chg_true_s")}
    _acc_grid_free(core, B, mask,
                   torch.zeros(B["tgt"].shape[0], N_SLOT, K), gf)
    assert gf["nv"] == 0.0, "metric ran without centroids — it would be wrong"
