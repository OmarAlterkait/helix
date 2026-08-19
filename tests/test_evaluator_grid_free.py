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
    core.set_bins(edges, cent_asinh=ca, cent_ratio=cr)
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


def test_edges_only_still_scores_via_derived_centroids():
    """A sidecar with no centroid tables must NOT silence the metric.

    This asserted the opposite for four commits: `_acc_grid_free` returned early
    when a centroid buffer was NaN, and the test pinned that as intended. It is
    the bug's own shape — the trainer then passed centroids into the wrong
    keyword slot, every charge number vanished from the log, and the only thing
    that would have complained was a test asserting they vanish.

    `set_bins` derives what it is not given, so edges alone are enough to score.
    """
    from helix.integrations.pimm import _acc_grid_free
    edges, _, _ = _tables()
    core = build_fm(dict(ARCH))
    core.set_bins(edges)                          # edges only — no centroids
    assert torch.isfinite(core.bin_cent_asinh).all()
    assert torch.isfinite(core.bin_cent_ratio).all()
    assert core.bin_cent_measured.tolist() == [0, 0], "derived, not measured"

    B = _batch()
    mask = torch.ones(B["tgt"].shape[0], dtype=torch.bool)
    gf = {k: 0.0 for k in ("sse", "sy", "syy", "nv", "chg_pred", "chg_true",
                           "chg_pred_s", "chg_true_s")}
    _acc_grid_free(core, B, mask,
                   torch.zeros(B["tgt"].shape[0], N_SLOT, K), gf)
    want = float((B["valid"].bool() & B["occ"].bool()).sum())
    assert gf["nv"] == want, "derived centroids must score the full support"
    assert gf["chg_true"] > 0.0


def test_derived_asinh_centroids_match_the_measured_convention():
    """The derived token-space table is the bin midpoint, closure included.

    `_tables()` builds its midpoints by hand with the same reflected-width rule;
    if the two ever disagree, one of them is inventing an outer edge differently
    and var_expl silently shifts.
    """
    from helix.model.tokenize import bin_centroids_asinh
    edges, ca, _ = _tables()
    assert np.allclose(bin_centroids_asinh(edges), ca, atol=1e-6)


def test_measured_centroids_are_flagged_and_kept_verbatim():
    edges, ca, cr = _tables()
    core = build_fm(dict(ARCH))
    core.set_bins(edges, cent_ratio=cr)           # one measured, one derived
    assert core.bin_cent_measured.tolist() == [0, 1]
    assert np.allclose(core.bin_cent_ratio.numpy(), cr, atol=1e-6), \
        "a measured table must not be re-derived"
    assert not np.allclose(core.bin_cent_asinh.numpy(),
                           core.bin_cent_ratio.numpy()), \
        "the two tables live in different spaces; sharing values means a mix-up"


def test_centroids_are_keyword_only():
    """The mis-binding that disabled the metrics must be a TypeError now.

    Three call sites read `set_bins(edges, cent_asinh, cent_lin)`. When the
    signature grew a fourth table the third argument bound to `cent_ratio`, so
    the charge read-back silently used raw-ADC means. Positional is the hazard,
    not the particular table.
    """
    edges, ca, cr = _tables()
    core = build_fm(dict(ARCH))
    with pytest.raises(TypeError):
        core.set_bins(edges, ca, cr)


def test_apply_bins_is_the_one_seam_and_tolerates_old_sidecars():
    """`apply_bins` takes the sidecar mapping whole, unknown keys and all."""
    from helix.model.checkpoint import apply_bins
    edges, ca, cr = _tables()
    core = build_fm(dict(ARCH))
    # `cent_lin` is what pre-consolidation sidecars on disk still carry.
    apply_bins(core, dict(edges=edges, cent_asinh=ca, cent_ratio=cr,
                          cent_lin=np.zeros_like(ca), K=K, corpus="r1"))
    assert np.allclose(core.bin_cent_ratio.numpy(), cr, atol=1e-6)
    assert core.bin_cent_measured.tolist() == [1, 1]
    assert not hasattr(core, "bin_cent_lin"), "the biased table must not come back"


def test_apply_bins_warns_before_overwriting_different_edges():
    """Replacing a checkpoint's own grid with a sidecar's must not be silent."""
    from helix.model.checkpoint import apply_bins
    edges, ca, cr = _tables()
    core = build_fm(dict(ARCH))
    core.set_bins(edges)
    seen = []
    apply_bins(core, dict(edges=edges), log=seen.append)
    assert seen == [], "identical edges are not an overwrite worth reporting"
    apply_bins(core, dict(edges=edges * np.float32(1.5)), log=seen.append)
    assert seen and "max|delta|" in seen[0]


def test_backfill_derives_centroids_for_a_pre_fix_checkpoint():
    """An old state_dict with edges but no centroids must load AND be usable."""
    from helix.model.checkpoint import _backfill_centroids
    edges, ca, cr = _tables()
    core = build_fm(dict(ARCH))
    core.set_bins(edges, cent_asinh=ca, cent_ratio=cr)
    sd = {k: v for k, v in core.state_dict().items()
          if not k.startswith("bin_cent")}
    sd["bin_cent_lin"] = torch.zeros(N_BAND, K)     # a stale buffer to drop
    out = _backfill_centroids(core, sd)
    assert "bin_cent_lin" not in out, "a removed buffer must not reach strict load"
    fresh = build_fm(dict(ARCH))
    fresh.load_state_dict(out, strict=True)         # the property that matters
    assert torch.isfinite(fresh.bin_cent_ratio).all(), \
        "backfill must derive, not copy the constructor's NaN"
    assert fresh.bin_cent_measured.tolist() == [0, 0]
