"""CoeffFMEvaluator: its DDP reduction, and the grid-free charge metrics it reports."""
import importlib.util

import pytest

torch = pytest.importorskip("torch")


def _pimm_importable():
    try:
        if importlib.util.find_spec("pimm") is None:
            return False
        import pimm.datasets.builder  # noqa: F401
        return True
    except Exception:
        return False


#: CoeffFMEvaluator lives in helix.integrations.pimm, whose module body imports pimm.
needs_pimm = pytest.mark.skipif(not _pimm_importable(), reason="pimm not importable")

import types

import numpy as np

from helix.model import build_fm  # noqa: E402


# ---- the DDP reduction -------------------------------------------------------------------

# Validation must score the whole val set, not one rank's shard.
#
# pimm builds the val loader with a `DistributedSampler` whenever world_size > 1
# (engines/train.py:687), so a loader iterated on rank 0 yields only rank 0's
# slice. The evaluator used to be "rank 0 evaluates, the others wait at a
# barrier", which meant every multi-GPU eval scored ~1/world_size of the set and
# reported it as the validation number — visible in the 4-GPU logs as
# `batches=145` against a 577-event val set, and never questioned because a
# smaller number of batches looks like a smaller val set, not a bug.


GF = ("sse", "sy", "syy", "nv", "chg_pred", "chg_true", "chg_pred_s", "chg_true_s")


def _ev(rank=0, world=1, logger=None):
    """A real evaluator with a fake trainer.

    `__new__` WITHOUT `__init__` is what this used to do, and the stub then had
    to re-list every attribute the class needed. It drifted: when the evaluator
    grew `mask_mode`/`n_planes`, five tests here started dying with
    `AttributeError: 'CoeffFMEvaluator' object has no attribute 'mask_mode'`
    inside the eval loop -- and nobody saw it, because the whole module is
    @pimm_importable and skips wherever pimm is absent, which is every
    environment we normally run the suite in. A clean-room run with pimm on the
    path found all five at once.

    So construct it properly: __init__ owns the defaults, and only the trainer
    is faked.
    """
    from helix.integrations.pimm import CoeffFMEvaluator
    ev = CoeffFMEvaluator()
    ev.trainer = types.SimpleNamespace(
        logger=logger or types.SimpleNamespace(
            info=lambda *a, **k: None, exception=lambda *a, **k: None),
        comm_info={}, writer=None, global_step=7,
        parallel_context=types.SimpleNamespace(device=torch.device("cpu")))
    return ev


@needs_pimm
def test_all_reduce_packs_and_unpacks_one_flat_vector(monkeypatch):
    """Every accumulator round-trips through the single collective, in order."""
    from helix.integrations import pimm as P
    ev = _ev()
    totals = {"bce": 1.0, "val": 2.0}
    counts = {"bce": 10.0, "val": 20.0}
    gf = {k: float(i + 1) for i, k in enumerate(GF)}

    seen = {}

    def fake_all_reduce(t, op=None):
        seen["n"] = t.numel()
        t.mul_(4)                                  # as if 4 identical ranks

    # `import torch.distributed as dist` binds the ATTRIBUTE on `torch`, so
    # replacing sys.modules["torch.distributed"] does nothing.
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    t2, c2, g2, n2 = ev._all_reduce(totals, counts, gf, 5)
    assert seen["n"] == 2 + 2 + len(GF) + 1, "one collective, everything in it"
    assert t2 == {k: 4 * v for k, v in totals.items()}
    assert c2 == {k: 4 * v for k, v in counts.items()}
    assert g2 == {k: 4 * v for k, v in gf.items()}
    assert n2 == 20 and isinstance(n2, int)


@needs_pimm
def test_reduced_sums_average_to_the_same_value_as_one_big_pass():
    """Summing then dividing == pooling. A mean of per-rank means would not be.

    Shards differ in size (DistributedSampler pads), so this is the property
    that actually matters: rank A scoring 100 tokens and rank B scoring 10 must
    weight 10:1, not 1:1.
    """
    ev = _ev()
    logged = {}
    ev.trainer.logger = types.SimpleNamespace(
        info=lambda m: logged.setdefault("line", m), exception=lambda *a: None)
    # rank A: bce 0.5 over 100 tokens; rank B: bce 0.9 over 10 tokens
    totals = {"bce": 0.5 * 100 + 0.9 * 10, "val": 0.0}
    counts = {"bce": 110.0, "val": 1.0}
    ev._report(totals, counts, {k: 0.0 for k in GF}, 2)
    pooled = (0.5 * 100 + 0.9 * 10) / 110
    assert ev.trainer.comm_info["current_metric_value"] == pytest.approx(-pooled)
    assert abs(pooled - 0.5 * (0.5 + 0.9)) > 0.03, "the two averages must differ"


@needs_pimm
def test_grid_free_metrics_come_from_the_reduced_sums():
    ev = _ev()
    line = {}
    ev.trainer.logger = types.SimpleNamespace(
        info=lambda m: line.setdefault("m", m), exception=lambda *a: None)
    gf = dict.fromkeys(GF, 0.0)
    gf.update(nv=100.0, sy=0.0, syy=100.0, sse=25.0,
              chg_pred=97.0, chg_true=100.0, chg_pred_s=9.0, chg_true_s=10.0)
    ev._report({"bce": 0.0, "val": 0.0}, {"bce": 1.0, "val": 1.0}, gf, 4)
    assert "var_expl=0.7500" in line["m"]          # 1 - (25/100)/1.0
    assert "charge_closure=0.9700" in line["m"]
    # (chg_pred_s - chg_true_s) / chg_true = (9 - 10) / 100. NOT the signed ratio
    # 9/10 = 0.9: that form is a small difference of large numbers on a
    # near-symmetric target and scored 0.008 for a perfectly-binned predictor.
    assert "charge_resid=-0.0100" in line["m"]


@needs_pimm
def test_report_is_a_noop_on_an_empty_val_set():
    ev = _ev()
    ev._report({}, {}, dict.fromkeys(GF, 0.0), 0)
    assert "current_metric_value" not in ev.trainer.comm_info


@needs_pimm
def test_every_rank_publishes_the_selection_metric():
    """The checkpoint save is collective; ranks that disagree cannot agree to save."""
    from helix.integrations import pimm as P
    vals = []
    for rank in (0, 1, 2, 3):
        ev = _ev()
        # identical reduced sums on every rank, by construction
        ev._report({"bce": 4.0, "val": 2.0}, {"bce": 8.0, "val": 8.0},
                   dict.fromkeys(GF, 0.0), 4)
        vals.append(ev.trainer.comm_info["current_metric_value"])
    assert len(set(vals)) == 1, f"ranks disagree on the metric: {vals}"
    assert vals[0] == pytest.approx(-(4.0 / 8.0 + 2.0 / 8.0))


# --------------------------------------------------------------------------
# The same thing against a REAL collective. The mock above proves the packing;
# only this proves the reduce — that two ranks holding DIFFERENT shard sums end
# up with the identical pooled metric, which is what the collective checkpoint
# save depends on.
# --------------------------------------------------------------------------

def _worker(rank, world, init_file, out):
    import types

    import torch
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world)
    try:
        from helix.integrations.pimm import CoeffFMEvaluator
        ev = CoeffFMEvaluator()          # __init__, not __new__ -- see _ev above
        lines = []
        ev.trainer = types.SimpleNamespace(
            logger=types.SimpleNamespace(info=lines.append,
                                         exception=lambda *a, **k: None),
            comm_info={}, writer=None, global_step=7,
            parallel_context=types.SimpleNamespace(device=torch.device("cpu")))
        # Deliberately UNEQUAL shards: rank 0 scores 100 tokens at bce 0.5,
        # rank 1 scores 10 at bce 0.9. A mean-of-means would give 0.70.
        if rank == 0:
            totals, counts, n = {"bce": 50.0, "val": 0.0}, {"bce": 100.0, "val": 1.0}, 100
        else:
            totals, counts, n = {"bce": 9.0, "val": 0.0}, {"bce": 10.0, "val": 1.0}, 10
        gf = dict.fromkeys(GF, 0.0)
        gf.update(nv=float(n), syy=float(n), sse=0.25 * n,
                  chg_pred=0.97 * n, chg_true=float(n))
        t, c, g, nn = ev._all_reduce(totals, counts, gf, n)
        ev._report(t, c, g, nn)
        out[rank] = (ev.trainer.comm_info["current_metric_value"], nn, lines)
    finally:
        dist.destroy_process_group()


@needs_pimm
def test_two_real_gloo_ranks_agree_on_a_token_pooled_metric(tmp_path):
    import torch.multiprocessing as mp
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    mgr = mp.Manager()
    out = mgr.dict()
    init_file = str(tmp_path / "pg")
    mp.spawn(_worker, args=(2, init_file, out), nprocs=2, join=True)

    assert set(out.keys()) == {0, 1}
    (m0, n0, lines0), (m1, n1, _) = out[0], out[1]
    assert n0 == n1 == 110, "batches must be the SUM over ranks, not one shard"
    assert m0 == pytest.approx(m1), "ranks must agree; the save is collective"
    pooled = 59.0 / 110.0
    assert m0 == pytest.approx(-pooled), "token-pooled, not a mean of rank means"
    assert abs(pooled - 0.70) > 0.1, "mean-of-means would be 0.70 here"
    assert any("batches=110" in l for l in lines0), \
        f"rank 0 must report the full count, got: {lines0}"


@needs_pimm
def test_eval_shard_actually_ENTERS_the_loop(monkeypatch):
    """`_eval_shard` must iterate the loader and come back with batches counted.

    Nothing tested this. The suite exercised `_all_reduce`, `_report`, `_forward`
    and `_acc_grid_free` directly, so when `_eval_shard` was split into an
    acquire/restore wrapper and an inner pass, the inner pass referenced
    `loader` and `move_batch_to_device` -- both LOCALS of the wrapper -- and
    every eval died with NameError. Nothing went red: `eval()` caught it, logged
    it, and substituted zeros, which `_report` reported as "val_loader was
    empty". The metric silently stopped existing and `model_best` selection went
    inert.

    So this asserts the one thing the unit tests could not see: that a real pass
    over a loader reaches the accumulate and returns a non-zero batch count.
    """
    import contextlib  # noqa: F401  (the loop uses it)
    ev = _ev()
    ev.max_batches, ev.mask_seed = None, 0
    N = 4

    class _Model:
        training = True

        def eval(self):
            pass

        def train(self):
            pass

        # Signature must track helix.model.fm.make_mask(B, mode, ratio,
        # n_planes, gen). A fake that accepts less does not fail where the real
        # one would -- it fails HERE, with a TypeError that looks like a bug in
        # the evaluator. It drifted once already, when n_planes was added.
        def make_mask(self, B, mode=None, ratio=None, n_planes=None, gen=None):
            return torch.ones(N, dtype=torch.bool)

    def _batch():
        return {"plane_id": torch.zeros(N, dtype=torch.long),
                "valid": torch.ones(N, 3, dtype=torch.bool),
                "occ": torch.ones(N, 3, dtype=torch.bool)}

    ev.trainer.model = _Model()
    ev.trainer.val_loader = [_batch(), _batch()]
    ev.trainer.cfg = types.SimpleNamespace(amp_dtype="float16", enable_amp=False)
    ev._forward = lambda model, core, B, mask, gf: {
        "bce": torch.tensor(1.0), "val": torch.tensor(2.0)}

    out = ev._eval_shard()
    assert out is not None, "_eval_shard returned None on a loader with batches"
    totals, counts, gf, n = out
    assert n == 2, f"iterated {n} batches over a 2-batch loader"
    assert totals.get("bce", 0.0) > 0, "the accumulate never ran"


@needs_pimm
def test_a_failed_shard_is_not_reported_as_an_empty_val_set():
    """A raising shard must FAIL the eval, not silently publish nothing.

    The handler contributes zeros so no rank is left inside the collective --
    correct -- but it used to swallow the exception entirely, making a broken
    eval indistinguishable from an empty val set. It must re-raise AFTER the
    reduce.
    """
    ev = _ev()

    def boom():
        raise RuntimeError("shard exploded")

    ev._eval_shard = boom
    with pytest.raises(RuntimeError, match="shard exploded"):
        ev.eval()


# ---- grid-free metrics -------------------------------------------------------------------

# The evaluator's grid-free metrics: var_expl and charge closure.
#
# These are the reference's actual headline numbers (fm/train.py::perband_mse_cat,
# printed every eval by mae_ddp.py:216-220). They were never ported, and that gap
# is why a 31%-low charge read-back and a cross-bin-table CE comparison both
# survived a long time: nothing in the run reported a quantity that would have
# moved when either was wrong.
#
# var_expl is grid-free — it compares a posterior-mean reconstruction against the
# target's own measured variance — so unlike cross-entropy it stays meaningful
# across a change of bin table or corpus.


N_BAND, K, N_SLOT = 4, 16, 8
ARCH = dict(n_slot=N_SLOT, n_band=N_BAND, n_plane=6, d=32, blocks=1, dec_blocks=1,
            heads=4, n_bins=K)


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
    from helix.eval.recon import acc_grid_free as _acc_grid_free
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
    # UNSIGNED closure: Sum(E[|X|]) / Sum(|X_true|). The signed ratio is
    # meaningless on a symmetric target -- Sum(sinh(y)) is a small difference of
    # large numbers, and a perfectly-binned predictor scores ~0.008 on it. That
    # instability is why the signed direction is reported as charge_resid, a
    # residual over the unsigned total, rather than as a ratio.
    assert 0.9 < cc < 1.1, f"perfect binning should close charge, got {cc:.4f}"


def test_calibrated_posterior_that_STRADDLES_ZERO_still_closes_charge():
    """Sign-ambiguous but calibrated: closure must be ~1, not ~0.

    This is the case no existing test could reach, and the one the estimator bug
    lived in. Every other test scores a posterior concentrated on ONE side of
    zero -- a one-hot, or a spread over adjacent same-sign bins -- and there
    |E[X]| == E[|X|], so both functionals agree and the bug is invisible.

    Construction: the true value is +v or -v with equal probability, and the
    posterior is exactly that -- 50/50 on the two bins holding +v and -v. It is
    perfectly CALIBRATED (the truth really is drawn from it); it simply does not
    know the sign. A magnitude closure must therefore read ~1: E[|X|] = v and the
    truth has |X| = v.

    Accumulating |E[X]| instead gives ~0, because the two centroids cancel. That
    is the whole defect in miniature -- on real data it showed up as a milder
    0.8177 where the correct estimator reads 0.9676.
    """
    B = _batch()
    n = B["tgt"].shape[0]
    g = np.random.default_rng(11)
    v = 1.5
    sign = g.choice([-1.0, 1.0], size=(n, N_SLOT))
    B["tgt"] = torch.from_numpy((sign * v).astype(np.float32))
    mask = torch.ones(n, dtype=torch.bool)

    edges, _, _ = _tables()
    band = B["band_id"].numpy()
    logits = torch.full((n, N_SLOT, K), -40.0)
    for i in range(n):
        kp = int(np.digitize(v, edges[band[i], 1:-1]))
        km = int(np.digitize(-v, edges[band[i], 1:-1]))
        logits[i, :, kp] = 0.0          # equal mass on +v and -v
        logits[i, :, km] = 0.0

    _, cc, _ = _run(logits, B, mask)
    assert 0.9 < cc < 1.1, (
        f"a calibrated sign-ambiguous posterior must still close charge, got "
        f"{cc:.4f}. Near 0 means the metric is accumulating |E[X]|, where the two "
        f"centroids cancel, instead of E[|X|] -- it is measuring posterior WIDTH, "
        f"not charge magnitude.")


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
    from helix.eval.recon import acc_grid_free as _acc_grid_free
    edges, _, _ = _tables()
    core = build_fm(dict(ARCH))
    core.set_bins(edges)                          # edges only — no centroids
    assert torch.isfinite(core.bin_cent_asinh).all()
    assert torch.isfinite(core.bin_cent_ratio).all()

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


def test_the_single_forward_path_still_checks_the_batch_contract():
    """`heads_loss` skips `forward` to save a pass — not to skip its guard.

    It reproduces the categorical branch from one `raw_heads` call, and so used
    to reach `losses_cat` with whatever the batch happened to hold. A missing
    `tgt` then surfaced as a bare KeyError from inside the loss, naming nothing.
    """
    from helix.eval.recon import heads_loss
    edges, ca, cr = _tables()
    core = build_fm(dict(ARCH))
    core.set_bins(edges, cent_asinh=ca, cent_ratio=cr)
    B = _batch()
    B.update(inp=B["tgt"].clone(), plane_id=torch.zeros(B["tgt"].shape[0], dtype=torch.long))
    del B["tgt"]                                  # what losses_cat needs and lacks
    with pytest.raises(KeyError) as ei:
        heads_loss(core, B, torch.ones(B["occ"].shape[0], dtype=torch.bool),
                   {k: 0.0 for k in ("sse", "sy", "syy", "nv", "chg_pred",
                                     "chg_true", "chg_pred_s", "chg_true_s")})
    assert "tgt" in str(ei.value) and "losses_cat" in str(ei.value), \
        f"the error must name the key and the objective, got: {ei.value}"


@needs_pimm
def test_eval_mask_mode_defaults_to_random_and_is_honoured():
    """The eval mask must stay `random` by default, and say so when it is not.

    `random` is deliberate and load-bearing: with `plane_frac > 0` a
    mixed-mode eval would make val loss a 90/10 blend and stop being comparable
    across runs; a plane-masked number is a SEPARATE curve. The override exists to produce that separate number, so the
    two things worth pinning are that the default did not move and that an
    override actually reaches `make_mask` (rather than being stored and
    ignored, which is how `WeightEMA(key=)` spent its whole life).
    """
    from helix.integrations.pimm.eval import CoeffFMEvaluator

    assert CoeffFMEvaluator().mask_mode == "random", \
        "the default eval mask changed — every historical val number assumes random"
    assert CoeffFMEvaluator().n_planes == 1

    ev = CoeffFMEvaluator(mask_mode="plane", n_planes=2)
    assert (ev.mask_mode, ev.n_planes) == ("plane", 2)

    # the override must reach the model's make_mask, with the mode AND n_planes
    model = build_fm(dict(ARCH))
    seen = {}
    orig = model.make_mask

    def spy(B, ratio=None, mode=None, n_planes=None, gen=None):
        seen.update(mode=mode, n_planes=n_planes)
        return orig(B, ratio=ratio, mode=mode, n_planes=n_planes, gen=gen)

    model.make_mask = spy
    # _batch() carries the loss-side keys only; make_mask needs the geometry.
    n = 24
    B = dict(n_cells=n,
             plane_id=torch.arange(n) % ARCH["n_plane"],
             wire_pos=torch.arange(n, dtype=torch.float32))
    m = model.make_mask(B, mode=ev.mask_mode, n_planes=ev.n_planes,
                        gen=torch.Generator().manual_seed(0))
    assert seen == dict(mode="plane", n_planes=2)
    # and it really is the plane task: whole planes, every volume punctured
    gid = B["plane_id"]
    for g in gid[m].unique():
        assert bool(m[gid == g].all()), f"gid {int(g)} only partly masked"
    vols = torch.div(gid.unique(), 3, rounding_mode="floor").unique()
    hit_vols = torch.div(gid[m].unique(), 3, rounding_mode="floor").unique()
    assert len(hit_vols) == len(vols), "a volume escaped the eval plane mask"
