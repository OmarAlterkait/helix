"""Validation must score the whole val set, not one rank's shard.

pimm builds the val loader with a `DistributedSampler` whenever world_size > 1
(engines/train.py:687), so a loader iterated on rank 0 yields only rank 0's
slice. The evaluator used to be "rank 0 evaluates, the others wait at a
barrier", which meant every multi-GPU eval scored ~1/world_size of the set and
reported it as the validation number — visible in the 4-GPU logs as
`batches=145` against a 577-event val set, and never questioned because a
smaller number of batches looks like a smaller val set, not a bug.
"""
import importlib.util
import types

import numpy as np
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


pytestmark = pytest.mark.skipif(not _pimm_importable(), reason="pimm not importable")

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


def test_all_reduce_packs_and_unpacks_one_flat_vector(monkeypatch):
    """Every accumulator round-trips through the single collective, in order."""
    from helix.integrations import pimm as P
    ev = _ev()
    totals = {"bce": 1.0, "val": 2.0, "masked_frac": 3.0}
    counts = {"bce": 10.0, "val": 20.0, "masked_frac": 30.0}
    gf = {k: float(i + 1) for i, k in enumerate(GF)}

    seen = {}

    def fake_all_reduce(t, op=None):
        seen["n"] = t.numel()
        t.mul_(4)                                  # as if 4 identical ranks

    # `import torch.distributed as dist` binds the ATTRIBUTE on `torch`, so
    # replacing sys.modules["torch.distributed"] does nothing.
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    t2, c2, g2, n2 = ev._all_reduce(totals, counts, gf, 5)
    assert seen["n"] == 3 + 3 + len(GF) + 1, "one collective, everything in it"
    assert t2 == {k: 4 * v for k, v in totals.items()}
    assert c2 == {k: 4 * v for k, v in counts.items()}
    assert g2 == {k: 4 * v for k, v in gf.items()}
    assert n2 == 20 and isinstance(n2, int)


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
    totals = {"bce": 0.5 * 100 + 0.9 * 10, "val": 0.0, "masked_frac": 0.0}
    counts = {"bce": 110.0, "val": 1.0, "masked_frac": 1.0}
    ev._report(totals, counts, {k: 0.0 for k in GF}, 2)
    pooled = (0.5 * 100 + 0.9 * 10) / 110
    assert ev.trainer.comm_info["current_metric_value"] == pytest.approx(-pooled)
    assert abs(pooled - 0.5 * (0.5 + 0.9)) > 0.03, "the two averages must differ"


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


def test_report_is_a_noop_on_an_empty_val_set():
    ev = _ev()
    ev._report({}, {}, dict.fromkeys(GF, 0.0), 0)
    assert "current_metric_value" not in ev.trainer.comm_info


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

        # Signature must track helix.model.mask.make_mask(B, mode, ratio,
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
