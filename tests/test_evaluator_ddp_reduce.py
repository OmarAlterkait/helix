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
    from helix.integrations.pimm import CoeffFMEvaluator
    ev = CoeffFMEvaluator.__new__(CoeffFMEvaluator)
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
    assert "charge_bias=0.9000" in line["m"]


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
        ev = CoeffFMEvaluator.__new__(CoeffFMEvaluator)
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
