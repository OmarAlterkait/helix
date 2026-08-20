"""Validation for the coefficient FM.

Kept in helix rather than pimm, against ``RESEARCH_EXTRACTION_MAP.md``'s "EVAL
is pimm's job": that line was written when eval meant downstream probes and
baselines, which do belong there. This is the TRAINING metric, and it reaches
into helix internals no framework should know about — ``core.raw_heads``, the
``bin_edges``/``bin_cent_*`` buffers, ``losses_cat``. Putting it in pimm would
invert the dependency the whole integration is built to keep pointing one way.
"""

from __future__ import annotations

import contextlib

import torch

from pimm.distributed import unwrap_model
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase
from pimm.utils import comm


def _acc_grid_free(core, B, mask, logits, gf):
    """Accumulate var_expl / charge-closure sums for one eval batch.

    Both are computed on the SAME support the value loss uses
    (masked & valid & truly-occupied), and both are pooled over tokens by the
    caller — per-batch means would reweight events by density.

    Two read-outs, deliberately:
      * ``cent_asinh`` -> the posterior mean in TOKEN space, for var_expl. That
        is the space the target lives in, so the comparison needs no sinh.
      * ``cent_ratio`` -> the posterior mean of ``raw/sigma`` = E[sinh t], for
        charge. Applying sinh to the asinh-space mean instead is Jensen-biased
        ~31% low; see ``tokenize.decode_categorical``.
    """
    # No presence/finiteness test: `set_bins` DERIVES any centroid table it is
    # not given, so both are populated whenever `bin_edges` is. The guard that
    # used to stand here returned silently, and when a caller passed centroids
    # positionally into the wrong slot every charge metric vanished from the log
    # with nothing to say it had. If a table is NaN now that is a real bug and
    # the metrics should come out NaN and say so.
    ca, cr = core.bin_cent_asinh, core.bin_cent_ratio
    tgt, occ_t, valid = B["tgt"], B["occ"].bool(), B["valid"].bool()
    sel = mask[:, None] & valid & occ_t
    if not bool(sel.any()):
        return
    band = B["band_id"].long()
    # `einsum` is on autocast's lower-precision list, so under the trainer's
    # autocast context these contractions run in bf16 (~3e-3 relative) even with
    # float32 inputs — measured on A100. The elementwise form is float32 but
    # materialises a (n_cells, n_slot, K) product, ~17 MB more at eval shapes;
    # disabling autocast keeps einsum's memory AND float32.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        p = torch.softmax(logits.float(), -1)                 # (n_cells, n_slot, K)
        rec_a = torch.einsum("csk,ck->cs", p, ca[band].float())
        rec_r = torch.einsum("csk,ck->cs", p, cr[band].float())
    y = tgt.float()[sel]
    d = rec_a[sel] - y
    gf["sse"] += float((d * d).sum())
    gf["sy"] += float(y.sum())
    gf["syy"] += float((y * y).sum())
    gf["nv"] += float(sel.sum())
    # UNSIGNED is the primary: the signed sum of a near-symmetric coefficient
    # distribution is a small difference of large numbers, so the signed ratio
    # swings wildly (a perfectly-binned predictor scored 0.008 on it in test).
    # Research reports both for exactly this reason; the signed one is kept as a
    # bias indicator, not as a closure measure.
    r_sel, t_sel = rec_r[sel], torch.sinh(y)
    gf["chg_pred"] += float(r_sel.abs().sum())
    gf["chg_true"] += float(t_sel.abs().sum())
    gf["chg_pred_s"] += float(r_sel.sum())
    gf["chg_true_s"] += float(t_sel.sum())


@HOOKS.register_module()
class CoeffFMEvaluator(HookBase):
    """Validation for the coefficient FM.

    Deliberately not pimm's ``MAEEvaluator``: that one passes ``return_pred=``
    (which ``FMModel.forward`` does not take) and reports ``coord_loss`` /
    ``feat_loss``, which are a point-cloud MAE's quantities. Ours are occupancy
    BCE and coefficient value — renaming them to match would make the numbers
    misleading rather than compatible.

    Masks are drawn from a SEEDED generator keyed on the batch index, so the
    same events are masked the same way at every evaluation. Without that the
    metric moves epoch to epoch because the mask moved, which reads as model
    variance. (Research's eval intended this and did not get it: its 'plane' mode
    ignored the generator and drew from the global RNG — see helix.model.mask.)

    Publishes ``-avg_loss`` as ``neg_val_loss`` so pimm's CheckpointSaver can
    select on it (higher is better, by its convention).
    """

    def __init__(self, every_n_steps=0, max_batches=None, mask_seed=7,
                 grid_free=True):
        self.every_n_steps = int(every_n_steps)
        self.max_batches = max_batches
        self.mask_seed = int(mask_seed)
        # var_expl + charge closure, computed from the SAME forward as the
        # loss (see _forward) — so this is free, not a second pass. Silently
        # inert unless the checkpoint carries the bin centroids.
        self.grid_free = bool(grid_free)

    def after_step(self):
        if self.every_n_steps <= 0:
            return
        step = (self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"] * self.trainer.comm_info["epoch"])
        if (step + 1) % self.every_n_steps == 0:
            self.eval()

    def after_epoch(self):
        if self.every_n_steps <= 0:
            self.eval()

    #: Every accumulator the evaluator sums. One tuple, because `_all_reduce`
    #: packs positionally and both sides must agree on the order.
    _RED = ("bce", "val", "masked_frac",                       # weighted means
            "n_bce", "n_val", "n_masked_frac",                 # their weights
            "sse", "sy", "syy", "nv",                          # var_expl
            "chg_pred", "chg_true", "chg_pred_s", "chg_true_s")  # charge

    def eval(self):
        """EVERY rank evaluates its shard; the sums are all-reduced.

        This used to be "rank 0 evaluates, the others wait at a barrier", which
        was correct about the barrier and wrong about the data: pimm builds the
        val loader with a ``DistributedSampler`` whenever world_size > 1
        (engines/train.py:687), so rank 0's loader yields only rank 0's SHARD.
        Every multi-GPU eval we have logged therefore scored ~1/world_size of
        the validation set and reported it as the validation number — that is
        the `batches=145` in the 4-GPU logs against a 577-event val set.

        Having all ranks work costs nothing: the other ranks were blocked in
        ``synchronize()`` for the whole of rank 0's pass anyway, so wall time is
        the same shard-sized pass it always was, now covering the whole set.

        The all-reduce is itself a collective, so the symmetry the old barrier
        needed is now structural rather than something the try/finally has to
        maintain. That mattered: getting it wrong did not fail, it HUNG — the
        run trained, evaluated, logged a checkpoint, and then deadlocked with
        rank 0 inside the collective save and the others still in the barrier.

        Caveat, logged rather than papered over: ``DistributedSampler`` pads the
        last shard by repeating samples, so a val set that does not divide by
        world_size double-counts a few events (3 of 580 at 577/4, ~0.5%).
        """
        world = comm.get_world_size()
        try:
            acc = self._eval_shard()
        except Exception:
            # A rank that dies before the reduce hangs every other rank inside
            # it. Contribute zeros and let the run fail on the logged error
            # rather than on a wedged collective.
            self.trainer.logger.exception("CoeffFMEvaluator: shard eval failed")
            acc = None
        if acc is None:
            acc = ({}, {}, {k: 0.0 for k in self._RED[6:]}, 0)
        totals, counts, gf, n = acc
        if world > 1:
            totals, counts, gf, n = self._all_reduce(totals, counts, gf, n)
        self._report(totals, counts, gf, n)

    def _all_reduce(self, totals, counts, gf, n):
        """Sum every accumulator across ranks in ONE collective.

        One flat tensor, not a call per key: the key set is fixed by ``_RED``, so
        packing positionally removes any chance of ranks disagreeing on iteration
        order and reducing mismatched quantities into each other — which would
        not raise, just produce a plausible wrong number.
        """
        import torch.distributed as dist

        flat = dict(totals)
        flat.update({f"n_{k}": v for k, v in counts.items()})
        flat.update(gf)
        t = torch.tensor([flat.get(k, 0.0) for k in self._RED] + [float(n)],
                         dtype=torch.float64,
                         device=self.trainer.parallel_context.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        v = dict(zip(self._RED, t.tolist()))
        return ({k: v[k] for k in ("bce", "val", "masked_frac")},
                {k: v[f"n_{k}"] for k in ("bce", "val", "masked_frac")},
                {k: v[k] for k in self._RED[6:]},
                int(round(t[-1].item())))

    def _forward(self, model, core, B, mask, gf):
        """One forward per batch: the loss AND the grid-free metrics from the
        same heads.

        The obvious version calls ``model(B, tok_mask=mask)`` for the loss and
        ``core.raw_heads(B, mask)`` again for the metrics — a SECOND full
        encoder+decoder pass. Evaluation is not cheap here: measured on
        `coeff-fm-train`, eval is 43.8 s x 100 evals = 73 min of a 469 min run,
        i.e. **15.5% of wall time**. Doubling it would cost ~15% of every future
        run to compute two scalars.

        So for the categorical head — the only one we train — reproduce
        ``FMModel.forward``'s branch here from a single ``raw_heads`` call. Any
        other head configuration falls back to ``forward``, which keeps this
        correct for heads it does not know about rather than silently scoring
        them with the wrong loss.

        Calling ``core`` rather than ``model`` skips the DDP wrapper; under
        ``no_grad`` in eval there is no gradient to synchronise, and the previous
        code already reached for ``core.raw_heads`` for the same reason.
        """
        if self.grid_free and getattr(core, "n_bins", 0) > 0:
            from helix.model.loss import losses_cat
            assert torch.isfinite(core.bin_edges).all(), \
                ("n_bins > 0 requires set_bins(edges) before evaluation — the "
                 "edges buffer is still unset (NaN).")
            # The same check `forward` runs. Skipping `forward` to save a second
            # pass must not also skip its contract.
            core.require_batch_keys(B)
            occ, val, _ = core.raw_heads(B, mask)
            bce, vloss = losses_cat(occ, val, B, mask, core.bin_edges,
                                    vis_w=core.vis_w)
            _acc_grid_free(core, B, mask, val, gf)
            return {"loss": bce + vloss, "bce": bce.detach(),
                    "val": vloss.detach(),
                    "masked_frac": mask.float().mean().detach()}
        return model(B, tok_mask=mask)

    def _eval_shard(self):
        """Score THIS rank's shard. -> (totals, counts, gf, n_batches).

        Sums only; no division. Averaging is `_report`'s job, after the reduce,
        because a mean cannot be summed across ranks.
        """
        loader = getattr(self.trainer, "val_loader", None)
        if loader is None:
            if comm.get_rank() == 0:
                self.trainer.logger.info("CoeffFMEvaluator: no val_loader; skipping")
            return None

        from pimm.distributed import move_batch_to_device
        if comm.get_rank() == 0:
            self.trainer.logger.info(">>>>>>>> Coeff FM validation >>>>>>>>")
        model = self.trainer.model
        was_training = model.training
        model.eval()
        core = unwrap_model(model)
        device = self.trainer.parallel_context.device

        totals, counts, n = {}, {}, 0
        gf = {k: 0.0 for k in ("sse", "sy", "syy", "nv", "chg_pred",
                               "chg_true", "chg_pred_s", "chg_true_s")}
        with torch.no_grad():
            for i, input_dict in enumerate(loader):
                if self.max_batches is not None and i >= self.max_batches:
                    break
                B = move_batch_to_device(input_dict, device)
                B.setdefault("n_cells", B["plane_id"].shape[0])
                # Keyed on RANK as well as batch index: without the rank term
                # every rank would mask its own (different) events with the
                # identical pattern. Fixed for a given world_size, so the metric
                # is reproducible eval-to-eval — but a run at a different
                # world_size shards differently and is not mask-comparable,
                # which was already true and is now at least deliberate.
                gen = torch.Generator(device=device).manual_seed(
                    self.mask_seed + 100003 * comm.get_rank() + i)
                # mode="random" EXPLICITLY, matching research: mae_ddp.py:216
                # passes args.mask_mode to perband_mse_cat, so the eval metric is
                # a pure random-mask number even when plane_frac > 0 (research
                # reports the plane-masked number as a SEPARATE curve, and only
                # for non-categorical heads — m113 is categorical, so its eval
                # was pure random). Without this, plane_frac makes val loss a
                # 90/10 mixture and it stops being comparable across runs.
                mask = core.make_mask(B, mode="random", gen=gen)   # SAME every eval
                ctx = (torch.autocast(
                           device_type=device.type,
                           dtype=(torch.bfloat16
                                  if self.trainer.cfg.amp_dtype == "bfloat16"
                                  else torch.float16))
                       if getattr(self.trainer.cfg, "enable_amp", False)
                       else contextlib.nullcontext())
                with ctx:
                    out = self._forward(model, core, B, mask, gf)
                # POOL over tokens, do not average per-event means. Each loss
                # term already divided by ITS OWN support inside the loss, so
                # summing those means weights a 5k-token event equally with a
                # 40k-token one. Dense events are both heavier and harder —
                # corr(per-event CE, token count) = 0.79 — so the unweighted
                # mean is biased optimistic by ~0.09 nats, which is larger than
                # the k30-vs-R1 difference these numbers were used to compare.
                # Research pools (fm/train.py:143, fm/cross_nll.py:41).
                #
                # The supports are recomputed here rather than returned from the
                # loss, so `losses*` stay byte-comparable with research.
                mrow = mask[:, None]
                w_occ = float((mrow & B["valid"].bool()).sum())
                w_val = float((B["occ"].bool() & B["valid"].bool() & mrow).sum())
                wt = {"bce": w_occ, "val": w_val, "masked_frac": float(mask.numel())}
                for k, v in out.items():
                    if torch.is_tensor(v) and v.ndim == 0 and k != "loss":
                        w = wt.get(k, 1.0)
                        totals[k] = totals.get(k, 0.0) + float(v) * w
                        counts[k] = counts.get(k, 0.0) + w
                n += 1

        if was_training:
            model.train()
        return totals, counts, gf, n

    def _report(self, totals, counts, gf, n):
        """Average the reduced sums, log, and publish the selection metric.

        Runs on EVERY rank, and only rank 0 prints. `current_metric_value` has
        to be set everywhere: the checkpoint save is collective, so if the ranks
        disagreed about which step was best they would disagree about whether to
        save — and the values agree here precisely because they come from the
        same reduced sums.
        """
        rank0 = comm.get_rank() == 0
        if not n:
            if rank0:
                self.trainer.logger.info("CoeffFMEvaluator: val_loader was empty")
            return

        # `loss` is rebuilt from the pooled parts, not pooled itself: it is
        # bce + val, and the two have different denominators.
        avg = {k: v / max(counts.get(k, n), 1e-9) for k, v in totals.items()}
        avg["loss"] = avg.get("bce", 0.0) + avg.get("val", 0.0)
        if gf["nv"] > 0:
            # var_expl is GRID-FREE: it compares the posterior-mean asinh
            # reconstruction against the target's own measured variance, so it
            # survives a change of bin table or corpus in a way cross-entropy
            # does not. charge_closure is Sum(pred)/Sum(true) in units of sigma,
            # nominal 1.0. Research computes both every eval
            # (fm/train.py:perband_mse_cat, mae_ddp.py:216-220); their absence
            # here is why a 31%-low charge read-back and a cross-table CE
            # comparison both went unnoticed.
            mean_y = gf["sy"] / gf["nv"]
            var_y = max(gf["syy"] / gf["nv"] - mean_y * mean_y, 1e-12)
            avg["var_expl"] = 1.0 - (gf["sse"] / gf["nv"]) / var_y
            if gf["chg_true"] > 1e-9:
                avg["charge_closure"] = gf["chg_pred"] / gf["chg_true"]
            if abs(gf["chg_true_s"]) > 1e-6 * max(gf["chg_true"], 1e-9):
                avg["charge_bias"] = gf["chg_pred_s"] / gf["chg_true_s"]
        if rank0:
            self.trainer.logger.info(
                f"   [coeff-eval] batches={n} " +
                " ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items())))
        writer = getattr(self.trainer, "writer", None)
        if rank0 and writer is not None:
            # GLOBAL STEP, not epoch. EVAL_EVERY (1178) does not divide
            # iters_per_epoch (4758), so ~4 evals land inside each epoch and
            # three of every four were overwritten — 100 evals collapsed to 25
            # TB points. train.log kept all 100, which is why the curve script
            # parses the log instead. Research keys fm_curve.jsonl on step.
            step = int(getattr(self.trainer, "global_step", 0) or
                       self.trainer.comm_info.get("epoch", 0))
            for k, v in avg.items():
                writer.add_scalar(f"val/{k}", v, step)
        self.trainer.comm_info["current_metric_value"] = -avg["loss"]   # higher is better
        self.trainer.comm_info["current_metric_name"] = "neg_val_loss"
