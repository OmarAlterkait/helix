"""Validation inside a training run: the coefficient FM's and the denoiser's.

``CoeffFMEvaluator``'s metric is :mod:`helix.eval.recon` (pimm-free, also the ``recon`` metric
of ``python -m helix.eval``); ``DenoiseEvaluator``'s is the denoiser's own loss. Each runs on
pimm's val loader at the run's cadence, all-reduces the sums across ranks, logs, and publishes
the checkpoint-selection metric. The denoiser's full scoring is ``helix.eval``'s ``denoise``.
"""

from __future__ import annotations

import torch

from pimm.distributed import unwrap_model
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase
from pimm.utils import comm

from helix.eval import recon as R


@HOOKS.register_module()
class CoeffFMEvaluator(HookBase):
    """Validation for the coefficient FM.

    Deliberately not pimm's ``MAEEvaluator``: that one passes ``return_pred=`` (which
    ``FMModel.forward`` does not take) and reports a point-cloud MAE's quantities.

    Masks are drawn from a SEEDED generator keyed on the rank and batch index
    (:func:`helix.eval.recon.mask_for`), so the same events are masked the same way at every
    evaluation. ``mask_mode`` defaults to "random", and stays there for any number meant to be
    compared across runs; "plane" scores the plane-masked task as a SEPARATE number. A fixed
    ``mask_ratio`` scores runs trained at different ratios on the same task.

    Publishes ``-avg_loss`` as ``neg_val_loss`` so pimm's CheckpointSaver can select on it
    (higher is better, by its convention).
    """

    def __init__(self, every_n_steps=0, max_batches=None, mask_seed=R.MASK_SEED,
                 mask_mode="random", n_planes=1, mask_ratio=None):
        # "auto": FMTrainer fills it from the run's step count before training starts
        self.every_n_steps = every_n_steps if every_n_steps == "auto" else int(every_n_steps)
        self.max_batches = max_batches
        self.mask_seed = int(mask_seed)
        self.mask_mode = str(mask_mode)
        self.n_planes = int(n_planes)
        self.mask_ratio = None if mask_ratio is None else float(mask_ratio)

    def before_train(self):
        if self.every_n_steps == "auto":
            raise RuntimeError("CoeffFMEvaluator: every_n_steps='auto' was never resolved; "
                               "it needs train=dict(type='FMTrainer')")

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

    #: Every accumulator the evaluator sums. One tuple, because `_all_reduce` packs
    #: positionally and both sides must agree on the order.
    _RED = R.POOLED + tuple(f"n_{k}" for k in R.POOLED) + R.GF_KEYS

    def eval(self):
        """EVERY rank evaluates its shard; the sums are all-reduced.

        pimm builds the val loader with a ``DistributedSampler`` whenever world_size > 1, so
        rank 0's loader yields only rank 0's SHARD. The all-reduce is itself a collective, so
        every rank reaching it is structural: an asymmetry here does not fail, it HANGS.

        Caveat: ``DistributedSampler`` pads the last shard by repeating samples, so a val set
        that does not divide by world_size double-counts a few events (~0.5%).
        """
        world = comm.get_world_size()
        failure = None
        try:
            acc = self._eval_shard()
        except Exception as exc:
            # This rank still contributes zeros and enters the collective (otherwise every
            # other rank hangs in it), then FAILS: zeros are indistinguishable from an empty
            # val set.
            self.trainer.logger.exception("CoeffFMEvaluator: shard eval failed")
            failure = exc
            acc = None
        if acc is None:
            acc = ({}, {}, dict.fromkeys(R.GF_KEYS, 0.0), 0)
        totals, counts, gf, n = acc
        if world > 1:
            totals, counts, gf, n = self._all_reduce(totals, counts, gf, n)
        if failure is not None:
            raise failure          # after the collective, so no rank is left in it
        return self._report(totals, counts, gf, n)

    def _all_reduce(self, totals, counts, gf, n):
        """Sum every accumulator across ranks in ONE collective, packed positionally by
        ``_RED`` so ranks cannot reduce mismatched quantities into each other."""
        import torch.distributed as dist

        flat = dict(totals)
        flat.update({f"n_{k}": v for k, v in counts.items()})
        flat.update(gf)
        t = torch.tensor([flat.get(k, 0.0) for k in self._RED] + [float(n)],
                         dtype=torch.float64, device=self.trainer.parallel_context.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        v = dict(zip(self._RED, t.tolist()))
        return ({k: v[k] for k in R.POOLED}, {k: v[f"n_{k}"] for k in R.POOLED},
                {k: v[k] for k in R.GF_KEYS}, int(round(t[-1].item())))

    def _forward(self, model, core, B, mask, gf):
        """One forward per batch: the loss AND the grid-free sums (:func:`helix.eval.recon.
        heads_loss`). ``core`` rather than ``model`` skips the DDP wrapper; under ``no_grad``
        there is no gradient to synchronise."""
        return R.heads_loss(core, B, mask, gf)

    def _eval_shard(self):
        """Score THIS rank's shard. -> (totals, counts, gf, n_batches). Sums only."""
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
        try:
            return self._eval_shard_inner(model, loader, move_batch_to_device)
        finally:
            # MUST be a finally: `eval()` catches what this raises so every rank still reaches
            # the all-reduce, and training must not continue in eval mode.
            if was_training:
                model.train()

    def _eval_shard_inner(self, model, loader, move_batch_to_device):
        core = unwrap_model(model)
        device = self.trainer.parallel_context.device
        dtype = torch.bfloat16 if self.trainer.cfg.amp_dtype == "bfloat16" else torch.float16
        amp = bool(getattr(self.trainer.cfg, "enable_amp", False))
        totals, counts, n = {}, {}, 0
        gf = dict.fromkeys(R.GF_KEYS, 0.0)
        with torch.no_grad():
            for i, input_dict in enumerate(loader):
                if self.max_batches is not None and i >= self.max_batches:
                    break
                B = move_batch_to_device(input_dict, device)
                B.setdefault("n_cells", B["plane_id"].shape[0])
                mask = R.mask_for(core, B, i, rank=comm.get_rank(), seed=self.mask_seed,
                                  mode=self.mask_mode, ratio=self.mask_ratio, n_planes=self.n_planes)
                with R.autocast(device, dtype, enabled=amp):
                    out = self._forward(model, core, B, mask, gf)
                R.pool(out, B, mask, totals, counts)
                n += 1
        return totals, counts, gf, n

    def _report(self, totals, counts, gf, n):
        """Average the reduced sums, log, publish the selection metric, RETURN it.

        Runs on EVERY rank, and only rank 0 prints. ``current_metric_value`` is set on every
        rank because the values agree -- they come from the same reduced sums.
        """
        rank0 = comm.get_rank() == 0
        if not n:
            if rank0:
                self.trainer.logger.info("CoeffFMEvaluator: val_loader was empty")
            return None
        avg = R.average(totals, counts, gf)
        if rank0:
            self.trainer.logger.info(
                f"   [coeff-eval] batches={n} mask={self.mask_mode}"
                + (f"/{self.n_planes}" if self.mask_mode != "random" else "") + " " +
                " ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items())))
        writer = getattr(self.trainer, "writer", None)
        if rank0 and writer is not None:
            # GLOBAL STEP, not epoch: several evals land inside each epoch
            step = int(getattr(self.trainer, "global_step", 0) or
                       self.trainer.comm_info.get("epoch", 0))
            for k, v in avg.items():
                writer.add_scalar(f"val/{k}", v, step)
        self.trainer.comm_info["current_metric_value"] = -avg["loss"]   # higher is better
        self.trainer.comm_info["current_metric_name"] = "neg_val_loss"
        return avg


@HOOKS.register_module()
class DenoiseEvaluator(HookBase):
    """Validation for the denoiser: its training loss (``mse``, ``bce``, ``loss``) on the first
    ``max_batches`` events of each rank's share of the val split, averaged over events.

    The val split's ``DenoiseRows`` has a fixed ``seed``, so every evaluation scores the same
    cells. Publishes ``-loss`` as ``neg_val_loss``: with ``CheckpointSaver(evaluator_every_n_steps=
    ...)`` at the same cadence, ``model/model_best.pth`` is the lowest validation loss.
    """

    KEYS = ("loss", "mse", "bce")

    def __init__(self, every_n_steps="auto", max_batches=16):
        self.every_n_steps = every_n_steps if every_n_steps == "auto" else int(every_n_steps)
        self.max_batches = None if max_batches is None else int(max_batches)

    def before_train(self):
        if self.every_n_steps == "auto":
            raise RuntimeError("DenoiseEvaluator: every_n_steps='auto' was never resolved; "
                               "it needs train=dict(type='FMTrainer')")

    def after_step(self):
        step = (self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"] * self.trainer.comm_info["epoch"])
        if self.every_n_steps > 0 and (step + 1) % self.every_n_steps == 0:
            self.eval()

    def after_epoch(self):
        if self.every_n_steps <= 0:
            self.eval()

    def eval(self):
        """Every rank scores its shard and enters the all-reduce, even on failure (an asymmetry
        here hangs every other rank); a failure is raised after it."""
        from pimm.distributed import move_batch_to_device
        loader = self.trainer.val_loader
        model = self.trainer.model
        dev = self.trainer.parallel_context.device
        dtype = torch.bfloat16 if self.trainer.cfg.amp_dtype == "bfloat16" else torch.float16
        amp = bool(getattr(self.trainer.cfg, "enable_amp", False))
        acc = torch.zeros(len(self.KEYS) + 1, dtype=torch.float64, device=dev)
        failure, was_training = None, model.training
        model.eval()
        try:
            core = unwrap_model(model)
            with torch.no_grad():
                for i, batch in enumerate(loader or []):
                    if self.max_batches is not None and i >= self.max_batches:
                        break
                    with R.autocast(dev, dtype, enabled=amp):
                        out = core(move_batch_to_device(batch, dev))
                    acc += torch.stack([out[k].detach().double() for k in self.KEYS]
                                       + [torch.ones((), dtype=torch.float64, device=dev)])
        except Exception as exc:
            self.trainer.logger.exception("DenoiseEvaluator: shard eval failed")
            failure, acc = exc, torch.zeros_like(acc)
        finally:
            if was_training:
                model.train()
        if comm.get_world_size() > 1:
            import torch.distributed as dist
            dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        if failure is not None:
            raise failure
        n = int(round(acc[-1].item()))
        if not n:
            if comm.get_rank() == 0:
                self.trainer.logger.info("DenoiseEvaluator: val_loader was empty")
            return None
        avg = {k: v / n for k, v in zip(self.KEYS, acc[:-1].tolist())}
        if comm.get_rank() == 0:
            self.trainer.logger.info(f"   [denoise-eval] events={n} " +
                                     " ".join(f"{k}={v:.5f}" for k, v in avg.items()))
            writer = getattr(self.trainer, "writer", None)
            if writer is not None:
                step = int(getattr(self.trainer, "global_step", 0) or 0)
                for k, v in avg.items():
                    writer.add_scalar(f"val/{k}", v, step)
        self.trainer.comm_info["current_metric_value"] = -avg["loss"]   # higher is better
        self.trainer.comm_info["current_metric_name"] = "neg_val_loss"
        return avg
