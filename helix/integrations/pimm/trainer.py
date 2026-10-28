"""``FMTrainer`` and the WSD schedules.

The trainer exists because the FM has no event separation — attention spans the
whole batch — so ``batch_size_per_gpu`` must be 1; and because muP needs
per-group LR ratios pimm's default optimizer build does not produce.
"""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR as _LambdaLR

from pimm.distributed import unwrap_model
from pimm.engines.train import TRAINERS, Trainer
from pimm.utils import comm
from pimm.utils.optimizer import OPTIMIZERS
from pimm.utils.scheduler import SCHEDULERS

from helix.model.mup import param_group_ratios


@TRAINERS.register_module()
class FMTrainer(Trainer):
    """pimm's Trainer with what the FM needs, and nothing else.

    Config: ``train = dict(type="FMTrainer")``.

    pimm's own machinery covers everything else — DDP, resume, logging, hooks,
    the loop itself, the training step. Three things cannot be expressed in
    config: the one-event-per-rank check, muP's param groups, and the
    ``"auto"`` cadences that follow from the run's step count.
    """

    def build_train_loader(self):
        """Refuse a per-GPU batch other than 1.

        pimm's ``batch_size`` is GLOBAL and ``batch_size_per_gpu`` is derived as
        ``batch_size // world_size``. The FM has NO event separation — attention
        runs over whatever tokens it is handed — so a per-GPU batch above 1
        concatenates unrelated events into one token set and silently trains a
        model whose tokens attend across event boundaries. Nothing downstream
        can see that; the loss simply means something else.

        It is easy to hit by accident, because the correct global value tracks
        the GPU count: ``batch_size = 4`` is right on 4 ranks and wrong on 1.
        See docs/archive/MULTI_EVENT_BATCHING.md.
        """
        per_gpu = self.cfg.batch_size // comm.get_world_size()
        if per_gpu != 1:
            raise ValueError(
                f"batch_size={self.cfg.batch_size} over world_size="
                f"{comm.get_world_size()} gives {per_gpu} events per GPU. The FM "
                f"requires exactly 1: it has no event separation, so a larger "
                f"per-GPU batch trains attention ACROSS unrelated events without "
                f"failing. Set batch_size to the number of ranks.")
        return super().build_train_loader()

    def build_optimizer(self):
        """Take param groups from the MODEL rather than from name matching.

        pimm's ``build_optimizer`` groups parameters by substring matches against
        ``cfg.param_dicts`` keywords. muP cannot be written that way: the hidden
        group's ``lr / m`` and its compensating ``weight_decay * m`` come from
        the model's width multiplier, not from anything in a parameter's name.
        ``FMModel.param_groups`` already computes them.
        """
        model = unwrap_model(self.model)          # may be DDP-wrapped by build_model
        if not hasattr(model, "param_groups"):
            raise TypeError(
                f"{type(model).__name__} has no param_groups(); FMTrainer exists "
                f"to use it. Use pimm's DefaultTrainer for models without muP.")
        if self.cfg.param_dicts:
            raise ValueError(
                "param_dicts is set, but FMTrainer takes its groups from "
                "model.param_groups(). Keyword matching cannot express muP, and "
                "having both would silently pick one — remove param_dicts.")

        cfg = dict(self.cfg.optimizer)
        base_lr = cfg.get("lr")
        groups = model.param_groups(base_lr, weight_decay=cfg.get("weight_decay"))
        cfg["params"] = groups                    # type/betas/etc still honoured
        opt = OPTIMIZERS.build(cfg)
        ratios = param_group_ratios(opt.param_groups, base_lr)
        self.logger.info(
            f"muP param groups: " + ", ".join(
                f"[{i}] {sum(p.numel() for p in g['params'])/1e6:.1f}M "
                f"lr x{r:.3f} wd={g.get('weight_decay')}"
                for i, (g, r) in enumerate(zip(opt.param_groups, ratios))))
        return opt

    def build_scheduler(self):
        """Resolve the ``"auto"`` cadences, then build as pimm does."""
        self._resolve_cadence()
        return super().build_scheduler()

    def _total_steps(self):
        """The run's optimizer steps: pimm's own ``iters_per_epoch * epoch``."""
        per_epoch = getattr(self, "_iters_per_epoch", None)
        n = per_epoch() if callable(per_epoch) else len(self.train_loader)
        return int(n) * int(self.cfg.epoch)

    def _resolve_cadence(self):
        """Fill every ``"auto"`` cadence from the run's actual step count (:mod:`helix.core.cadence`).

        A config states intent (epochs, model width), never a step count: that follows from
        the corpus, the split and the GPU count. ``"auto"`` is accepted for the scheduler's
        ``warmup`` and, in every hook, ``save_freq``, ``every_n_steps`` and ``evaluator_every_n_steps``; hooks
        are patched as config and, if pimm has built them already, as objects.
        """
        from helix.core import cadence
        steps = self._total_steps()
        m = self.cfg.model
        val = dict(save_freq=cadence.save_every(steps), every_n_steps=cadence.eval_every(steps),
                   evaluator_every_n_steps=cadence.eval_every(steps))
        sch = self.cfg.scheduler
        if sch.get("warmup") == "auto":
            sch.warmup = cadence.warmup_steps(steps, m.get("d", 128), m.get("d_base", 128))
        for h in list(self.cfg.hooks) + list(getattr(self, "hooks", None) or []):
            for k, v in val.items():
                if isinstance(h, dict) and h.get(k) == "auto":
                    h[k] = v
                elif not isinstance(h, dict) and getattr(h, k, None) == "auto":
                    setattr(h, k, v)
        self.logger.info(f"cadence over {steps} steps: warmup {sch.get('warmup')}, "
                         f"eval every {val['every_n_steps']}, save every {val['save_freq']}")


@SCHEDULERS.register_module()
class WSDStableLR(_LambdaLR):
    """Warmup-stable phase with an ABSOLUTE warmup step count.

    ``Trainer.build_scheduler`` (pimm engines/train.py:733) OVERWRITES
    ``cfg.scheduler.total_steps`` with ``iters_per_epoch * epoch`` before
    building, unconditionally — so any ``warmup_rate`` (pimm's
    ``MultiStepWithWarmupLR``, for one) is a RATIO of a number that moves with
    max_len, epoch and GPU count. 4,000 warmup steps written as a rate against a
    long run become ~6 steps on a 1,500-step one: a cold transformer at full LR
    by step 7, the classic loss-spike configuration.

    So warmup is specified in STEPS here and total_steps is ignored entirely
    (the stable phase is flat, so it needs no horizon — which is the point of
    WSD). The ramp is ``lr * s / warmup``.
    """

    def __init__(self, optimizer, warmup=4000, total_steps=None, last_epoch=-1):
        # total_steps is accepted and ignored: the trainer injects it whether or
        # not it is wanted, and silently dropping it beats failing on a kwarg
        # the caller never set.
        def stable(s):
            return min(1.0, s / warmup) if warmup > 0 else 1.0

        super().__init__(optimizer=optimizer, lr_lambda=stable,
                         last_epoch=last_epoch)


@SCHEDULERS.register_module()
class WSDCooldownLR(_LambdaLR):
    """The cooldown half of warmup-stable-decay: ``lr * max(floor, 1 - sqrt(p))``.

    The stable run commits to no total step count, so it can be extended; the
    decay is a SEPARATE short run started from a stable-phase checkpoint. A
    stable-phase checkpoint is therefore not an annealed model and should not be
    read as one.

    pimm has no equivalent. ``PolyLR`` is ``(1-p)**power``, which is a different
    curve: at p=0.25 this gives 0.500 and PolyLR(power=0.5) gives 0.866.
    ``1 - sqrt(p)`` drops fast and early, which is what a short cooldown wants.
    Warmup, if any, is in absolute steps for the reason given on
    :class:`WSDStableLR`.

    Being a LambdaLR, this scales each param group's own ``base_lr``, so muP's
    per-group ratios survive.

    ``stable_frac`` holds the LR flat for that fraction of the post-warmup steps
    before the decay starts, so ONE run can be warmup-stable-decay end to end
    (e.g. a short branch that must be compared cooled). 0 is the pure cooldown.
    """

    def __init__(self, optimizer, total_steps, warmup=0, floor=1e-3,
                 stable_frac=0.0, last_epoch=-1):
        flat = warmup + stable_frac * (total_steps - warmup)

        def wsd(s):
            if warmup and s < warmup:
                return s / warmup
            p = max(0.0, s - flat) / max(1, total_steps - flat)
            return max(floor, 1.0 - math.sqrt(min(p, 1.0)))

        super().__init__(optimizer=optimizer, lr_lambda=wsd, last_epoch=last_epoch)
