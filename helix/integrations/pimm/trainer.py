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

from helix.model.mup import expand_max_lr, param_group_ratios


@TRAINERS.register_module()
class FMTrainer(Trainer):
    """pimm's Trainer with the two things muP needs, and nothing else.

    Config: ``train = dict(type="FMTrainer")``.

    pimm's own machinery covers everything else — DDP, resume, logging, hooks,
    the loop itself. Only the optimizer and the scheduler cannot be expressed in
    config, for one reason each.
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
        See MULTI_EVENT_BATCHING.md.
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
        ``FMModel.param_groups`` already computes them, and is verified
        bit-identical to the research implementation.
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
        self._mup_ratios = param_group_ratios(opt.param_groups, base_lr)
        self.logger.info(
            f"muP param groups: " + ", ".join(
                f"[{i}] {sum(p.numel() for p in g['params'])/1e6:.1f}M "
                f"lr x{r:.3f} wd={g.get('weight_decay')}"
                for i, (g, r) in enumerate(zip(opt.param_groups, self._mup_ratios))))
        return opt

    def build_scheduler(self):
        """Give the scheduler a PER-GROUP peak LR so muP survives it.

        ``OneCycleLR`` takes ``max_lr`` as a scalar or a per-group list. A scalar
        assigns every group the same peak, which discards muP silently — the
        hidden group would climb to ``base_lr`` instead of ``base_lr / m``.
        """
        ratios = getattr(self, "_mup_ratios", None)
        if ratios and "max_lr" in self.cfg.scheduler:
            self.cfg.scheduler.max_lr = expand_max_lr(
                self.cfg.scheduler.max_lr, ratios)
        return super().build_scheduler()


@SCHEDULERS.register_module()
class WSDStableLR(_LambdaLR):
    """Warmup-stable phase with an ABSOLUTE warmup step count.

    Exists because ``warmup_rate`` cannot express what m113 did.
    ``Trainer.build_scheduler`` (pimm engines/train.py:733) OVERWRITES
    ``cfg.scheduler.total_steps`` with ``iters_per_epoch * epoch`` before
    building, unconditionally — so a config that sets ``total_steps`` has it
    discarded, and any ``warmup_rate`` is reinterpreted against the
    trainer-derived total.

    That is not a hypothetical. Expressing m113's 4,000 warmup steps as
    ``warmup_rate = 4000/1_010_000`` against a 1,500-step run yields **5.94
    steps** of warmup: the LR reaches 1.1e-3 by step 7 instead of step 4000 —
    571x research's value at that point, 5.3x the integrated LR over the run —
    on a cold 12-block d=512 transformer. That is the classic loss-spike
    configuration, and it silently rescales again with any change to max_len,
    epoch or GPU count.

    So warmup is specified in STEPS here and total_steps is ignored entirely
    (the stable phase is flat, so it needs no horizon — which is the point of
    WSD: "flat, no horizon baked in", mae_ddp.py:164).

    Matches research's ``lr * s / warmup`` ramp exactly.
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

    m113 trained the STABLE phase — `lr_mode: const`, described in mae_ddp.py as
    "WSD stable phase: flat, no horizon baked in". That is the point of WSD: the
    stable run commits to no total step count, so it can be extended, and the
    decay is a SEPARATE short run started from a stable-phase checkpoint. A
    checkpoint like m113's at 1,010,000 steps is therefore not an annealed model
    and should not be read as one.

    pimm has no equivalent. ``PolyLR`` is ``(1-p)**power``, which is a different
    curve: at p=0.25 research gives 0.500 and PolyLR(power=0.5) gives 0.866.
    ``1 - sqrt(p)`` drops fast and early, which is what a short cooldown wants.

    The warmup-then-constant STABLE phase needs no new code —
    ``MultiStepWithWarmupLR(milestones=[])`` leaves the decay factor at 1.0
    forever, which is exactly it.

    Being a LambdaLR, this scales each param group's own ``base_lr``, so muP's
    per-group ratios survive without the ``max_lr`` expansion OneCycleLR needs.
    """

    def __init__(self, optimizer, total_steps, warmup=0, floor=1e-3,
                 last_epoch=-1):
        def wsd(s):
            if warmup and s < warmup:
                return s / warmup                      # research: lr * s / warmup
            p = (s - warmup) / max(1, total_steps - warmup)
            return max(floor, 1.0 - math.sqrt(min(p, 1.0)))

        super().__init__(optimizer=optimizer, lr_lambda=wsd, last_epoch=last_epoch)
