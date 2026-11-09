"""The training schedule, derived rather than written down: warmup and checkpoint cadence
from the run's step count, and the muP learning-rate ratios a scheduler must keep.

Pure (no torch, no pimm), so it is testable without a trainer; ``FMTrainer``
(:mod:`helix.integrations.pimm.trainer`) applies both.

CADENCE
=======
Applied at run time: a config writes ``"auto"`` and FMTrainer fills it from the run's own step
count (``iters_per_epoch * epoch``), so no config carries a step count or a training-set size.

WHY THIS IS A MODULE AND NOT THREE LINES IN A CONFIG
----------------------------------------------------
It was three lines in a config. Then it was three lines in TWO configs, because
the eight-run config has a different ``N_TRAIN_EVENTS`` and therefore recomputes
``STEPS``, and the cadences were copied along with it. Fixing one left the other
shadowing it, which is how a fix gets applied and has no effect.

WHAT WAS WRONG WITH THE OLD RULE
--------------------------------
All three cadences were a FRACTION of ``STEPS`` with a small floor::

    WARMUP     = max(100, 0.0040 * STEPS)
    EVAL_EVERY = max( 50, 0.0099 * STEPS)
    SAVE_EVERY = max( 50, 0.0020 * STEPS)

and ``STEPS = n_train_events * epochs // batch_size`` SHRINKS as the batch grows.
So every cadence tightened with batch size until the floors took over, inverting
what each one is for. Over the eight-run corpus at 25 epochs:

===== ========= ============ ==============
  B       STEPS       warmup      save_freq
===== ========= ============ ==============
    4   938,993        3,756          1,878
   16   234,748          939            469
  128    29,343   100 (floor)            59
  512     7,335   100 (floor)     50 (floor)
===== ========= ============ ==============

At B=512 that is 100 warmup steps at an 8x learning rate on a cold 12-block
transformer, and a checkpoint every 50 steps -- about every 13 seconds, with all
ranks barriered while rank 0 serially removes the previous one. A fraction of a
batch-dependent quantity is not a cadence.

Each quantity below is derived from the thing that actually determines it, and
none of them depends on batch size except through the step budget they are
capped by.

muP
===
``FMModel.param_groups`` emits AdamW groups whose per-group ``lr`` encodes muP:
hidden weights get ``base_lr / m`` for width multiplier ``m = d / d_base``. That
encoding is fragile in exactly one place — the scheduler. Anything that assigns a
single LR to every group silently discards muP, and the run looks fine: the loss
still falls, just to a worse place, and only at another width would you notice.
The WSD schedules are ``LambdaLR``s, which scale each group's own ``lr`` and so
keep the ratios.
"""
from __future__ import annotations

#: A run must checkpoint at least this often, however short it is. The cost-derived floor
#: below is computed from a long run's economics and would exceed a short arm outright, which
#: would then checkpoint only at its end.
MIN_SAVES = 2

#: The wall-clock costs the cadences are derived from, measured once and owned
#: here. Perlmutter A100-40GB, helix 3aa96b2 (one implementation, compiled
#: blocks), d512, B=16 over four nodes: the steady step and one save (3 x 226 MiB:
#: 4 s from four nodes, 11 s from one -- the slower is used).
#: Re-measure when the model or the card changes; nothing else should restate
#: them.
STEP_SECONDS = 0.14
CKPT_SECONDS = 11.0


def warmup_steps(steps, d, d_base, base_steps=500):
    """Linear-warmup length, in optimizer steps.

    An OPTIMISATION timescale, not a fraction of the run: warmup exists to let
    Adam's second moments settle and to survive the early steps where the
    critical batch size is near zero, and both are step-count phenomena. It must
    therefore NOT shrink when the batch grows -- which is precisely what the old
    rule did.

    Scaled with model WIDTH, following Porian et al.'s correction (warmup
    proportional to model size), who measured that getting this wrong costs
    ~0.096 of the compute-optimal exponent -- three times the cost of a
    mismatched decay schedule, and the largest schedule-shaped error in the
    scaling literature.

    Capped at a fifth of the run so a short arm is never mostly warmup: Porian's
    own acceptance test is that every run trains at least 5x its warmup.
    """
    w = round(base_steps * (float(d) / float(d_base)) ** 0.5)
    return min(max(base_steps, w), max(1, steps // 5))


def save_every(steps, ckpt_seconds=CKPT_SECONDS, step_seconds=STEP_SECONDS, overhead=0.02,
               target_count=20):
    """Checkpoint cadence, in optimizer steps.

    A WALL-CLOCK risk decision, not an optimisation one: how much work may a
    preemption destroy, against how much time checkpointing may consume.

    Holding checkpoint overhead under `overhead` of wall time needs

        save_freq >= ckpt_seconds / (step_seconds * overhead)

    which at the measured defaults (CKPT_SECONDS, STEP_SECONDS) is ~3,900 steps. Independent of batch size, which is
    the entire point -- the old floor of 50 steps meant one save every 13 s at
    B=512, and the job would have spent more time deleting checkpoints than
    training.

    `target_count` keeps a long run checkpointing often enough that a preempted
    window loses a bounded slice.
    """
    floor = round(ckpt_seconds / (step_seconds * overhead))
    # ...but never so rare that a short run checkpoints once or not at all. The
    # overhead floor is derived from a LONG run's economics; on an arm of a few
    # hundred steps it would exceed the run, which is how a cadence designed to
    # bound cost ends up producing nothing.
    ceiling = max(1, steps // MIN_SAVES)
    return min(max(floor, max(1, steps // target_count)), ceiling)


# ---- muP -------------------------------------------------------------------------------------

def param_group_ratios(param_groups, base_lr):
    """Per-group ``lr / base_lr`` — the muP ratios to preserve under a schedule.

    Capture these right after building the optimizer, before any scheduler has
    touched it.
    """
    if base_lr is None or base_lr == 0:
        raise ValueError(f"base_lr must be a non-zero number, got {base_lr!r}")
    out = []
    for i, g in enumerate(param_groups):
        lr = g.get("lr")
        if lr is None:
            raise ValueError(f"param group {i} has no 'lr'; muP ratios are undefined")
        out.append(lr / base_lr)
    return out

