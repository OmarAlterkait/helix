"""How often to warm up, evaluate and checkpoint — derived, not written down.

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
ranks barriered while rank 0 serially removes the previous one. The fractions
were honest (they are m113's values expressed as fractions of its 1,010,000-step
run) but a fraction of a batch-dependent quantity is not a cadence.

Each quantity below is derived from the thing that actually determines it, and
none of them depends on batch size except through the step budget they are
capped by.
"""
from __future__ import annotations

#: A run must produce at least this many of each, however short it is. The
#: cost-derived floors below are computed from a long run's economics and will
#: exceed a short arm outright; without these a 880-step run at B=512 evaluates
#: zero times and checkpoints zero times, and reports nothing.
MIN_EVALS = 8
MIN_SAVES = 2

#: The wall-clock costs the cadences are derived from, measured once and owned
#: here. Perlmutter A100-40GB, helix 3aa96b2 (one implementation, compiled
#: blocks), d512, B=16 over four nodes: the steady step, one evaluation of the
#: 256-event val set (~2-4 s at B=16, longer on fewer ranks), and one save
#: (3 x 226 MiB: 4 s from four nodes, 11 s from one -- the slower is used).
#: Re-measure when the model or the card changes; nothing else should restate
#: them.
STEP_SECONDS = 0.14
EVAL_SECONDS = 4.0
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
    return max(base_steps, min(w, max(1, steps // 5)))


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


def eval_every(steps, eval_seconds=EVAL_SECONDS, step_seconds=STEP_SECONDS, overhead=0.02,
               target_count=50):
    """Evaluation cadence, in optimizer steps.

    Same shape as the checkpoint cadence and for the same reason: enough points
    to read a curve, never so often that evaluation is a material share of the
    run.
    """
    floor = round(eval_seconds / (step_seconds * overhead))
    # Same ceiling, same reason: at B=512 over this corpus STEPS is ~880 and the
    # 2%-overhead floor is 4,000, so an uncapped rule evaluates ZERO times and
    # the run reports no curve at all.
    ceiling = max(1, steps // MIN_EVALS)
    return min(max(floor, max(1, steps // target_count)), ceiling)


def cadence(n_train_events, epochs, batch_size, d, d_base, *,
            warmup_base_steps=500, step_seconds=STEP_SECONDS, overhead=0.02,
            ckpt_seconds=CKPT_SECONDS, save_target_count=20,
            eval_seconds=EVAL_SECONDS, eval_target_count=50):
    """All three, plus the step budget they derive from.

    Returns ``dict(STEPS, WARMUP, SAVE_EVERY, EVAL_EVERY)`` so a config can do

        _c = cadence(N_TRAIN_EVENTS, epoch, batch_size, model["d"], model["d_base"])
        STEPS, WARMUP = _c["STEPS"], _c["WARMUP"]

    and there is exactly one place where the rule lives.

    Keyword-only and explicit, not ``**kw``: a filtered ``**kw`` dropped a
    misspelled name in silence, and one ``target_count`` reached the save
    cadence but not the eval cadence -- one name, two behaviours.
    """
    steps = n_train_events * epochs // batch_size
    return dict(
        STEPS=steps,
        WARMUP=warmup_steps(steps, d, d_base, base_steps=warmup_base_steps),
        SAVE_EVERY=save_every(steps, ckpt_seconds=ckpt_seconds,
                              step_seconds=step_seconds, overhead=overhead,
                              target_count=save_target_count),
        EVAL_EVERY=eval_every(steps, eval_seconds=eval_seconds,
                              step_seconds=step_seconds, overhead=overhead,
                              target_count=eval_target_count),
    )
