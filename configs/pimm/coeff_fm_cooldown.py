"""pimm config: the COOLDOWN half of warmup-stable-decay.

WSD's stable phase commits to no horizon — that is the point of it. `lr_mode:
const` in m113 is described in mae_ddp.py as "WSD stable phase: flat, no horizon
baked in", so the stable run can be extended indefinitely and the decay is a
SEPARATE, SHORT run started from a stable-phase checkpoint. This is that run.

Why it exists as its own config rather than a tail bolted onto the stable one:

  * A flat-LR checkpoint is NOT an annealed model, and reading one as if it were
    is a live hazard here — m113's checkpoint at 1,010,000 steps is a stable-phase
    checkpoint, and every probe number taken off raw stable weights (rather than
    the EMA) has been noisier than it looked.
  * The stable run therefore does not know when it will end, so it cannot bake a
    decay schedule. Only this config knows `total_steps`, and `1 - sqrt(p)` needs
    it.

What a cooldown BUYS, concretely, beyond a better model: it produces raw weights
worth reading. During the stable phase the raw weights sit at full LR noise and
`WeightEMA` is what stands in for an annealed model; after a cooldown the raw
weights ARE annealed, so the EMA stops being load-bearing and — see the
CheckpointSaver note below — checkpoint selection starts to mean something.

Run (4 GPUs, as the stable phase)::

    srun --partition=ampere --account=mli:cider-ml --gpus=4 --ntasks=4 \\
         --cpus-per-task=8 --mem=256G --time=4:00:00 \\
         singularity exec --nv -B /sdf,/lscratch \\
         /sdf/data/neutrino/youngsam/images/pimm-latest.sif \\
         bash -lc 'python3 -m pimm.train --config-file .../coeff_fm_cooldown.py'
"""

# The sys.path bootstrap lives in the BASE config and runs when pimm execs it
# (Config._file2dict execs each base before merging), so it is not repeated
# here. `custom_imports`, the model, the corpus, the transform pipeline and the
# optimizer are all inherited — this file is only the delta, so the two runs
# cannot drift apart in the parts that must match.
_base_ = ["./coeff_fm_train.py"]

# ---------------------------------------------------------------------------
# what differs from the stable phase
# ---------------------------------------------------------------------------

# WARM START, not resume. The distinction is load-bearing:
#   * `weight=` loads model weights only (CheckpointLoader, hooks/checkpoint.py:196).
#   * `resume=` restores optimizer state AND the step counter — which would put
#     this run at step ~119,000 of a ~14,000-step schedule, i.e. p > 1, i.e. the
#     floor from the first step. A cooldown that silently ran at lr*floor
#     throughout would look like it worked (loss falls) and anneal nothing.
#
# From the RAW final checkpoint, not `model_ema.pth`. The EMA is a stand-in for
# annealing; annealing it would apply the same correction twice, and the result
# is neither the EMA nor a cooled model.
#
# That file records `trainer.global_step = 118950`, `epoch 25` — the stable phase
# ran to completion — and `best_metric_value: -inf`, which is the diagnosis in
# the artifact's own hand: across 118,950 steps `_update_best` never fired once,
# because CheckpointSaver had no `evaluator_every_n_steps`. There is no
# model_best.pth in that directory to point at even if we wanted one.
#
# It carries `module.bin_edges` but NOT the centroid buffers: it predates their
# becoming persistent. `checkpoint._backfill_centroids` derives them from the
# edges on load, which is exactly the pre-fix path it exists for. The v2 sidecar
# this config inherits has edges BIT-IDENTICAL to the v1 the run trained with, so
# `apply_bins` reports no delta — if it ever does report one here, stop: the
# model would be annealing against a grid it was not trained on.
weight = "/sdf/data/neutrino/omara/exp/helix/coeff-fm-train-r1/model/model_last.pth"
resume = False

# Length. The stable phase is 19,034 x 25 // 4 = 118,962 steps; this is ~12% of
# it, which is in the usual 10-20% band for a WSD cooldown. It is the main knob
# worth tuning and the reason to keep this file separate: a different cooldown
# length is a different run, not a different flag on the same one.
epoch = 3
N_TRAIN_EVENTS = 19_034
batch_size = 4
STEPS = N_TRAIN_EVENTS * epoch // batch_size          # 14,275

# No warmup: the model is already trained. Warmup exists to keep a COLD
# transformer stable; re-warming an annealing run just delays the anneal.
#
# `total_steps` is set by Trainer.build_scheduler (pimm engines/train.py:733),
# which OVERWRITES whatever is here with the real horizon — unlike the stable
# phase, where WSDStableLR ignores it, this schedule depends on it, so the two
# must agree. They do: the trainer computes it from epoch x iters, the same
# quantity as STEPS above.
scheduler = dict(type="WSDCooldownLR", warmup=0, total_steps=STEPS, floor=1e-3)

# ~100 evals across the cooldown, matching the stable phase's density, so the
# two curves are read at the same resolution.
EVAL_EVERY = max(50, round(0.0099 * STEPS))
# NOT the stable phase's 0.0020: that rate is floored at 50 steps, and 50 on a
# 14,275-step run is 285 saves of a 710 MB checkpoint — 200 GB to cool one model.
# The floor exists for preemption on a very long run; here ~20 saves loses at
# most 5% of the run to a preemption, which is the same guarantee at 1/14th the
# disk.
SAVE_EVERY = max(250, round(0.05 * STEPS))

save_path = "/sdf/data/neutrino/omara/exp/helix/coeff-fm-cooldown-r1"

# ---------------------------------------------------------------------------
# hooks — same list as the stable phase, with model_best turned back ON
# ---------------------------------------------------------------------------
# This is the one config where `evaluator_every_n_steps` belongs. coeff_fm_train
# deliberately omits it because a FLAT schedule has no best step: raw val loss
# there is a plateau plus noise, and its max is the luckiest eval. An annealing
# schedule does have a minimum, the raw weights are the annealed model, and the
# curve is monotone enough that "best" is a real claim rather than an argmax over
# noise. So here the selected artifact is also the one worth reading.
hooks = [
    dict(type="HelixPathBootstrap"),
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=1),
    dict(type="InformationWriter"),
    # Kept through the cooldown, so a like-for-like comparison against the
    # stable phase's EMA is still possible — not because it is needed after
    # annealing.
    dict(type="WeightEMA", decay=0.9999, save_freq=SAVE_EVERY),
    dict(type="CoeffFMEvaluator", every_n_steps=EVAL_EVERY, max_batches=200),
    dict(type="CheckpointSaver", save_freq=SAVE_EVERY,
         evaluator_every_n_steps=EVAL_EVERY),
]
