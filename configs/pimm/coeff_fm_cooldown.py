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
# Derived from the EIGHT-RUN stable phase, which is the run that exists: it
# completed 112,679 steps (3 epochs, 150,239 train events).
#
# Its FINAL eval (train.log:115002, the last of 101 `[coeff-eval]` lines) reads
#   bce 0.1037 / val 2.9488 / var_expl 0.6602 / charge_closure 0.7956 /
#   charge_bias 0.9406
# This comment previously quoted var_expl 0.6507 / charge_closure 0.8397 /
# charge_bias 1.0000 as "final". That tuple is real but is train.log:60613 --
# `Train: [2/3][21589/37559]`, about 52% through -- so it was a MID-RUN eval
# wearing the word final, with five decimal-matched metrics giving it false
# precision.
#
# Both tuples predate the charge-closure fix (E[|X|], not |E[X]|), so BOTH
# charge_closure figures are low by that estimator artifact and neither should be
# quoted as the model's magnitude fidelity. var_expl and bce are unaffected.
#
# The one-run config this used to derive from was superseded before its cooldown
# ever ran, and keeping a cooldown for a stable phase nobody will use is dead
# config.
# A derived config runs BEFORE pimm processes `_base_`, so the base's sys.path
# bootstrap has not happened yet. It does NOT get its own: a config that touches
# sys.path must also register HelixPathBootstrap so a RESUMED job can still
# import helix (tests/test_pimm_config_contract.py pins that pairing), and the
# hook belongs to the base. So this relies on helix already being importable --
# which it is, because the launcher exports PYTHONPATH -- and fails loudly if not.
from helix.paths import root as _root                          # noqa: E402

_base_ = ["./coeff_fm_train_8run.py"]

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
# `model/last` is a DIRECTORY (weights.pth + trainer.dcp + .complete), which is
# what this pimm writes; the flat `model_last.pth` belongs to the older one-run
# job. Getting that backwards restarted a training chain from step 1 once
# already — see scripts/submit_coeff_fm_train.sh.
#
# Loading it needs no centroid backfill: the 8-run checkpoints carry bin_edges,
# bin_cent_asinh and bin_cent_ratio, and the v2 sidecar this config inherits is
# the same table they were trained against, so `apply_bins` reports no delta. If
# it ever does report one here, stop — the model would be annealing against a
# grid it was not trained on.
weight = str(_root("HELIX_EXP") / "coeff-fm-train-r1-8run" / "model" / "last")
resume = False

# Length: ONE EPOCH over the first THREE runs = 14,264 steps = 12.7% of the
# stable phase's 112,679, which is inside the usual 10-20% band for a WSD
# cooldown.
#
# A subset is needed because `epoch` is an integer and one epoch over all eight
# runs is 37,559 steps — 33% of the stable phase, and there is no way to ask for
# a third of an epoch. The subset is a RUN subset rather than `max_len`, which
# would take the first N events of an index ordered by run and so anneal almost
# entirely on run 1.
#
# Annealing on 3 of 8 runs is sound because those runs are the same
# distribution, measured rather than assumed: bins derived on run 2 alone versus
# run 1 alone, both converged, differ by 0.129% of a band's span — 0.81x the
# sampling noise of the shipped table. A subset of an iid corpus is the same
# corpus. See coeff_fm_train_8run.py.
RUNS = [
    "run_0027575715", "run_0027587651", "run_0027651463",
]
_over = dict(data_root=str(_root("HELIX_CORPUS").parent), split=RUNS)
data = dict(train=dict(**_over), val=dict(**_over), test=dict(**_over))

epoch = 1
N_TRAIN_EVENTS = 57_059          # resolved from the identity split over RUNS
batch_size = 4
STEPS = N_TRAIN_EVENTS * epoch // batch_size          # 14,264

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
# 14,264-step run is 285 saves of a 710 MB checkpoint — 200 GB to cool one model.
# The floor exists for preemption on a very long run; here ~20 saves loses at
# most 5% of the run to a preemption, which is the same guarantee at 1/14th the
# disk.
SAVE_EVERY = max(250, round(0.05 * STEPS))

save_path = str(_root("HELIX_EXP") / "coeff-fm-cooldown-r1-8run")

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

del _root
