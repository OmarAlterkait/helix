"""Score a trained checkpoint on the PROBE split of one corpus — no training step.

**Why this exists.** The two runs in the 8-run-vs-1-run comparison did not log
comparable numbers. `coeff-fm-train-r1` (2026-08-16) predates the fix that made
the grid-free metrics fire, so its logs carry only `bce`/`val` — no `var_expl`,
no `charge_closure`, no `charge_resid`. And even the CE is not comparable: each
run evaluated its OWN val split (577 run-1 events vs 4,641 across eight runs),
at different `max_batches`. A head-to-head needs both models scored on one event
set through one code path.

**Why the `probe` split and not `val`.** `assign_split` keys on event IDENTITY
(`blake2b(run/source_file)` + event) and depends only on `(seed, fractions)` —
never on which runs are in the corpus. Both configs use seed 0 and
train .95 / val .03 / probe .02, so these 388 events are held out from BOTH
models by construction, whether the model saw one run or eight. `val` drove model
selection during training; `probe` was reserved for exactly this measurement.

**Pass the checkpoint at launch**, so one config serves both arms and neither is
described by a path baked into a file:

    scripts/eval_checkpoint.py --config configs/pimm/coeff_fm_eval_probe.py \
        --options weight=<run>/model/model_ema.pth save_path=<somewhere>

Use `model_ema.pth`, not `last`: the schedule is flat by design for the whole
stable phase, so the raw weights sit at full LR noise and the EMA is what stands
in for an annealed model. `model_ema.pth` is `{"state_dict", "decay", "step"}`,
which is the shape pimm's weight loader reads.
"""

_base_ = ["./coeff_fm_train.py"]

# Weights only — this is a measurement, not a continuation. `resume=True` would
# also pull optimizer/scheduler/step state, and `weight` alone is inert without
# it ONLY for resume; for a plain weight restore this is the right pair.
weight = None                      # supplied via --options
resume = False
evaluate = True
save_path = "exp/coeff_fm_eval_probe"

# pimm's `batch_size` is GLOBAL across ranks, and FMTrainer requires exactly one
# event per GPU — the model has no event separation, so a rank holding two would
# attend across unrelated events. The base sets 4 for its 4-rank launches; this
# runs on ONE GPU, so 4 would trip FMTrainer's guard before any eval happened.
batch_size = 1
batch_size_val = 1
batch_size_test = 1

# The bin table must be the one the checkpoint's head was TRAINED against: a
# K=128 categorical head indexes a specific partition, so decoding an older run
# through a newer table reads its logits against edges it never saw. The three
# runs used three tables (v0 pre-R1, v1 R1, v2 R1+measured centroids), so this is
# an override rather than a constant.
#
# `_os`, and `del` after: Config._file2dict keeps every module-level name not
# starting with `__`, so a bare `os` would enter the config dict as a MODULE and
# Config.dump would emit `os = <module 'os'>`, which yapf rejects -- killing the
# run during setup. Same trap the base file documents.
import os as _os
_BINS = (_os.environ.get("COEFF_EVAL_BINS")
         or "/sdf/data/neutrino/omara/archive/coeff_bins_r1_tau05_run0027575715_v2.pt")
del _os
model = dict(bins=_BINS)

# The whole point: BOTH arms on the SAME 388 events.
_eval_data = dict(split_role="probe")
data = dict(val=_eval_data, test=_eval_data)

# Redefined wholesale. pimm REPLACES a base's list value outright — `_merge_a_into_b`
# takes its positional branch only when `allow_list_keys=True`, which `_file2dict`
# does not pass, so a list fails the dict-recursion test and lands on `b[k] = v`.
# (Positional merge exists only on the `--options hooks.0.foo=…` CLI path.)
# Stating it because the opposite is easy to assume and wrong: shortening the list
# here would NOT inherit the base's `every_n_steps`/`save_freq` — it would drop the
# base's hooks entirely, which is what 119 of 124 configs in pimm do unremarked.
hooks = [
    # Still required: pimm dumps the RESOLVED config, which drops the sys.path
    # statements at the top of the base file, and anything re-reading that dump
    # cannot import helix without this.
    dict(type="HelixPathBootstrap"),
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="InformationWriter", log_frequency=200),
    # max_batches=None: score every event in the split rather than a prefix.
    # A truncated eval is what makes two "val numbers" quietly incomparable,
    # which is the defect this config exists to remove.
    dict(type="CoeffFMEvaluator", max_batches=None),
]
