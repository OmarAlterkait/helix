#!/bin/bash
# One LINK of the coefficient-FM stable phase, on preemptable GPUs.
#
# A link is short by design and the run is a CHAIN of them. Preemptable QoS has
# priority 1 against normal's 10000, so a long job is not a job that runs for
# long — it is a job that waits. Short links start sooner, and losing one costs
# at most SAVE_EVERY steps.
#
# Two independent recovery paths, because they fail differently:
#   --requeue        SLURM puts the SAME job back on preemption, mid-link.
#   --dependency     the NEXT link starts when this one ends for any reason,
#                    including hitting the wall clock.
# Both land here, and both are handled by the same rule below: resume if a
# checkpoint exists, start fresh if not. Nothing has to know WHY it restarted.
#
# Usage — submit a chain of N links:
#   scripts/chain_coeff_fm_train.sh <N> [config]
# or a single link by hand:
#   sbatch --export=ALL,CFG=<config> scripts/submit_coeff_fm_train.sh
#
#SBATCH --job-name=coeff8
#SBATCH --partition=ampere
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --requeue
# NO --signal=B:USR1: SLURM delivers it, pimm installs no handler, and the
# default disposition for SIGUSR1 is to TERMINATE. Link 1 died at 03:54:40 of a
# 4 h limit with State=FAILED ExitCode=0:10 — signal 10 is SIGUSR1. It bought
# nothing (there is no graceful-checkpoint path to trigger) and cost five
# minutes plus a FAILED that reads like a crash.
set -euo pipefail

H=${HELIX_ROOT:-/sdf/group/neutrino/omara/helix-extraction}
CFG=${CFG:-$H/configs/pimm/coeff_fm_train_8run.py}
IMG=${IMG:-/sdf/data/neutrino/youngsam/images/pimm-latest.sif}
PIMM=${PIMM_ROOT:-/sdf/group/neutrino/omara/pimm-fm}
PDATA=${PIMM_DATA_SRC:-/sdf/group/neutrino/omara/pimm-data/src}

# K=128 at full event size needs Ampere: the logits are (n_cells, n_slot, K),
# ~2.4 GB for one event, and the backward doubles it. An 11 GB Turing card is
# where the smoke config lives, not this.
[ -f "$CFG" ] || { echo "FATAL: no config at $CFG"; exit 1; }

# ASK THE CONFIG, do not re-derive from its text. Grepping out save_path and
# recomputing STEPS in shell would be a second source of truth for numbers the
# config already computes — and a `_base_`-derived config keeps most of them in
# its parent, where the grep would not see them at all.
export PYTHONPATH="$PIMM:$H:$PDATA"
read -r SAVE TOTAL < <(apptainer exec -B /sdf,/lscratch "$IMG" \
  env PYTHONPATH="$PYTHONPATH" /opt/pimm/.venv/bin/python - "$CFG" <<'PY'
import sys
from pimm.utils.config import Config
c = Config.fromfile(sys.argv[1])
print(c.save_path, int(c.STEPS))
PY
) || { echo "FATAL: could not resolve $CFG"; exit 1; }
[ -n "$SAVE" ] && [ -n "$TOTAL" ] || { echo "FATAL: config gave no save_path/STEPS"; exit 1; }
echo "config: $CFG -> $SAVE, $TOTAL steps"

# Resume iff there is something to resume from. This is what makes a preempted
# link, a wall-clock link and a fresh start all the same case — the alternative
# is a flag the submitter has to get right on every link but the first, which is
# a rule that holds until the one time it does not.
# ASK PIMM which checkpoint to resume from. `pimm.utils.path latest-checkpoint`
# checks `last`, `last.prev` AND `model_last.pth`, validates each properly
# (`.complete` sentinel plus weights.pth plus a complete trainer.dcp), and picks
# the newest by mtime.
#
# This replaces a hand-rolled `[ -d last ] && [ -f last/.complete ]`, which was a
# strictly weaker subset in three ways, one of them dangerous:
#   * it knew only `last/`, so it silently missed the flat `model_last.pth` the
#     older run writes (that was one of two bugs that restarted a chain at step 1);
#   * it checked only the sentinel, not that weights.pth and trainer.dcp were
#     actually there;
#   * with a TORN `last/` — sentinel absent mid-save — it fell through to
#     `resume=False` and would have retrained from ZERO, where pimm falls back to
#     `last.prev`. The comment above it claimed the opposite.
#
# `resume=True` still needs `weight=` beside it: pimm resumes from `cfg.weight`
# (utils/checkpoints.py:921) and logs "No weight found" without it. That is why
# pimm's own train.sh passes `resume=$RESUME weight=$WEIGHT` together.
WEIGHT=$(apptainer exec -B /sdf,/lscratch "$IMG" \
  env PYTHONPATH="$PYTHONPATH" /opt/pimm/.venv/bin/python -m pimm.utils.path \
  latest-checkpoint "$SAVE/model" 2>/dev/null || true)
if [ -n "$WEIGHT" ]; then
  OPTS="resume=True weight=$WEIGHT"
  # A FLOOR on progress, not the step. `iter_N.pth` is written on the
  # CheckpointSaver's cadence, so N is always a multiple of SAVE_EVERY and the
  # true step is somewhere in [N, N+SAVE_EVERY). The exact step lives in
  # `last/trainer.dcp`, which cannot be read without building a trainer, and
  # `iter_N.pth` is not the resumable artifact anyway — `last/` is.
  #
  # That asymmetry is what makes this safe: the floor can only UNDER-report, so
  # the test below can miss a finished run (costing one link, which pimm itself
  # then exits in seconds with "Training already complete") but can never skip an
  # unfinished one. Report it as a floor rather than as the step — reading
  # "step 112500 of 112679" as real progress is what produced a spurious
  # "chain exhausted" on a run that had in fact completed at 112,677.
  FLOOR=$(ls "$SAVE"/model/iter_*.pth 2>/dev/null |
          sed 's/.*iter_\([0-9]*\)\.pth/\1/' | sort -n | tail -1)
  FLOOR=${FLOOR:-0}
  echo "resuming from $WEIGHT (at least step ${FLOOR} of ${TOTAL}; pimm decides)"
  if [ "$FLOOR" -ge "$TOTAL" ]; then
    echo "training already complete (>=${FLOOR}/${TOTAL}) — nothing to do"
    exit 0
  fi
else
  OPTS="resume=False"
  echo "no complete checkpoint under $SAVE/model — starting fresh, target ${TOTAL} steps"
fi

mkdir -p "$SAVE"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
# Rank 0 of a 4-rank job is the only one that writes; the others must not race
# it for the same HDF5 page cache. Left at pimm's default otherwise.
export HDF5_USE_FILE_LOCKING=FALSE

# PREFLIGHT. A node whose GPUs are held reports healthy to SLURM and then fails
# the job in ~60 s — and with `afterany` chaining that eats one link per minute:
# links 2, 3 and 4 all died on sdfampere010 with
#   torch.AcceleratorError: CUDA-capable device(s) is/are busy or unavailable
# having consumed three links in three minutes while the node still showed
# State=MIXED.
#
# So check before committing the link, and REQUEUE rather than exit: a bad node
# then costs a trip through the queue instead of a place in the chain. The sleep
# is not politeness — without it a requeue that lands on the same node hot-loops.
if ! apptainer exec --nv -B /sdf,/lscratch "$IMG" /opt/pimm/.venv/bin/python -c '
import sys, torch
n = torch.cuda.device_count()
assert n >= 4, f"only {n} GPU(s) visible"
for i in range(4):
    torch.zeros(8, device=f"cuda:{i}")      # a context, not just a count
' 2>&1; then
  echo "PREFLIGHT FAILED on $(hostname) — requeueing rather than burning a chain link"
  sleep 60
  scontrol requeue "$SLURM_JOB_ID" || exit 1
  exit 0
fi

srun --kill-on-bad-exit=1 apptainer exec --nv -B /sdf,/lscratch "$IMG" \
  /opt/pimm/.venv/bin/python -m pimm.train \
    --config-file "$CFG" --num-gpus 4 --options $OPTS
