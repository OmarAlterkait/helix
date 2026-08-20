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
# `model/last/` — a DCP DIRECTORY, not model_last.pth. Two things were wrong
# here and each alone was enough to silently restart from scratch:
#
#   * This pimm writes `model/last/{weights.pth,trainer.dcp,.complete}`. The
#     older run in exp/helix/coeff-fm-train-r1 has a flat `model_last.pth`, and
#     that is what this checked for. Nothing matched, so every link passed
#     resume=False. `iter_N.pth` is no substitute — it holds `state_dict` only,
#     with no optimizer, scheduler or step.
#   * `resume=True` ALONE IS INERT. pimm resumes from `cfg.weight`
#     (utils/checkpoints.py:921); with weight unset it logs "No weight found"
#     and trains from zero. Both flags are required.
#
# Link 2 restarted at step 1 with 26,150 steps sitting on disk. Checked by the
# `.complete` marker rather than the directory alone, so a link preempted MID
# save resumes from the previous good checkpoint instead of a torn one.
LAST="$SAVE/model/last"
if [ -d "$LAST" ] && [ -f "$LAST/.complete" ]; then
  OPTS="resume=True weight=$LAST"
  # Highest iter_N.pth as the progress probe: it is a filename, so this needs no
  # torch load, and the DCP directory cannot be read with one anyway.
  DONE=$(ls "$SAVE"/model/iter_*.pth 2>/dev/null |
         sed 's/.*iter_\([0-9]*\)\.pth/\1/' | sort -n | tail -1)
  DONE=${DONE:-0}
  # Stop the chain rather than burn a GPU-hour re-entering a finished run: later
  # links are submitted up front, so most of them exist to be unnecessary.
  echo "resuming from step ${DONE} of ${TOTAL}"
  if [ "$DONE" -ge "$TOTAL" ]; then
    echo "training already complete (${DONE}/${TOTAL}) — nothing to do"
    exit 0
  fi
else
  OPTS="resume=False"
  echo "no checkpoint under $SAVE — starting fresh, target ${TOTAL} steps"
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
