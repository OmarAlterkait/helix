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
#SBATCH --signal=B:USR1@300
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
LAST="$SAVE/model/model_last.pth"
if [ -f "$LAST" ]; then
  OPTS="resume=True"
  # Stop the chain rather than burn a GPU-hour re-entering a finished run: later
  # links are submitted up front, so most of them exist to be unnecessary.
  DONE=$(apptainer exec -B /sdf,/lscratch "$IMG" /opt/pimm/.venv/bin/python - "$LAST" <<'PY' 2>/dev/null || echo 0
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(int(ck.get("trainer", {}).get("global_step", 0)))
except Exception:
    print(0)
PY
)
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

srun --kill-on-bad-exit=1 apptainer exec --nv -B /sdf,/lscratch "$IMG" \
  /opt/pimm/.venv/bin/python -m pimm.train \
    --config-file "$CFG" --num-gpus 4 --options $OPTS
