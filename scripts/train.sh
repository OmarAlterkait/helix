#!/bin/bash
# Train: a helix pimm config (configs/pimm/CONFIG.py, or a path) as a chain of jobs on this site's
# scheduler (helix/sites/<site>.yaml, scheduler.train), each running pimm's trainer in the site's
# container (scripts/helix_run.sh):
#
#   scripts/train.sh fm                                one link
#   LINKS=8 scripts/train.sh fm                        eight, each starting when the last one ends
#   scripts/train.sh fm_cooldown
#   GPUS=1 scripts/train.sh fm_smoke epoch=2           anything after CONFIG is a pimm --options key=value
#
# A run lives in $HELIX_EXP/NAME (NAME=CONFIG by default). Each link resumes from the run's latest
# complete checkpoint, or starts the run when there is none, so a link that is preempted, times out
# or fails costs at most one checkpoint interval and the next one carries on: the chain uses
# --dependency=afterany, which continues whether the scheduler requeues a preempted job (NERSC) or
# cancels it (S3DF). Submitting a config again under the same NAME continues that run; with
# different options it is refused (another NAME starts another run).
#
# Environment: LINKS (1), NAME, GPUS (4, one node), TIME (24:00:00), MEM (50G per GPU, where the
# site allocates by partition), LOGS ($HELIX_EXP/NAME/slurm).
set -euo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$H/scripts/helix_env.sh"
if [ "${1:-}" = link ]; then LINK=1; shift; else LINK=0; fi
CFG=${1:?usage: scripts/train.sh CONFIG [key=value ...]}
shift
case "$CFG" in
  */*|*.py) CFG=$(realpath "$CFG") ;;
  *) CFG=$H/configs/pimm/$CFG.py ;;
esac
[ -f "$CFG" ] || { echo "train.sh: no config $CFG" >&2; exit 1; }
NAME=${NAME:-$(basename "$CFG" .py)}
GPUS=${GPUS:-4}
SAVE=$HELIX_EXP/$NAME
RUN=("$H/scripts/helix_run.sh")
export HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE HELIX_FORWARD_ENV="HDF5_USE_FILE_LOCKING"

if [ "$LINK" = 1 ]; then                 # inside a job
  WEIGHT=$("${RUN[@]}" python -m pimm.utils.path latest-checkpoint "$SAVE/model") \
    || { echo "train.sh: the checkpoint resolver failed for $SAVE/model; not starting over" >&2; exit 2; }
  if [ -n "$WEIGHT" ]; then RESUME=(resume=True "weight=$WEIGHT"); echo "resuming from $WEIGHT"
  else RESUME=(resume=False); echo "starting $SAVE"; fi
  # one task per GPU: pimm takes its ranks from Slurm's tasks (pimm.utils.comm.setup_distributed)
  exec srun --kill-on-bad-exit=1 "${RUN[@]}" python -m pimm.train --config-file "$CFG" \
    --num-gpus "$GPUS" --options "save_path=$SAVE" "${RESUME[@]}" "$@"
fi

# what this run is: the same NAME continues only the same config and options
mkdir -p "$SAVE"
WHAT="$CFG $*"
if [ -s "$SAVE/train.cmd" ] && [ "$(cat "$SAVE/train.cmd")" != "$WHAT" ]; then
  echo "train.sh: $SAVE is a run of: $(cat "$SAVE/train.cmd")" >&2
  echo "          not of:            $WHAT  (set NAME= for a new run)" >&2
  exit 2
fi
echo "$WHAT" > "$SAVE/train.cmd"

LOGS=${LOGS:-$SAVE/slurm}
mkdir -p "$LOGS"
SB=(--account="$HELIX_SLURM_TRAIN_ACCOUNT" --nodes=1 --ntasks-per-node="$GPUS" --cpus-per-task=8
    --time="${TIME:-24:00:00}" --job-name="$NAME" --output="$LOGS/%j.out" --requeue)
[ -n "${HELIX_SLURM_TRAIN_QOS:-}" ] && SB+=(--qos="$HELIX_SLURM_TRAIN_QOS")
[ -n "${HELIX_SLURM_TRAIN_CONSTRAINT:-}" ] && SB+=(--constraint="$HELIX_SLURM_TRAIN_CONSTRAINT")
[ -n "${HELIX_SLURM_TRAIN_TIME_MIN:-}" ] && SB+=(--time-min="$HELIX_SLURM_TRAIN_TIME_MIN")
if [ -n "${HELIX_SLURM_TRAIN_PARTITION:-}" ]; then      # S3DF: a partition, memory asked for
  SB+=(--partition="$HELIX_SLURM_TRAIN_PARTITION" --mem="${MEM:-$((50 * GPUS))G}")
fi
if [ "${HELIX_SLURM_TRAIN_GPUS_FLAG:-gpus}" = gpus-per-node ]; then SB+=(--gpus-per-node="$GPUS")
else SB+=(--gpus="$GPUS"); fi
read -r -a EXTRA <<< "${HELIX_SLURM_TRAIN_EXTRA:-}"
SB+=("${EXTRA[@]}")

PREV=""
for i in $(seq 1 "${LINKS:-1}"); do
  JID=$(sbatch --parsable "${SB[@]}" ${PREV:+--dependency=afterany:$PREV} --export=ALL,NAME="$NAME",GPUS="$GPUS" \
        --wrap="bash $H/scripts/train.sh link $(printf '%q ' "$CFG" "$@")")
  echo "link $i: job $JID${PREV:+ (after $PREV)}"
  PREV=$JID
done
echo "run: $SAVE   logs: $LOGS"
