# Sourced by the study submitters: site facts, the study directory, one sbatch wrapper.
#
#   source scripts/study/common.sh
#   study_sbatch <nodes> <job-name> <slot.sbatch|cool1.sbatch> "<K=V,K=V,...>" [extra sbatch args]
#
# DRY=1 (the default) prints the sbatch line instead of submitting; DRY=0 submits.
# Scheduler facts come from the site profile via helix_env.sh (#SBATCH lines cannot
# read variables, so they reach sbatch as CLI flags). Runs, logs and checkpoints go
# under HELIX_STUDY_DIR, default $HELIX_SCRATCH/helix_work.
set -euo pipefail
STUDY_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$STUDY_HERE/../helix_env.sh" >/dev/null
export HELIX_STUDY_DIR=${HELIX_STUDY_DIR:-$HELIX_SCRATCH/helix_work}
mkdir -p "$HELIX_STUDY_DIR"
DRY=${DRY:-1}

study_sbatch() {
  local nodes=$1 name=$2 script=$3 exp=$4; shift 4
  local args=(--nodes="$nodes" --job-name="$name"
              --account="${HELIX_SLURM_TRAIN_ACCOUNT:?no scheduler.train.account in the site profile}"
              --constraint="${HELIX_SLURM_TRAIN_CONSTRAINT:-gpu}" --qos="${STUDY_QOS:-preempt}"
              --output="$HELIX_STUDY_DIR/slot-%x-%j.out"
              "--export=ALL,HELIX_ROOT=$HELIX_ROOT,HELIX_STUDY_DIR=$HELIX_STUDY_DIR,$exp" "$@" "$STUDY_HERE/$script")
  if [ "$DRY" = 1 ]; then echo "[dry] sbatch ${args[*]}"; else echo "$name $(sbatch --parsable "${args[@]}")"; fi
}
