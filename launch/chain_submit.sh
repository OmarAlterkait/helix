#!/bin/bash
# Submit a CHAIN of dependent training jobs, because this cluster cancels
# preempted jobs rather than requeueing them (PreemptMode=CANCEL, QoS
# preemptable = "within,cancel"). Each link starts when the previous one ENDS for
# any reason (--dependency=afterany) and resumes from the last checkpoint, so a
# preemption costs at most SAVE_EVERY=238 steps instead of the whole run.
#
#   ./chain_submit.sh <n_jobs> <run_name> [extra sbatch args...]
#
# Every link carries ALLOW_RESUME=1: after a preempt-cancel the continuation is a
# NEW job with SLURM_RESTART_COUNT=0, which the wrapper otherwise treats as a
# run-name collision and refuses.
set -euo pipefail
N=${1:?usage: chain_submit.sh <n_jobs> <run_name> [sbatch args...]}
RUN=${2:?usage: chain_submit.sh <n_jobs> <run_name> [sbatch args...]}
shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PREV=""
for i in $(seq 1 "$N"); do
  DEP=""
  [ -n "$PREV" ] && DEP="--dependency=afterany:${PREV}"
  # shellcheck disable=SC2086
  JID=$(sbatch --parsable $DEP "$@" \
        --export=ALL,RUN_NAME="${RUN}",ALLOW_RESUME=1 \
        "${HERE}/coeff_fm_train.sbatch")
  echo "link ${i}/${N}: job ${JID}${PREV:+ (after ${PREV})}"
  PREV=$JID
done
echo "chain submitted; each link resumes ${RUN} from its last checkpoint"
