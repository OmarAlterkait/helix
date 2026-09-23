#!/bin/bash
# Run one profiling script from THIS checkout, in the container, on any site.
#
#   source scripts/helix_env.sh      # site facts -> environment
#   sbatch -A "$HELIX_SLURM_TRAIN_ACCOUNT" -q "$HELIX_SLURM_TRAIN_QOS" -N 1 --gpus-per-node=1 \
#          -o <log> tools/profile/sbatch_run.sh p1_shapes_step.py [args...]
#
# Account, QOS and constraint are site facts (helix/sites/<site>.yaml) and reach
# sbatch as flags because #SBATCH cannot read variables. Results go to $PROF_OUT,
# else <HELIX_EXP>/profiling/out (tools/profile/common.py:prof_out).
#
# An earlier version ran the scripts from a copy in one user's S3DF directory,
# not from the repository, so the committed tools were not what it executed.
#SBATCH --time=02:00:00
set -euo pipefail
cd "${HELIX_ROOT:?source scripts/helix_env.sh first}"
export HELIX_INTERACTIVE=0
[ -n "${PROF_OUT:-}" ] && export HELIX_FORWARD_ENV="PROF_OUT"
echo "node=$(hostname) job=${SLURM_JOB_ID:-} script=$*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
scripts/helix_run.sh python "tools/profile/$1" "${@:2}"
