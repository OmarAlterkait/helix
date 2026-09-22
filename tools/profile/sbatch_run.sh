#!/bin/bash
#SBATCH --output=/sdf/data/neutrino/omara/exp/helix/profiling/logs/%x-%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# usage: sbatch --job-name=hx-p1 [slurm flags] jobs/run.sh prof/p1_shapes_step.py [args...]
set -euo pipefail
P=/sdf/data/neutrino/omara/exp/helix/profiling
export PROF_OUT=${PROF_OUT:-$P/out}
export HELIX_INTERACTIVE=0
export HELIX_FORWARD_ENV="PROF_OUT"
cd /sdf/group/neutrino/omara/helix
echo "node=$(hostname) job=$SLURM_JOB_ID script=$*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
scripts/helix_run.sh python "$P/$@"
