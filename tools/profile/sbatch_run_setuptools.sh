#!/bin/bash
#SBATCH --output=/sdf/data/neutrino/omara/exp/helix/profiling/logs/%x-%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
set -euo pipefail
P=/sdf/data/neutrino/omara/exp/helix/profiling
export PROF_OUT=$P/out
export HELIX_INTERACTIVE=0
export HELIX_EXTRA_PYTHONPATH=/sdf/scratch/users/o/omara/hx_setuptools/site
export TORCHINDUCTOR_CACHE_DIR=/lscratch/$USER/inductor_$SLURM_JOB_ID
export TRITON_CACHE_DIR=/lscratch/$USER/triton_$SLURM_JOB_ID
export HELIX_FORWARD_ENV="PROF_OUT TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR"
cd /sdf/group/neutrino/omara/helix
echo "node=$(hostname) job=$SLURM_JOB_ID script=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
# helix_run.sh sets PYTHONPATH=pimm:helix; prepend the setuptools site dir by
# wrapping the interpreter call instead of fighting the script.
scripts/helix_run.sh python -c "
import sys; sys.path.insert(0, '$HELIX_EXTRA_PYTHONPATH')
import runpy; sys.argv = ['$P/$1'] + '''${@:2}'''.split()
runpy.run_path('$P/$1', run_name='__main__')
"
