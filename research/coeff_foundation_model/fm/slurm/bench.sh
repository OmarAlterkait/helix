#!/bin/bash
# bench.sh — run a one-off script (default bench_mae.py) on one GPU inside the
# PIMM container. Submit: sbatch [--account=.. --qos=..] slurm/bench.sh [script.py]
#SBATCH --job-name=mae_bench
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120000M
#SBATCH --time=00:30:00
set -e
FMDIR=/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
IMAGE=/sdf/data/neutrino/youngsam/containers/pimm.sif
SCRIPT=${1:-bench_mae.py}
echo "=== $SCRIPT on $(hostname)  CUDA=${CUDA_VISIBLE_DEVICES:-?}  $(date) ==="
singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch "$IMAGE" \
  bash -lc "cd $FMDIR && PYTHONPATH='' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 $SCRIPT"
echo "=== bench done $(date) ==="
