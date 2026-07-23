#!/bin/bash
#SBATCH --job-name=fm_ridge3d
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --gpus=1 --cpus-per-task=8 --mem=96G --time=01:00:00
#SBATCH --output=slurm_logs/ridge3d_%j.out --error=slurm_logs/ridge3d_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif \
  python3 probe_3d_ridge.py --ckpt "${CKPT}" --tag "${TAG}" --layer "${LAYER:-12}"
