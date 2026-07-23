#!/bin/bash
#SBATCH --job-name=fm_mupcc
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --gpus=1 --cpus-per-task=8 --mem=64G --time=00:25:00
#SBATCH --output=slurm_logs/mupcc_%j.out --error=slurm_logs/mupcc_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # forwarded into the container
# optional: SINGULARITYENV_MUP_WIDTHS=128,256,512 to restrict widths (small-GPU fallback)
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 mup_coordcheck.py
