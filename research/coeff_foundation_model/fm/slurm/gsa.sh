#!/bin/bash
#SBATCH --job-name=fm_gsa_prof
#SBATCH --partition=ampere --account=mli:cider-ml --gpus=1 --cpus-per-task=16 --mem=96G --time=00:30:00
#SBATCH --output=slurm_logs/gsa_%j.out --error=slurm_logs/gsa_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
singularity exec --nv -B /sdf --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $IMG \
    python3 prof_gsa.py --ns 16000,32000,64000,128000,256000
