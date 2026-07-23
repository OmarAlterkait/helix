#!/bin/bash
#SBATCH --job-name=fm_xa
#SBATCH --partition=ampere
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --gpus=1 --cpus-per-task=8 --mem=110G --time=02:30:00
#SBATCH --output=slurm_logs/xa_%A_%x.out --error=slurm_logs/xa_%A_%x.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 pb_xattn.py $XAARGS
