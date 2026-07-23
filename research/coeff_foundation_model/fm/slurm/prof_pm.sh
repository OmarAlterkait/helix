#!/bin/bash
#SBATCH --job-name=fm_prof_pm
#SBATCH --partition=ampere --account=mli:cider-ml --gpus=1 --cpus-per-task=16 --mem=96G --time=00:20:00
#SBATCH --output=slurm_logs/prof_pm_%j.out --error=slurm_logs/prof_pm_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
singularity exec --nv -B /sdf $IMG python3 prof_pipeline.py
