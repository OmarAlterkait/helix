#!/bin/bash
#SBATCH --job-name=fm_pblabels
#SBATCH --partition=ampere
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --gpus=1 --cpus-per-task=8 --mem=48G --time=00:40:00
#SBATCH --output=slurm_logs/pblab_%j.out --error=slurm_logs/pblab_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 pb_labels.py --events ${EVENTS:-30000-30009}
