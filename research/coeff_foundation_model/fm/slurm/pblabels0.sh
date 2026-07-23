#!/bin/bash
#SBATCH --job-name=fm_pblab0
#SBATCH --partition=ampere
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --gpus=1 --cpus-per-task=8 --mem=48G --time=00:30:00
#SBATCH --output=slurm_logs/pblab0_%j.out --error=slurm_logs/pblab0_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
QTOT_MIN=0 singularity exec -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 pb_labels.py --events 30000-30002
