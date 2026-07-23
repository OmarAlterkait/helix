#!/bin/bash
#SBATCH --job-name=fm_lay
#SBATCH --partition=ampere
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --gpus=1 --cpus-per-task=8 --mem=48G --time=00:40:00
#SBATCH --output=slurm_logs/lay_%j.out --error=slurm_logs/lay_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 pb_layers.py
