#!/bin/bash
#SBATCH --job-name=fm_pfinal
#SBATCH --partition=ampere
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --gpus=1 --cpus-per-task=8 --mem=64G --time=00:15:00
#SBATCH --output=slurm_logs/pfinal_%j.out --error=slurm_logs/pfinal_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 probe_final.py
