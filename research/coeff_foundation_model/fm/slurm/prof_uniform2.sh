#!/bin/bash
#SBATCH --job-name=fm_uniform2
#SBATCH --partition=ampere --account=mli:default --qos=preemptable --gpus=1 --cpus-per-task=8 --mem=96G --time=00:30:00
#SBATCH --output=slurm_logs/uniform2_%j.out --error=slurm_logs/uniform2_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 prof_uniform2.py
