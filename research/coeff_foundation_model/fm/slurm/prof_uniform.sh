#!/bin/bash
#SBATCH --job-name=fm_uniform
#SBATCH --partition=ampere --account=mli:default --qos=preemptable --gpus=1 --cpus-per-task=8 --mem=96G --time=00:30:00
#SBATCH --output=slurm_logs/uniform_%j.out --error=slurm_logs/uniform_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 prof_uniform.py
