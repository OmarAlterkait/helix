#!/bin/bash
#SBATCH --job-name=fm_bd
#SBATCH --partition=ampere --account=mli:default --qos=preemptable --gpus=1 --cpus-per-task=8 --mem=128G --time=00:35:00
#SBATCH --output=slurm_logs/bd_%j.out --error=slurm_logs/bd_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 prof_batch_dec.py
