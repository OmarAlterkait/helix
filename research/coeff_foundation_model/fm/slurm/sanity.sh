#!/bin/bash
#SBATCH --job-name=fm_sanity
#SBATCH --partition=ampere --account=mli:default --qos=preemptable --gpus=1 --cpus-per-task=8 --mem=64G --time=00:15:00
#SBATCH --output=slurm_logs/sanity_%j.out --error=slurm_logs/sanity_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 sanity_serial.py
