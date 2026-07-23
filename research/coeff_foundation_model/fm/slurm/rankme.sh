#!/bin/bash
#SBATCH --job-name=fm_rankme --partition=ampere --account=mli:default --qos=preemptable
#SBATCH --gpus=1 --cpus-per-task=8 --mem=48G --time=00:20:00
#SBATCH --output=slurm_logs/rankme_%j.out --error=slurm_logs/rankme_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 rankme_cmp.py
