#!/bin/bash
#SBATCH --job-name=fm_pbprobe
#SBATCH --partition=ampere
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --gpus=1 --cpus-per-task=8 --mem=100G --time=02:00:00
#SBATCH --output=slurm_logs/pbprobe_%A_%x.out --error=slurm_logs/pbprobe_%A_%x.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 pb_probe.py --arm ${ARM:-nll}
