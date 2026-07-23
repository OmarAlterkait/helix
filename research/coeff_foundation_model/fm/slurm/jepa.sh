#!/bin/bash
#SBATCH --job-name=fm_jepa
#SBATCH --partition=ampere
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --gpus=1 --cpus-per-task=8 --mem=64G --time=00:15:00
#SBATCH --output=slurm_logs/jepa_%A_%x.out --error=slurm_logs/jepa_%A_%x.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif python3 train_jepa.py --config $CFG --resume
