#!/bin/bash
#SBATCH --job-name=fm_probe_long
#SBATCH --partition=ampere --account=mli:cider-ml --gpus=1 --cpus-per-task=16 --mem=96G --time=00:30:00
#SBATCH --output=slurm_logs/probe_long_%j.out --error=slurm_logs/probe_long_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
CK=$1; TAG=$2
singularity exec --nv -B /sdf $IMG python3 probe_3d_mlp.py --out probe_serial_long.jsonl \
  --ckpt "$CK" --tag "$TAG" --serial 1 --rope_split 0
