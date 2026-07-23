#!/bin/bash
#SBATCH --job-name=fm_probe_ds
#SBATCH --partition=ampere --account=mli:cider-ml --gpus=1 --cpus-per-task=16 --mem=96G --time=01:30:00
#SBATCH --output=slurm_logs/probe_ds_%j.out --error=slurm_logs/probe_ds_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
for S in 50000 100000 150000 200000 250000 300000; do
  singularity exec --nv -B /sdf $IMG python3 probe_3d_mlp.py --out probe_dscale.jsonl \
    --ckpt ckpt_dscale_20k_snap${S}.pt --tag d20k_${S} --serial 1 --rope_split 0
done
echo "###### DONE ######"
