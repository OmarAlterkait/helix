#!/bin/bash
#SBATCH --job-name=fm_probe_traj
#SBATCH --partition=ampere --account=mli:cider-ml --gpus=1 --cpus-per-task=16 --mem=96G --time=01:00:00
#SBATCH --output=slurm_logs/probe_traj_%j.out --error=slurm_logs/probe_traj_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
for S in 200000 300000 400000; do
  echo "###### snap $S ######"
  singularity exec --nv -B /sdf $IMG python3 probe_3d_mlp.py --out probe_serial_long.jsonl \
    --ckpt ckpt_serial_long_snap${S}.pt --tag long_${S} --serial 1 --rope_split 0
done
echo "###### DONE ######"
