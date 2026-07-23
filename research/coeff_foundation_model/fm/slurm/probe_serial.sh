#!/bin/bash
#SBATCH --job-name=fm_probe_serial
#SBATCH --partition=ampere --account=mli:cider-ml --gpus=1 --cpus-per-task=16 --mem=96G --time=01:00:00
#SBATCH --output=slurm_logs/probe_serial_%j.out --error=slurm_logs/probe_serial_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
run(){ singularity exec --nv -B /sdf $IMG python3 probe_3d_mlp.py --out probe_serial_ab.jsonl "$@"; }
echo "###### BASE (full attention) ######"
run --ckpt ckpt_nll_base_fix_snap300000.pt --tag base_full
echo "###### SERIAL norope (global RoPE) ######"
run --ckpt ckpt_serial_norope.pt --tag serial_norope --serial 1 --rope_split 0
echo "###### SERIAL split (cross-plane time-only) ######"
run --ckpt ckpt_serial_split.pt --tag serial_split --serial 1 --rope_split 1
echo "###### DONE ######"
