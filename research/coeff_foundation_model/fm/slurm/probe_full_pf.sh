#!/bin/bash
#SBATCH --job-name=fm_probe_fpf
#SBATCH --partition=ampere --account=mli:cider-ml --gpus=1 --cpus-per-task=16 --mem=96G --time=00:45:00
#SBATCH --output=slurm_logs/probe_fpf_%j.out --error=slurm_logs/probe_fpf_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
P(){ singularity exec --nv -B /sdf $IMG python3 probe_3d_mlp.py --out probe_full_pf.jsonl --ckpt "$1" --tag "$2"; }
P ckpt_pf0_lr16.pt  full_pf0
P ckpt_pf10_lr16.pt full_pf10
P ckpt_pf25_lr16.pt full_pf25
echo "###### DONE ######"
