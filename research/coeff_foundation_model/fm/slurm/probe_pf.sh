#!/bin/bash
#SBATCH --job-name=fm_probe_pf
#SBATCH --partition=ampere --account=mli:cider-ml --gpus=1 --cpus-per-task=16 --mem=96G --time=01:30:00
#SBATCH --output=slurm_logs/probe_pf_%j.out --error=slurm_logs/probe_pf_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
P(){ singularity exec --nv -B /sdf $IMG python3 probe_3d_mlp.py --out probe_pf.jsonl --serial 1 --rope_split 0 --ckpt "$1" --tag "$2"; }
P ckpt_serial_long.pt long_500k
for S in 100000 200000 300000 400000; do P ckpt_serial_pf01_snap${S}.pt pf01_${S}; done
for S in 100000 200000 300000; do P ckpt_serial_pf025_snap${S}.pt pf025_${S}; done
echo "###### DONE ######"
