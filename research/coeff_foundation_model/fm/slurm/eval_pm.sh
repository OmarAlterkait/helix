#!/bin/bash
#SBATCH --job-name=fm_eval_pm
#SBATCH --partition=ampere --account=mli:cider-ml --gpus=1 --cpus-per-task=16 --mem=96G --time=00:30:00
#SBATCH --output=slurm_logs/eval_pm_%j.out --error=slurm_logs/eval_pm_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
echo "###### FULL-ATTN base (nll_base_fix) ######"
singularity exec --nv -B /sdf $IMG python3 eval_planemask.py --ckpt ckpt_nll_base_fix.pt --tag base_full
echo "###### GROUPED-SERIAL (serial_long snap400k) ######"
singularity exec --nv -B /sdf $IMG python3 eval_planemask.py --ckpt ckpt_serial_long_snap400000.pt --tag serial_long400 --serial 1 --rope_split 0
echo "###### DONE ######"
