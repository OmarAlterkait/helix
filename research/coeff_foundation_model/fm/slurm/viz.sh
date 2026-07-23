#!/bin/bash
#SBATCH --job-name=fm_viz
#SBATCH --partition=ampere --account=mli:nu-ml-dev --qos=normal --gpus=1 --cpus-per-task=8 --mem=96G --time=00:40:00
#SBATCH --output=slurm_logs/viz_%j.out --error=slurm_logs/viz_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
singularity exec --nv -B /sdf $IMG python3 viz_maerecon.py --ckpt ckpt_dscale600_20k.pt --serial 1 --tag d20k
singularity exec --nv -B /sdf $IMG python3 viz_maerecon.py --ckpt ckpt_dscale600_80k.pt --serial 1 --tag d80k
echo ALLDONE
