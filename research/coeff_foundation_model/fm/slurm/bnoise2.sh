#!/bin/bash
#SBATCH --job-name=fm_bnoise2
#SBATCH --partition=ampere --account=mli:nu-ml-dev --qos=normal --gpus=1 --cpus-per-task=16 --mem=300G --time=02:30:00
#SBATCH --output=slurm_logs/bnoise2_%j.out --error=slurm_logs/bnoise2_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
singularity exec --nv -B /sdf $IMG python3 measure_bnoise.py --out bnoise2.jsonl \
  --ckpt ckpt_dscale600_80k_snap600000.pt --tag b80k_600k_M384 --M 384 --Bs 1,2,4,8,16,32,48,64,96,128,192,256
echo DONE
