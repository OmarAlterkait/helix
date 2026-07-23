#!/bin/bash
#SBATCH --job-name=fm_bnoise
#SBATCH --partition=ampere --account=mli:nu-ml-dev --qos=normal --gpus=1 --cpus-per-task=16 --mem=180G --time=01:30:00
#SBATCH --output=slurm_logs/bnoise_%j.out --error=slurm_logs/bnoise_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
R(){ singularity exec --nv -B /sdf $IMG python3 measure_bnoise.py --out bnoise.jsonl --M 96 "$@"; }
echo "###### 80k @600k (converged-ish, where continued training sits) ######"
R --ckpt ckpt_dscale600_80k_snap600000.pt --tag b80k_600k
echo "###### 80k @100k (early, for the evolution trend) ######"
R --ckpt ckpt_dscale600_80k_snap100000.pt --tag b80k_100k
echo DONE
