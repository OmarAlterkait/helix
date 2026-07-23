#!/bin/bash
#SBATCH --job-name=fm_probe_ends
#SBATCH --partition=ampere --account=mli:nu-ml-dev --qos=normal --gpus=1 --cpus-per-task=16 --mem=240G --time=02:00:00
#SBATCH --output=slurm_logs/probe_ends_%j.out --error=slurm_logs/probe_ends_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
P(){ singularity exec --nv -B /sdf $IMG python3 probe_3d_mlp.py --out probe_ds600.jsonl --serial 1 --rope_split 0 --events 30000-30199 "$@"; }
for D in 20k 80k; do for S in 500000 600000; do
  grep -q "\"ds${D}_${S}\"" probe_ds600.jsonl 2>/dev/null && { echo "skip $D $S"; continue; }
  echo "### $D $S ###"; P --ckpt ckpt_dscale600_${D}_snap${S}.pt --tag ds${D}_${S}
done; done
echo DONE
