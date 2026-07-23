#!/bin/bash
#SBATCH --job-name=fm_probe_ds600
#SBATCH --partition=ampere --account=mli:nu-ml-dev --qos=normal --gpus=1 --cpus-per-task=16 --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/probe_ds600_%j.out --error=slurm_logs/probe_ds600_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # defrag: passes into container
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
P(){ singularity exec --nv -B /sdf $IMG python3 probe_3d_mlp.py --out probe_ds600.jsonl --serial 1 --rope_split 0 --events 30000-30199 "$@"; }
for D in 20k 80k; do
  for CK in ckpt_dscale600_${D}_snap*.pt ckpt_dscale600_${D}.pt; do
    [ -f "$CK" ] || continue
    S=$(echo "$CK" | grep -oE 'snap[0-9]+' || echo final); S=${S#snap}
    grep -q "\"ds${D}_${S}\"" probe_ds600.jsonl 2>/dev/null && { echo "skip $D $S (done)"; continue; }
    echo "### probing $D $S ###"; P --ckpt "$CK" --tag ds${D}_${S}
  done
done
echo DONE
