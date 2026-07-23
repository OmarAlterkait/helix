#!/bin/bash
#SBATCH --job-name=fm_probe_b4
#SBATCH --partition=ampere --account=mli:nu-ml-dev --qos=normal --gpus=1 --cpus-per-task=16 --mem=240G --time=03:00:00
#SBATCH --output=slurm_logs/probe_b4_%j.out --error=slurm_logs/probe_b4_%j.out
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IMG=/sdf/data/neutrino/youngsam/containers/pimm.sif
P(){ singularity exec --nv -B /sdf $IMG python3 probe_3d_mlp.py --out probe_b4.jsonl --serial 1 --rope_split 0 --events 30000-30199 "$@"; }
for CK in ckpt_dscale600b4_20k_snap*.pt; do S=$(echo $CK|grep -oE 'snap[0-9]+'); S=${S#snap}
  grep -q "\"b4_20k_${S}\"" probe_b4.jsonl 2>/dev/null && continue; echo "### 20k $S ###"; P --ckpt $CK --tag b4_20k_${S}; done
for CK in ckpt_dscale600b4_80k_snap*.pt; do S=$(echo $CK|grep -oE 'snap[0-9]+'); S=${S#snap}
  grep -q "\"b4_80k_${S}\"" probe_b4.jsonl 2>/dev/null && continue; echo "### 80k $S ###"; P --ckpt $CK --tag b4_80k_${S}; done
echo DONE
