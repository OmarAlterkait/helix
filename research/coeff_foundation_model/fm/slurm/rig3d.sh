#!/bin/bash
#SBATCH --job-name=fm_rig3d
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --gpus=1 --cpus-per-task=8 --mem=64G --time=01:30:00
#SBATCH --output=slurm_logs/rig3d_%j.out --error=slurm_logs/rig3d_%j.out
# env: CKPT TAG LAYER(=12) SEEDS(=6) NORAND(=0 -> arms=trained,random ; 1 -> trained only)
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
ARMS="trained,random"; [ "${NORAND:-0}" = "1" ] && ARMS="trained"
singularity exec --nv -B /sdf /sdf/data/neutrino/youngsam/containers/pimm.sif \
  python3 probe_3d_rigor.py --ckpt "${CKPT}" --tag "${TAG}" \
    --layers "${LAYERS:-12}" --seeds "${SEEDS:-4}" --arms "${ARMS}" \
    --patch "${PATCH:-0}" --out "${OUT:-probe_3d_rigor_wr1.jsonl}"
