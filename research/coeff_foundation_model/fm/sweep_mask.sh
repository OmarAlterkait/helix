#!/bin/bash
# Masking-scheme comparison sweep (single A100, sequential).
# Same model config as the random@0.5 baseline (d=256 blk=8, 1000 ev) so the only
# variable is the masking scheme. Interpretation:
#   random   = patch-level (can shortcut across bands at the same wire/time)
#   plane    = cross-plane MI probe (hide whole plane(s), reconstruct from others)
#   block    = wire-slab tube across all bands (spatial inpainting, no cross-band cheat)
set -e
cd "$(dirname "$0")"
COMMON="--events 1000 --steps 5000 --d 256 --blocks 8 --cache_dir ../artifacts/fm_cache_tpc"

echo "=== [1/5] random @0.50 (re-baseline) ==="
python train.py $COMMON --mask_mode random --mask 0.50
echo "=== [2/5] random @0.75 (harsher) ==="
python train.py $COMMON --mask_mode random --mask 0.75
echo "=== [3/5] plane n=1 (cross-plane, hide 1/6) ==="
python train.py $COMMON --mask_mode plane --n_planes 1
echo "=== [4/5] plane n=3 (cross-plane, hide whole volume) ==="
python train.py $COMMON --mask_mode plane --n_planes 3
echo "=== [5/5] block @0.60 (wire-slab tube) ==="
python train.py $COMMON --mask_mode block --mask 0.60
echo "=== sweep done ==="
