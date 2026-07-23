#!/bin/bash
# Remaining masking configs (runs 1-2 already logged). Lazy loader = RAM-safe.
set -e
cd "$(dirname "$0")"
COMMON="--events 1000 --steps 5000 --d 256 --blocks 8 --cache_dir ../artifacts/fm_cache_tpc"

echo "=== [3/5] plane n=1 (cross-plane, hide 1/6) ==="
python train.py $COMMON --mask_mode plane --n_planes 1
echo "=== [4/5] plane n=3 (cross-plane, hide whole volume) ==="
python train.py $COMMON --mask_mode plane --n_planes 3
echo "=== [5/5] block @0.60 (wire-slab tube) ==="
python train.py $COMMON --mask_mode block --mask 0.60
echo "=== rest done ==="
