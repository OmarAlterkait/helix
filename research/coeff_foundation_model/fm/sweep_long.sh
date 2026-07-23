#!/bin/bash
# Matched LONG pair to settle cross-band vs cross-plane (optimization-bound check).
# Same budget; only the masking AXIS differs. Learning curve every 200 steps lets
# us see whether cross-plane CATCHES UP (=> harder optimization) or PLATEAUS below
# cross-band (=> genuinely lower cross-plane MI). 400 ev staged (fast, no OOM).
set -e
cd "$(dirname "$0")"
COMMON="--events 400 --steps 6000 --d 256 --blocks 8 --cache_dir ../artifacts/fm_cache_tpc"

echo "=== random @0.50 (cross-band) LONG ==="
python train.py $COMMON --mask_mode random --mask 0.50
echo "=== plane n=1 (cross-plane) LONG ==="
python train.py $COMMON --mask_mode plane --n_planes 1
echo "=== long pair done ==="
