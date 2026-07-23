#!/bin/bash
# COMPACT masking-direction diagnostic (not a convergence run).
# Matched short budget so the only variable is the masking AXIS. We read
# ratio-to-classical as a mutual-information probe per direction:
#   random = cross-band MI (can shortcut A4/D4/D3 -> D2 at same wire/time)
#   plane  = cross-plane MI (hide a plane, predict from the other 5)
#   block  = cross-wire MI  (hide a wire-slab across all bands)
# ~200 ev staged (fast), 1000 steps. Learning curve printed every 200.
set -e
cd "$(dirname "$0")"
COMMON="--events 200 --steps 1000 --d 256 --blocks 8 --cache_dir ../artifacts/fm_cache_tpc"

echo "=== random @0.50 (cross-band) ==="
python train.py $COMMON --mask_mode random --mask 0.50
echo "=== random @0.75 (cross-band, harsher) ==="
python train.py $COMMON --mask_mode random --mask 0.75
echo "=== plane n=1 (cross-plane, hide 1/6) ==="
python train.py $COMMON --mask_mode plane --n_planes 1
echo "=== block @0.60 (cross-wire slab) ==="
python train.py $COMMON --mask_mode block --mask 0.60
echo "=== diag done ==="
