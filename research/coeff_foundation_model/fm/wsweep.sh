#!/bin/bash
cd "$(dirname "$0")"
for W in 8 16 24 32; do
  echo "=== WORKERS $W ==="
  python train.py --events 2000 --steps 300 --d 512 --blocks 10 --dec_blocks 4 --mask 0.75 \
    --vis_w 0 --workers $W --cache_dir ../artifacts/fm_cache_tpc --tag wsweep 2>&1 | grep "step   200\| 200:"
done
echo "=== wsweep done ==="
