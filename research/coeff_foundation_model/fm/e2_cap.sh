#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== CAP d=768 enc12 dec4 (~115M) ==="
python train.py --events 6000 --steps 15000 --d 768 --blocks 12 --dec_blocks 4 \
  --mask_mode random --mask 0.75 --vis_w 1.0 --workers 8 --lr 4e-4 --warmup 1500 \
  --eval_every 2500 --eval_n 80 --cache_dir ../artifacts/fm_cache_tpc --tag cap_d768
echo "=== E2 done ==="
