#!/bin/bash
# E3: LR edge -- short probes to find divergence. Pick highest stable LR for later.
set -e
cd "$(dirname "$0")"
C="--events 6000 --steps 2000 --d 512 --blocks 10 --dec_blocks 4 --mask_mode random --mask 0.75 \
   --vis_w 1.0 --workers 8 --warmup 300 --eval_every 1000 --eval_n 80 --cache_dir ../artifacts/fm_cache_tpc"
for lr in 8e-4 1.5e-3 3e-3; do
  echo "=== LR $lr ==="
  python train.py $C --lr $lr --tag lr_$lr
done
echo "=== E3 done ==="
