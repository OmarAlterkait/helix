#!/bin/bash
# E1: mask-ratio edge. Where does high mask collapse masked-inference? Low mask -> ceiling.
# All d=512, lr 4e-4 (proven), 10k steps. Compare masked var_expl @10k vs mask0.75's 49.5%@10k.
set -e
cd "$(dirname "$0")"
C="--events 6000 --steps 10000 --d 512 --blocks 10 --dec_blocks 4 --mask_mode random \
   --vis_w 1.0 --workers 8 --lr 4e-4 --warmup 1500 --eval_every 2500 --eval_n 80 \
   --cache_dir ../artifacts/fm_cache_tpc"
for mk in 0.5 0.9 0.95; do
  echo "=== MASK $mk ==="
  python train.py $C --mask $mk --tag mask_$mk
done
echo "=== E1 done ==="
