#!/bin/bash
# E4: data-starve edge. Fewer events -> where does train/test gap blow up (overfit)?
# d=512 (fits ~14GB). Compare gap vs 6k-event baseline (~0.03 @5ep).
set -e
cd "$(dirname "$0")"
C="--steps 10000 --d 512 --blocks 10 --dec_blocks 4 --mask_mode random --mask 0.75 \
   --vis_w 1.0 --workers 8 --lr 4e-4 --warmup 1500 --eval_every 2500 --eval_n 80 \
   --cache_dir ../artifacts/fm_cache_tpc"
for ev in 1000 2000; do
  echo "=== EVENTS $ev ==="
  python train.py $C --events $ev --tag data_$ev
done
echo "=== E4 done ==="
