#!/bin/bash
# M6d / M6dc full-plane maps with dense/model_dump2.py, 4 GPUs, events 260-387, output on CFS.
cd /global/u1/o/oalter/nu/helix
D=/global/cfs/cdirs/m5238/users/oalter/helix_runs
O=$D/dense3/model
r=(260-291 292-323 324-355 356-387)
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE scripts/helix_run.sh python \
    baselines/uboone_sp/dense/model_dump2.py --events ${r[$k]} --out $O \
    --ckpt M6d=$D/denoise/M6d_ft_full_dec/final.pt --ckpt M6dc=$D/denoise/M6dc_ft_full_dec_cond/final.pt \
    > $D/dense3/logs/dump_$k.log 2>&1 &
done
wait
