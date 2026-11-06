#!/bin/bash
# M4n / M5d / M5dc full-plane maps with dense/model_dump2.py, 4 GPUs, events 260-387, output on CFS.
cd /global/u1/o/oalter/nu/helix
D=/global/cfs/cdirs/m5238/users/oalter/helix_runs
O=$D/dense2/model
r=(260-291 292-323 324-355 356-387)
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE scripts/helix_run.sh python \
    baselines/uboone_sp/dense/model_dump2.py --events ${r[$k]} --out $O \
    --ckpt M4n=/pscratch/sd/o/oalter/helix_work/denoise/M4n_ft_full_near/final.pt \
    --ckpt M5d=$D/denoise/M5d_ft_full_dec/final.pt --ckpt M5dc=$D/denoise/M5dc_ft_full_dec_cond/final.pt \
    > $D/dense2/logs/dump_$k.log 2>&1 &
done
wait
