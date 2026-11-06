#!/bin/bash
# M7d / M7dc fp32 inference (causes/bright_dump.py --precision fp32) on the test events 260-387, 4 GPUs -> m7diag/fp32.
cd /global/u1/o/oalter/nu/helix
R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; D=$R/denoise; O=$R/m7diag/fp32; mkdir -p $O $R/m7diag/logs
r=(260-291 292-323 324-355 356-387)
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 scripts/helix_run.sh python \
    baselines/uboone_sp/causes/bright_dump.py --events ${r[$k]} --precision fp32 --out $O \
    --ckpt M7d=$D/M7d_ft_full_dec_ct25/final.pt --ckpt M7dc=$D/M7dc_ft_full_dec_cond_ct25/final.pt \
    > $R/m7diag/logs/fp32_$k.log 2>&1 &
done
wait
