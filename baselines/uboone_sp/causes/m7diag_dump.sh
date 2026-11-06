#!/bin/bash
# M7d / M7dc full-plane maps (dense/model_dump2.py) on the test events 260-387, 4 GPUs -> products/model_test_m7 (m7_diagnostics.md).
cd /global/u1/o/oalter/nu/helix
R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; D=/global/cfs/cdirs/m5238/users/oalter/helix_runs/denoise; O=/global/cfs/cdirs/m5238/users/oalter/helix_runs/products/model_test_m7
r=(260-291 292-323 324-355 356-387)
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 scripts/helix_run.sh python     baselines/uboone_sp/dense/model_dump2.py --events ${r[$k]} --out $O     --ckpt M7d=$D/M7d_ft_full_dec_ct25/final.pt --ckpt M7dc=$D/M7dc_ft_full_dec_cond_ct25/final.pt     > $R/products/logs/dump_test_m7_$k.log 2>&1 &
done
wait
