#!/bin/bash
# M6d on the corpus re-built with the classical chain's noise draw (dense/model_dump_cn.py), 4 GPUs -> CFS.
cd /global/u1/o/oalter/nu/helix
D=/global/cfs/cdirs/m5238/users/oalter/helix_runs
r=(260-291 292-323 324-355 356-387)
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 scripts/helix_run.sh python \
    baselines/uboone_sp/dense/model_dump_cn.py --events ${r[$k]} --out $D/causes2/model_cn \
    --ckpt M6d=$D/denoise/M6d_ft_full_dec/final.pt > $D/causes2/logs/dump_cn_$k.log 2>&1 &
done
wait
