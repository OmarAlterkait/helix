#!/bin/bash
# M2n full-plane maps with v2/model_dump.py (unchanged), 4 GPUs, events 260-387.
cd /global/u1/o/oalter/nu/helix
S=/pscratch/sd/o/oalter/helix_work/uboone_sp/dense
CK=/pscratch/sd/o/oalter/helix_work/denoise/M2n_scratch_full_near/final.pt
r=(260-291 292-323 324-355 356-387)
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE scripts/helix_run.sh python \
    baselines/uboone_sp/v2/model_dump.py --events ${r[$k]} --out $S/model_m2n --ckpt M2n=$CK > $S/logs/m2n_$k.log 2>&1 &
done
wait
