#!/bin/bash
# Corpus-input Tikhonov / oracle floors (causes/floor.py --inputs corpus) on the CALIBRATION events (ev041 skipped:
# OST 61) -> m7diag/cal_floor, so that a model + linear-inversion blend can be fitted off the test events.
cd /global/u1/o/oalter/nu/helix
R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; O=$R/m7diag/cal_floor; mkdir -p $O
r=(0-20 21-40 42-52 53-63)
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu scripts/helix_run.sh python \
    baselines/uboone_sp/causes/floor.py --events ${r[$k]} --inputs corpus --out $O > $R/m7diag/logs/calfloor_$k.log 2>&1 &
done
wait
