#!/bin/bash
# Fresh noise draws k in $DRAWS (default 0..4) through the corpus's torch-GPU generator path (model_dump_cn.py --noise tn
# --seed-offset k); M6d, M6dc, M4n full-plane maps per draw -> CFS draws/model_d<k>. One GPU per event range.
cd /global/u1/o/oalter/nu/helix
D=/global/cfs/cdirs/m5238/users/oalter/helix_runs
r=(260-291 292-323 324-355 356-387)
for g in 0 1 2 3; do
  ( for k in ${DRAWS:-0 1 2 3 4}; do
      CUDA_VISIBLE_DEVICES=$g HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 scripts/helix_run.sh python \
        baselines/uboone_sp/dense/model_dump_cn.py --events ${r[$g]} --out $D/draws/model_d$k --noise tn --seed-offset $k \
        --ckpt M6d=$D/denoise/M6d_ft_full_dec/final.pt --ckpt M6dc=$D/denoise/M6dc_ft_full_dec_cond/final.pt \
        --ckpt M4n=/pscratch/sd/o/oalter/helix_work/denoise/M4n_ft_full_near/final.pt > $D/draws/logs/d${k}_g$g.log 2>&1
    done ) &
done
wait
