#!/bin/bash
# Probe-train events ev000-259 (held out from denoiser training too; ev041/ev160 skipped: their truth files sit on
# the hung OST 61): stored-corpus maps (model_dump2.py) and fresh draws k=0,1 (model_dump_cn.py --noise tn), M6d/M6dc/M4n.
cd /global/u1/o/oalter/nu/helix
D=/global/cfs/cdirs/m5238/users/oalter/helix_runs
CK="--ckpt M6d=$D/denoise/M6d_ft_full_dec/final.pt --ckpt M6dc=$D/denoise/M6dc_ft_full_dec_cond/final.pt --ckpt M4n=/pscratch/sd/o/oalter/helix_work/denoise/M4n_ft_full_near/final.pt"
R=("0-40 42-64" "65-129" "130-159 161-194" "195-259")
for g in 0 1 2 3; do
  ( for rr in ${R[$g]}; do
      CUDA_VISIBLE_DEVICES=$g HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 scripts/helix_run.sh python \
        baselines/uboone_sp/dense/model_dump2.py --events $rr --out $D/draws/pt_stored $CK >> $D/draws/logs/pt_stored_g$g.log 2>&1
    done
    for k in 0 1; do for rr in ${R[$g]}; do
      CUDA_VISIBLE_DEVICES=$g HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 scripts/helix_run.sh python \
        baselines/uboone_sp/dense/model_dump_cn.py --events $rr --out $D/draws/pt_d$k --noise tn --seed-offset $k $CK >> $D/draws/logs/pt_d${k}_g$g.log 2>&1
    done; done ) &
done
wait
