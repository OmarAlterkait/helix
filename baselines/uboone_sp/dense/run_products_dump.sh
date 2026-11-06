#!/bin/bash
# Support-threshold arms (sp_dec{,_cond}_c{25,102}/best.pt) on every token-covered cell of the calibration events
# ev000-063 minus ev041 (hung OST) -> products/model_cal, the input of scripts/denoise_products.py fit.
# Other models: OUT=<dir> CKPTS="TAG=path ..." run_products_dump.sh  (an existing evNNN.npz is skipped, so a new set needs a new OUT).
cd /global/u1/o/oalter/nu/helix
R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; O=${OUT:-$R/products/model_cal}; mkdir -p $O
CK=""; for t in sp_dec_c25 sp_dec_c102 sp_dec_cond_c25 sp_dec_cond_c102; do CK="$CK --ckpt $t=$R/denoise/$t/best.pt"; done
[ -n "${CKPTS:-}" ] && CK=$(for c in $CKPTS; do printf -- " --ckpt %s" $c; done)
C=("0-15" "16-31" "32-40 42-47" "48-63")
for g in 0 1 2 3; do
  ( for rr in ${C[$g]}; do
      CUDA_VISIBLE_DEVICES=$g HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 scripts/helix_run.sh python \
        baselines/uboone_sp/dense/model_dump2.py --events $rr --out $O $CK >> $R/products/logs/dump_cal_$(basename $O)_$g.log 2>&1
    done ) &
done
wait
