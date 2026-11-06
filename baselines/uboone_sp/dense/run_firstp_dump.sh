#!/bin/bash
# Label-arm checkpoints (lb_dec_{hits,cond_hits,ct,cond_ct}/best.pt) on every token-covered cell: the 128 test events
# (firstp/model) and calibration events ev000-063 minus ev041 (hung OST) (firstp/model_cal); M4n/M6d/M6dc calibration
# maps already exist in draws/pt_stored. dense/model_dump2.py, stored corpus (CFS copy).
cd /global/u1/o/oalter/nu/helix
D=/global/cfs/cdirs/m5238/users/oalter/helix_runs
CK=""; for t in lb_dec_hits lb_dec_cond_hits lb_dec_ct lb_dec_cond_ct; do CK="$CK --ckpt $t=$D/denoise/$t/best.pt"; done
T=(260-291 292-323 324-355 356-387); C=("0-15" "16-31" "32-40 42-47" "48-63")
for g in 0 1 2 3; do
  ( CUDA_VISIBLE_DEVICES=$g HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 scripts/helix_run.sh python \
      baselines/uboone_sp/dense/model_dump2.py --events ${T[$g]} --out $D/firstp/model $CK > $D/firstp/logs/dump_test_$g.log 2>&1
    for rr in ${C[$g]}; do
      CUDA_VISIBLE_DEVICES=$g HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 scripts/helix_run.sh python \
        baselines/uboone_sp/dense/model_dump2.py --events $rr --out $D/firstp/model_cal $CK >> $D/firstp/logs/dump_cal_$g.log 2>&1
    done ) &
done
wait
