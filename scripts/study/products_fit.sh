#!/bin/bash
# Fit scripts/denoise_products.py (prior correction, Jensen table, clean-map tau) for every denoiser on the
# calibration events ev000-063 minus ev041, from dense/model_dump2.py dumps. One json per model (merged later).
# Inside an allocation:  srun --jobid=<J> --overlap -n1 -c 120 scripts/study/products_fit.sh [NAME ...]
cd "$(dirname "$0")/../.."
R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; P=$R/products; D=$R/denoise
declare -A CK=( [M4n]=/pscratch/sd/o/oalter/helix_work/denoise/M4n_ft_full_near/final.pt
                [M6d]=$D/M6d_ft_full_dec/final.pt [M6dc]=$D/M6dc_ft_full_dec_cond/final.pt )
declare -A DUMP=( [M4n]=$R/draws/pt_stored [M6d]=$R/draws/pt_stored [M6dc]=$R/draws/pt_stored )
for t in lb_dec_hits lb_dec_cond_hits lb_dec_ct lb_dec_cond_ct; do CK[$t]=$D/$t/best.pt; DUMP[$t]=$R/firstp/model_cal; done
for t in sp_dec_c25 sp_dec_c102 sp_dec_cond_c25 sp_dec_cond_c102; do CK[$t]=$D/$t/best.pt; DUMP[$t]=$P/model_cal; done
CK[M7d]=$D/M7d_ft_full_dec_ct25/final.pt; CK[M7dc]=$D/M7dc_ft_full_dec_cond_ct25/final.pt; DUMP[M7d]=$P/model_cal_m7d; DUMP[M7dc]=$P/model_cal_m7dc
CK[M10lin]=$D/M10lin_ft_full_dec_cond_ct25/final.pt; DUMP[M10lin]=$P/model_cal_m10lin
CK[M11]=$D/M11_ft_full_dec_cond_ct25/final.pt; DUMP[M11]=$P/model_cal_m11
NAMES=${@:-M4n M6d M6dc lb_dec_hits lb_dec_cond_hits lb_dec_ct lb_dec_cond_ct}
for n in $NAMES; do
  HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 scripts/helix_run.sh python scripts/denoise_products.py fit \
    --model $n:${CK[$n]}:${DUMP[$n]} --workers ${WORKERS:-24} --out $P/params_$n.json > $P/logs/fit_$n.log 2>&1 &
done
wait
