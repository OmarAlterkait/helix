#!/bin/bash
# scripts/eval_denoise.py --products for every denoiser (stored corpus + 5 draws, complete truth), two runs per GPU.
# Inside an allocation:  srun --jobid=<J> --overlap -n1 -c 120 --gpus-per-node=4 scripts/study/products_eval.sh NAME ...
cd "$(dirname "$0")/../.."
W=/pscratch/sd/o/oalter/helix_work; R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; P=$R/products; D=$R/denoise
declare -A CK=( [M4n]=$W/denoise/M4n_ft_full_near/final.pt [M6d]=$D/M6d_ft_full_dec/final.pt
                [M6dc]=$D/M6dc_ft_full_dec_cond/final.pt )
for t in lb_dec_hits lb_dec_cond_hits lb_dec_ct lb_dec_cond_ct sp_dec_c25 sp_dec_c102 sp_dec_cond_c25 sp_dec_cond_c102; do
  CK[$t]=$D/$t/best.pt; done
i=${GPU0:-0}
for n in "$@"; do
  CUDA_VISIBLE_DEVICES=$(( i % 4 )) HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 scripts/helix_run.sh python \
    scripts/eval_denoise.py --checkpoint ${CK[$n]} --q0 2043.2578108909702 --truth $W/resolution/truth_v2 \
    --near $W/resolution/truth_v2_near --annot $W/denoise/noise_vs_hits.npz \
    --complete-truth /global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete --draws ${DRAWS:-5} \
    --products $P/params_$n.json --tag $n --out $P/results/dn_products_$n.jsonl > $P/logs/eval_$n.log 2>&1 &
  i=$((i + 1))
done
wait
