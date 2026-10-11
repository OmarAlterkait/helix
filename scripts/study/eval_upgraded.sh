#!/bin/bash
# Upgraded denoiser evaluation (scripts/eval_denoise.py --draws 5 --complete-truth) for M6d, M6dc, M4n,
# one GPU each, outputs on CFS. Run inside an allocation:  srun --jobid=<J> --overlap --gpus-per-node=4 scripts/study/eval_upgraded.sh
cd "$(dirname "$0")/../.."
W=/pscratch/sd/o/oalter/helix_work; R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; E=$R/eval_draws
mkdir -p $E/results $E/logs
i=0
for spec in M6d_ft_full_dec=$R/denoise/M6d_ft_full_dec/final.pt M6dc_ft_full_dec_cond=$R/denoise/M6dc_ft_full_dec_cond/final.pt \
            M4n_ft_full_near=$W/denoise/M4n_ft_full_near/final.pt; do
  T=${spec%%=*}; CK=${spec#*=}
  CUDA_VISIBLE_DEVICES=$i HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE scripts/helix_run.sh python scripts/eval_denoise.py \
    --checkpoint $CK --q0 2043.2578108909702 --truth $W/resolution/truth_v2 --near $W/resolution/truth_v2_near \
    --annot $W/denoise/noise_vs_hits.npz --complete-truth $E/complete_truth --draws ${DRAWS:-5} \
    --tag $T --out $E/results/dn_upgraded_$T.jsonl > $E/logs/eval_$T.log 2>&1 &
  i=$((i + 1))
done
wait
