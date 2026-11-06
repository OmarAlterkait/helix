#!/bin/bash
# After M13 (M7dc recipe + amplitude channel + charge-weighted loss k = 1 + charge-scaling augmentation, original
# volume-mixing layouts) finishes: paired per-bin comparison from 1 k e- vs M11 / M10lin / M7dc (c13), M13's kept
# checkpoints (steps 10k-40k, convergence of the bright error), the partial-scaling gain test (t2d), then the
# products standard.
set -uo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd); D=/global/cfs/cdirs/m5238/users/oalter/helix_runs/denoise; M=/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7tests
T=M13_ft_full_dec_cond_ct25_w1
until [ -f $D/$T/final.pt ]; do sleep 120; done; sleep 120
cd $H
J=""; until [ -n "$J" ]; do
  J=$(timeout 21600 salloc --no-shell --qos=interactive --account=m5238_g --constraint="gpu&hbm80g" --nodes=1 --gpus-per-node=4 \
      --time=02:00:00 --job-name=hx-m13eval 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
  [ -z "$J" ] && sleep 30
done
S=(srun --jobid=$J -N1 -n1 --overlap -c 16 --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE)
B=(--bins 1000,4000,16000,64000,256000)
"${S[@]}" CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/c13_eval_models.py "${B[@]}" \
  --ckpt M11=$D/M11_ft_full_dec_cond_ct25/final.pt --ckpt M13=$D/$T/final.pt --ckpt M10lin=$D/M10lin_ft_full_dec_cond_ct25/final.pt \
  --ckpt M7dc=$D/M7dc_ft_full_dec_cond_ct25/final.pt --ref M11 --out $M/c13_m13.json > $M/c13_m13.log 2>&1 &
"${S[@]}" CUDA_VISIBLE_DEVICES=1 scripts/helix_run.sh python baselines/uboone_sp/causes/c13_eval_models.py "${B[@]}" \
  --ckpt s40000=$D/$T/final.pt --ckpt s10000=$D/$T/step10000.pt --ckpt s20000=$D/$T/step20000.pt --ckpt s30000=$D/$T/step30000.pt \
  --ref s40000 --out $M/c13_m13_conv.json > $M/c13_m13_conv.log 2>&1 &
"${S[@]}" CUDA_VISIBLE_DEVICES=2 scripts/helix_run.sh python baselines/uboone_sp/causes/t2d_local.py \
  --ckpt $D/$T/final.pt --events 260-283 --out $M/t2d_M13 > $M/t2d_M13.log 2>&1 &
wait
"${S[@]}" scripts/helix_run.sh python baselines/uboone_sp/causes/t2d_t8_summary.py t2d_M13 > $M/t2d_M13_summary.log 2>&1
scancel $J; echo "[eval_m13] paired + convergence + gain tests done, released $J at $(date)"
scripts/study/products_model.sh M13 $D/$T/final.pt
echo "[eval_m13] products done at $(date)"
