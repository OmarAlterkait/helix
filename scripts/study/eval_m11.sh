#!/bin/bash
# After M11 finishes: paired bright-cell test vs M10lin and M7dc, the partial-scaling gain test, then the products standard.
set -uo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd); D=/global/cfs/cdirs/m5238/users/oalter/helix_runs/denoise; M=/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7tests
until [ -f $D/M11_ft_full_dec_cond_ct25/final.pt ]; do sleep 120; done; sleep 120
cd $H
J=""; until [ -n "$J" ]; do
  J=$(timeout 21600 salloc --no-shell --qos=interactive --account=m5238_g --constraint="gpu&hbm80g" --nodes=1 --gpus-per-node=4 \
      --time=02:00:00 --job-name=hx-m11eval 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
  [ -z "$J" ] && sleep 30
done
S=(srun --jobid=$J -N1 -n1 --overlap -c 16 --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE)
"${S[@]}" CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/c13_eval_models.py \
  --ckpt M10lin=$D/M10lin_ft_full_dec_cond_ct25/final.pt --ckpt M11=$D/M11_ft_full_dec_cond_ct25/final.pt \
  --ckpt M7dc=$D/M7dc_ft_full_dec_cond_ct25/final.pt --ref M10lin --out $M/c13_m11.json > $M/c13_m11.log 2>&1 &
"${S[@]}" CUDA_VISIBLE_DEVICES=1 scripts/helix_run.sh python baselines/uboone_sp/causes/t2d_local.py \
  --ckpt $D/M11_ft_full_dec_cond_ct25/final.pt --events 260-283 --out $M/t2d_M11 > $M/t2d_M11.log 2>&1 &
wait
"${S[@]}" scripts/helix_run.sh python baselines/uboone_sp/causes/t2d_t8_summary.py t2d_M11 > $M/t2d_M11_summary.log 2>&1
scancel $J; echo "[eval_m11] paired + gain tests done, released $J at $(date)"
scripts/study/products_model.sh M11 $D/M11_ft_full_dec_cond_ct25/final.pt
echo "[eval_m11] products done at $(date)"
