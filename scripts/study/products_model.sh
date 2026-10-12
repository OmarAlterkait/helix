#!/bin/bash
# One denoiser through the whole products standard in one 1-node allocation: calibration dump
# (dense/model_dump2.py, ev000-063 minus ev041) -> scripts/denoise_products.py fit -> scripts/eval_denoise.py
# --products (stored corpus + 5 noise draws, complete truth). NAME must have CK/DUMP entries in
# products_fit.sh and products_eval.sh:
#   products_model.sh NAME CKPT        (from a login shell: takes and releases its own allocation)
set -uo pipefail
N=$1; CK=$2
H=$(cd "$(dirname "$0")/../.." && pwd); R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; P=$R/products
D=$P/model_cal_$(echo $N | tr A-Z a-z)
mkdir -p $P/logs $P/results
J=""; until [ -n "$J" ]; do
  J=$(timeout 90 salloc --no-shell --qos=interactive --account=m5238_g --constraint="gpu&hbm80g" --nodes=1 --gpus-per-node=4 \
      --time=03:00:00 --job-name=hx-prod-$N 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
  [ -z "$J" ] && sleep 30
done
echo "[products] $N: got $J at $(date)"
S=(srun --jobid=$J -N1 -n1 --overlap -c 120 --gpus-per-node=4)
"${S[@]}" env OUT=$D CKPTS="$N=$CK" $H/baselines/uboone_sp/dense/run_products_dump.sh
echo "[products] $N: dumped $(ls $D/ev*.npz | grep -v tmp | wc -l) events at $(date)"
"${S[@]}" env WORKERS=48 $H/scripts/study/products_fit.sh $N && echo "[products] $N: fitted at $(date)"
"${S[@]}" $H/scripts/study/products_eval.sh $N && echo "[products] $N: scored at $(date)"
scancel $J; echo "[products] released $J"
