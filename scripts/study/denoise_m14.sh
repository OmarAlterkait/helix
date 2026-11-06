#!/bin/bash
# The conditional charge as a DISTRIBUTION (m7_tests_results.md, M12 / M13: a point head trained by squared error
# assumes every cell's charge is equally uncertain; charge weighting by the TRUE charge reallocates capacity but
# biases the faint side). Three 6k-step fine-tunes of M10lin, identical to M12's except the charge head / loss:
#   M14g05  gauss, beta-NLL beta 0.5     M14g0  gauss, plain likelihood     M14hl  hlgauss, 1024 bins, smoothed targets
# Each starts exactly at M10lin's head (CellDecoder.init_dist_from_point). Then, on the same allocation: paired
# per-bin comparison from 1 k e- vs the k = 0 control M12w0 (and M12w1), and the distributions' calibration (c15).
set -uo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd); W=/pscratch/sd/o/oalter/helix_work; A=$W/probe/artifact_nz_pw16_s1
O=/global/cfs/cdirs/m5238/users/oalter/ost61; R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; M=$R/m7tests
export DENOISE_TRUTH=/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete HELIX_STUDY_DIR=$R
D=$R/denoise; INIT=$D/M10lin_ft_full_dec_cond_ct25/final.pt
common=(--corpus-root $O/corpus --arch-from $A --init-denoiser $INIT --steps 6000 --warmup 200 --lr 1e-3 --lr-head 3e-3
        --val-every 1000 --ckpt-every 500 --keep-every 2000 --val-events 64 --cov-per-sig 2 --win-per-event 48
        --near-any-per-sig 1 --win-near-per-event 24 --presence --cond-charge --head decoder --support-min-q 25 --workers 8
        --override varlen=True fused_qk=True compile_blocks=True lin_amp_scale=30.0)
declare -A EXTRA=( [M14g05]="--dist gauss --beta 0.5" [M14g0]="--dist gauss --beta 0" [M14hl]="--dist hlgauss --n-bins 1024" )
TAGS=(M14g05 M14g0 M14hl)
alldone() { for t in "${TAGS[@]}"; do [ -f $D/$t/final.pt ] || return 1; done; [ -f $M/c13_m14.json ] && [ -f $M/c15_m14.json ]; }
cd $H
until alldone; do
  J=""; until [ -n "$J" ]; do
    J=$(timeout 21600 salloc --no-shell --qos=interactive --account=m5238_g --constraint="gpu&hbm80g" --nodes=4 --gpus-per-node=4 \
        --time=04:00:00 --job-name=hx-m14 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
    [ -z "$J" ] && sleep 30
  done
  echo "[m14] got $J at $(date)"
  for T in "${TAGS[@]}"; do
    [ -n "$(squeue -j $J -h 2>/dev/null)" ] || break
    [ -f $D/$T/final.pt ] && continue
    scripts/study/denoise_multi.sh $J $T 30151 "${common[@]}" ${EXTRA[$T]}
    echo "[m14] $T ended at $(date): $(grep '^step' $D/$T-$J.log 2>/dev/null | tail -1 | cut -c1-40)"
  done
  ok=1; for t in "${TAGS[@]}"; do [ -f $D/$t/final.pt ] || ok=0; done
  if [ $ok = 1 ] && [ -n "$(squeue -j $J -h 2>/dev/null)" ]; then
    S=(srun --jobid=$J -N1 -n1 --overlap -c 16 --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE)
    "${S[@]}" CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/c13_eval_models.py \
      --bins 1000,4000,16000,64000,256000 --ckpt M12w0=$D/M12w0/final.pt --ckpt M14g05=$D/M14g05/final.pt \
      --ckpt M14g0=$D/M14g0/final.pt --ckpt M14hl=$D/M14hl/final.pt --ckpt M12w1=$D/M12w1/final.pt \
      --ref M12w0 --out $M/c13_m14.json > $M/c13_m14.log 2>&1 &
    "${S[@]}" CUDA_VISIBLE_DEVICES=1 scripts/helix_run.sh python baselines/uboone_sp/causes/c15_dist_calib.py \
      --ckpt M14g05=$D/M14g05/final.pt --ckpt M14g0=$D/M14g0/final.pt --ckpt M14hl=$D/M14hl/final.pt \
      --out $M/c15_m14.json > $M/c15_m14.log 2>&1 &
    wait
  fi
  scancel $J; echo "[m14] released $J at $(date)"; sleep 30
done
echo "[m14] all done at $(date)"
