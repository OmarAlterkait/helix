#!/bin/bash
# Can cross-volume attention stay? M8a = M7dc fine-tuned 6k steps with the charge-scaling augmentation ONLY, the
# original (time-sorted, volume-mixing) drift layouts -- the missing arm of M8g (layout + augmentation) / M8c (layout
# only), same schedule. Then, on the same allocation: the partial-scaling gain test (t2d, 24 events) and the paired
# bright-cell comparison vs M8g / M8c / M7dc (c13, incl. the cathode region). 4-node PREEMPT allocation (2 h floor).
set -uo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd); W=/pscratch/sd/o/oalter/helix_work; A=$W/probe/artifact_nz_pw16_s1
O=/global/cfs/cdirs/m5238/users/oalter/ost61; R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; M=$R/m7tests
export DENOISE_TRUTH=/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete HELIX_STUDY_DIR=$R
INIT=$R/denoise/M7dc_ft_full_dec_cond_ct25/final.pt; D=$R/denoise
common=(--corpus-root $O/corpus --arch-from $A --init-denoiser $INIT --steps 6000 --warmup 200 --lr 1e-3 --lr-head 3e-3
        --val-every 1000 --ckpt-every 500 --val-events 64 --cov-per-sig 2 --win-per-event 48 --near-any-per-sig 1
        --win-near-per-event 24 --presence --cond-charge --head decoder --support-min-q 25 --workers 8
        --gain-sigma-plane 0.10 --gain-sigma-block 0.05 --clean-root /global/cfs/cdirs/m5238/users/oalter/coeff_tpc_r1
        --override varlen=True fused_qk=True compile_blocks=True)
cd $H
until [ -f $D/M8a/final.pt ] && [ -f $M/c13_m8a.json ] && [ -f $M/t2d_M8a_summary.log ]; do
  J=""; until [ -n "$J" ]; do
    J=$(timeout 21600 salloc --no-shell --qos=preempt --account=m5238_g --constraint="gpu&hbm80g" --nodes=4 --gpus-per-node=4 \
        --time-min=02:00:00 --time=24:00:00 --job-name=hx-m8a 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
    [ -z "$J" ] && sleep 30
  done
  echo "[m8a] got $J at $(date)"
  [ -f $D/M8a/final.pt ] || scripts/study/denoise_multi.sh $J M8a 30121 "${common[@]}"
  echo "[m8a] training ended at $(date): $(grep '^step' $D/M8a-$J.log 2>/dev/null | tail -1 | cut -c1-40)"
  if [ -f $D/M8a/final.pt ]; then
    S=(srun --jobid=$J -N1 -n1 --overlap -c 16 --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE)
    "${S[@]}" CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/c13_eval_models.py \
      --ckpt M8g=$D/M8g/final.pt --ckpt M8a=$D/M8a/final.pt --ckpt M8c=$D/M8c/final.pt --ckpt M7dc=$INIT \
      --ref M8g --out $M/c13_m8a.json > $M/c13_m8a.log 2>&1 &
    "${S[@]}" CUDA_VISIBLE_DEVICES=1 scripts/helix_run.sh python baselines/uboone_sp/causes/t2d_local.py \
      --ckpt $D/M8a/final.pt --events 260-283 --out $M/t2d_M8a > $M/t2d_M8a.log 2>&1 &
    wait
    "${S[@]}" scripts/helix_run.sh python baselines/uboone_sp/causes/t2d_t8_summary.py t2d_M8a > $M/t2d_M8a_summary.log 2>&1
  fi
  scancel $J; echo "[m8a] released $J at $(date)"
done
echo "[m8a] all done at $(date)"
