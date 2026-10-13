#!/bin/bash
# Removing the amplitude coupling (m7_tests_results.md C2-C6): fine-tune M7dc, one 4-node allocation, two runs
# one after the other (same schedule, so the augmentation's effect is separated from extra training):
#   M8g  volume-aware drift layouts + charge-scaling augmentation (sigma 0.10 per plane gid, 0.05 per 64-wire block)
#   M8c  volume-aware drift layouts only (control)
#   denoise_m8.sh            (from a login shell: takes and releases its own allocation)
set -uo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd); W=/pscratch/sd/o/oalter/helix_work; A=$W/probe/artifact_nz_pw16_s1
O=/global/cfs/cdirs/m5238/users/oalter/ost61; R=/global/cfs/cdirs/m5238/users/oalter/helix_runs
export DENOISE_TRUTH=/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete HELIX_STUDY_DIR=$R
INIT=$R/denoise/M7dc_ft_full_dec_cond_ct25/final.pt
common=(--corpus-root $O/corpus --arch-from $A --init-denoiser $INIT --steps 6000 --warmup 200 --lr 1e-3 --lr-head 3e-3
        --val-every 1000 --ckpt-every 500 --val-events 64 --cov-per-sig 2 --win-per-event 48 --near-any-per-sig 1
        --win-near-per-event 24 --presence --cond-charge --head decoder --support-min-q 25 --workers 8
        --override varlen=True fused_qk=True compile_blocks=True drift_planes_per_volume=3)
cd $H
J=""; until [ -n "$J" ]; do
  J=$(timeout 90 salloc --no-shell --qos=interactive --account=m5238_g --constraint="gpu&hbm80g" --nodes=4 --gpus-per-node=4 \
      --time=03:00:00 --job-name=hx-m8 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
  [ -z "$J" ] && sleep 30
done
echo "[m8] got $J at $(date)"
for T in M8g M8c; do
  extra=(); [ $T = M8g ] && extra=(--gain-sigma-plane 0.10 --gain-sigma-block 0.05 --clean-root /global/cfs/cdirs/m5238/users/oalter/coeff_tpc_r1)
  [ -f $R/denoise/$T/final.pt ] && { echo "[m8] $T already done"; continue; }
  scripts/study/denoise_multi.sh $J $T 30091 "${common[@]}" "${extra[@]}"
  echo "[m8] $T ended at $(date): $(grep '^step' $R/denoise/$T-$J.log | tail -1 | cut -c1-40)"
done
scancel $J; echo "[m8] released $J at $(date)"
