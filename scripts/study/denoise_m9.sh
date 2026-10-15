#!/bin/bash
# Encoder ablations for the ~3% bright floor (m7_tests_results.md): fine-tune M7dc with ONE change each, the same
# 6k-step schedule as M8c (the control: volume-aware drift layouts, no augmentation). 4-node allocations, resumable:
#   M9lin   + linear-amplitude input channel (lin_amp_scale=30; zero-initialised -> starts as M7dc)
#   M9grp   + attention groups doubled (gp 2048, gd 4096)
#   M9phys  physical drift coordinate x = +-(T_cathode - t), drift layouts sorted by x (volumes mix only at the cathode)
#   M8c_s1  M8c with --seed 1 (run-to-run spread of the fine-tuning)
#   denoise_m9.sh            (from a login shell: takes and releases its own allocations)
set -uo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd); W=/pscratch/sd/o/oalter/helix_work; A=$W/probe/artifact_nz_pw16_s1
O=/global/cfs/cdirs/m5238/users/oalter/ost61; R=/global/cfs/cdirs/m5238/users/oalter/helix_runs
export DENOISE_TRUTH=/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete HELIX_STUDY_DIR=$R
INIT=$R/denoise/M7dc_ft_full_dec_cond_ct25/final.pt
common=(--corpus-root $O/corpus --arch-from $A --init-denoiser $INIT --steps 6000 --warmup 200 --lr 1e-3 --lr-head 3e-3
        --val-every 1000 --ckpt-every 500 --val-events 64 --cov-per-sig 2 --win-per-event 48 --near-any-per-sig 1
        --win-near-per-event 24 --presence --cond-charge --head decoder --support-min-q 25 --workers 8)
declare -A EXTRA=(
  [M9lin]="--override varlen=True fused_qk=True compile_blocks=True drift_planes_per_volume=3 lin_amp_scale=30.0"
  [M9grp]="--override varlen=True fused_qk=True compile_blocks=True drift_planes_per_volume=3 gp=2048 gd=4096"
  [M9phys]="--override varlen=True fused_qk=True compile_blocks=True drift_planes_per_volume=3 drift_cathode_tick=3510.0"
  [M8c_s1]="--override varlen=True fused_qk=True compile_blocks=True drift_planes_per_volume=3 --seed 1" )
TAGS=(M9lin M9grp M9phys M8c_s1)
cd $H
while :; do
  todo=(); for t in "${TAGS[@]}"; do [ -f $R/denoise/$t/final.pt ] || todo+=($t); done
  [ ${#todo[@]} -eq 0 ] && break
  J=""; until [ -n "$J" ]; do
    J=$(timeout 21600 salloc --no-shell --qos=interactive --account=m5238_g --constraint="gpu&hbm80g" --nodes=4 --gpus-per-node=4 \
        --time=04:00:00 --job-name=hx-m9 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
    [ -z "$J" ] && sleep 30
  done
  echo "[m9] got $J at $(date): ${todo[*]}"
  for T in "${todo[@]}"; do
    [ -n "$(squeue -j $J -h 2>/dev/null)" ] || break
    scripts/study/denoise_multi.sh $J $T 30093 "${common[@]}" ${EXTRA[$T]}
    echo "[m9] $T segment ended at $(date): $(grep '^step' $R/denoise/$T-$J.log 2>/dev/null | tail -1 | cut -c1-40)"
  done
  scancel $J; echo "[m9] released $J at $(date)"; sleep 30
done
echo "[m9] all done at $(date)"
