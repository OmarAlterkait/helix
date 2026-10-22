#!/bin/bash
# Does the OBJECTIVE put the ~3% relative floor where it is? (m7_tests_results.md, Round 2: the networks' bright
# error is flat in charge, a learned linear map's falls ~1/q; the linear map minimises squared error in LINEAR
# charge, the networks squared error in log charge.) Three 6k-step fine-tunes of M10lin (amplitude channel,
# original volume-mixing layouts), identical except the conditional-charge loss weight (1 + q/Q0)^k:
#   M12w0  k = 0  the current loss (control: same extra training)
#   M12w1  k = 1  each unit of charge counts equally
#   M12w2  k = 2  ~ squared error in linear charge (the linear map's objective)
# Output representation (log1p q / Q0) unchanged. Checkpoints kept every 2000 steps. Then, on the same allocation:
# paired per-bin comparison vs M12w0 from 1 k e- up (c13, incl. the faint bins the weighting may cost).
# 4-node PREEMPT allocation (2 h floor, 24 h ceiling), resumable.
set -uo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd); W=/pscratch/sd/o/oalter/helix_work; A=$W/probe/artifact_nz_pw16_s1
O=/global/cfs/cdirs/m5238/users/oalter/ost61; R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; M=$R/m7tests
export DENOISE_TRUTH=/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete HELIX_STUDY_DIR=$R
D=$R/denoise; INIT=$D/M10lin_ft_full_dec_cond_ct25/final.pt
common=(--corpus-root $O/corpus --arch-from $A --init-denoiser $INIT --steps 6000 --warmup 200 --lr 1e-3 --lr-head 3e-3
        --val-every 1000 --ckpt-every 500 --keep-every 2000 --val-events 64 --cov-per-sig 2 --win-per-event 48
        --near-any-per-sig 1 --win-near-per-event 24 --presence --cond-charge --head decoder --support-min-q 25 --workers 8
        --override varlen=True fused_qk=True compile_blocks=True lin_amp_scale=30.0)
TAGS=(M12w0 M12w1 M12w2)
cd $H
until [ -f $D/M12w0/final.pt ] && [ -f $D/M12w1/final.pt ] && [ -f $D/M12w2/final.pt ] && [ -f $M/c13_m12.json ]; do
  J=""; until [ -n "$J" ]; do
    J=$(timeout 21600 salloc --no-shell --qos=preempt --account=m5238_g --constraint="gpu&hbm80g" --nodes=4 --gpus-per-node=4 \
        --time-min=02:00:00 --time=24:00:00 --job-name=hx-m12 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
    [ -z "$J" ] && sleep 30
  done
  echo "[m12] got $J at $(date)"
  for T in "${TAGS[@]}"; do
    [ -n "$(squeue -j $J -h 2>/dev/null)" ] || break
    [ -f $D/$T/final.pt ] && continue
    k=${T#M12w}
    scripts/study/denoise_multi.sh $J $T 30131 "${common[@]}" --charge-weight-pow $k
    echo "[m12] $T ended at $(date): $(grep '^step' $D/$T-$J.log 2>/dev/null | tail -1 | cut -c1-40)"
  done
  if [ -f $D/M12w0/final.pt ] && [ -f $D/M12w1/final.pt ] && [ -f $D/M12w2/final.pt ] && [ -n "$(squeue -j $J -h 2>/dev/null)" ]; then
    srun --jobid=$J -N1 -n1 --overlap -c 16 --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=0 \
      scripts/helix_run.sh python baselines/uboone_sp/causes/c13_eval_models.py --bins 1000,4000,16000,64000,256000 \
      --ckpt M12w0=$D/M12w0/final.pt --ckpt M12w1=$D/M12w1/final.pt --ckpt M12w2=$D/M12w2/final.pt --ckpt M10lin=$INIT \
      --ref M12w0 --out $M/c13_m12.json > $M/c13_m12.log 2>&1
  fi
  scancel $J; echo "[m12] released $J at $(date)"; sleep 30
done
echo "[m12] all done at $(date)"
