#!/bin/bash
# Full-label run with auto-resume across allocations, then the near-activity evaluation:
#   denoise_full.sh <tag> <port> <jobname> [extra train_denoise.py args...]
# Routed around the hung /pscratch OST 61 (overlay inputs, CFS outputs).
set -uo pipefail
T=$1; PORT=$2; JN=$3; shift 3
H=$(cd "$(dirname "$0")/../.." && pwd); W=/pscratch/sd/o/oalter/helix_work; A=$W/probe/artifact_nz_pw16_s1
O=/global/cfs/cdirs/m5238/users/oalter/ost61; R=/global/cfs/cdirs/m5238/users/oalter/helix_runs
mkdir -p $R/denoise/results; export DENOISE_TRUTH=$O/denoise_truth HELIX_STUDY_DIR=$R
cd $H
while [ ! -f $R/denoise/$T/final.pt ]; do
  until timeout 60 salloc --no-shell --qos=interactive --account=m5238_g --constraint=gpu --nodes=4 --gpus-per-node=4 --time=04:00:00 --job-name=$JN > /dev/null 2>&1; do sleep 30; done
  J=$(squeue -u $USER -n $JN -h -o %i | head -1); echo "[full] $T: got $JN = $J at $(date)"
  scripts/study/denoise_multi.sh $J $T $PORT --corpus-root $O/corpus --arch-from $A --init $A --steps 40000 --warmup 1000 \
    --lr 3e-3 --lr-head 1e-2 --val-every 2000 --ckpt-every 2000 --val-events 64 --cov-per-sig 2 --win-per-event 48 \
    --near-any-per-sig 1 --win-near-per-event 24 --presence --workers 8 "$@"
  echo "[full] $T: segment ended at $(date): $(grep '^step' $R/denoise/$T-$J.log 2>/dev/null | tail -1 | cut -c1-30)"
  if [ -f $R/denoise/$T/final.pt ] && [ -n "$(squeue -j $J -h 2>/dev/null)" ]; then
    n=$(scontrol show hostnames $(squeue -j $J -h -o %N) | head -1)
    srun --jobid=$J -N1 -n1 -w $n --overlap --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=0 \
      scripts/helix_run.sh python scripts/eval_denoise.py --checkpoint $R/denoise/$T/final.pt --q0 2043.2578108909702 \
      --truth $W/resolution/truth_v2 --near $W/resolution/truth_v2_near --annot $W/denoise/noise_vs_hits.npz \
      --tag $T --out $R/denoise/results/dn_full.jsonl > $R/denoise/results/evalfull_$T.log 2>&1 && echo "[full] scored $T"
  fi
  scancel $J 2>/dev/null; echo "[full] released $J"; sleep 30
done
echo "[full] $T done at $(date)"
