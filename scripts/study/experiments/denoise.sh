#!/bin/bash
# Supervised denoising (docs/DENOISE.md). Runs on interactive allocations; each
# function launches its runs and returns. A = the noisy-recipe FM whose arch every
# encoder uses (and whose weights the fine-tuned arms load).
#   source scripts/study/experiments/denoise.sh
#   ceiling <jobid>                        # M1b: from scratch, all train events, all nodes
#   label_eff <jobid> <init: scratch|ft> <lr> <N>...   # one N per node, B=4
set -euo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
W=${HELIX_STUDY_DIR:-/pscratch/sd/o/oalter/helix_work}
A=$W/probe/artifact_nz_pw16_s1

ceiling() {   # the supervised ceiling: d768 from scratch, ~150k events, 40k steps x 16 events
  nohup $H/scripts/study/denoise_multi.sh $1 M1b_scratch_full 29762 --arch-from $A --steps 40000 --warmup 1000 \
    --lr 1e-2 --lr-head 1e-2 --val-every 2000 --ckpt-every 2000 --val-events 64 --cov-per-sig 2 > /dev/null 2>&1 &
}

label_eff() { # N labelled events per run, one run per node; best checkpoint by val MSE
  local J=$1 init=$2 lr=$3; shift 3
  local nodes=($(scontrol show hostnames $(squeue -j $J -h -o %N))) i=0
  for n in "$@"; do
    local tag=le_${init}_N${n}_lr${lr} extra=()
    [ "$init" = ft ] && extra=(--init $A)
    nohup $H/scripts/study/denoise_node.sh $J ${nodes[$i]} $tag $((29800 + RANDOM % 100)) --arch-from $A "${extra[@]}" \
      --n-events $n --steps 4000 --warmup 300 --lr $lr --lr-head 1e-2 --val-every 250 --ckpt-every 1000 \
      --val-events 64 --cov-per-sig 2 > /dev/null 2>&1 &
    i=$((i + 1))
  done
}
