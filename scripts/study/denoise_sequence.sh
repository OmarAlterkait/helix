#!/bin/bash
# The denoising programme on ONE 4-node interactive allocation, in order:
# finish the supervised ceiling (M1b), the fine-tune LR sweep, then the
# label-efficiency grid (one run per node, rounds of 4). Logs under $W/denoise.
#   denoise_sequence.sh <jobid>
set -uo pipefail
J=$1
H=$(cd "$(dirname "$0")/../.." && pwd); W=${HELIX_STUDY_DIR:-/pscratch/sd/o/oalter/helix_work}
A=$W/probe/artifact_nz_pw16_s1
nodes=($(scontrol show hostnames $(squeue -j $J -h -o %N)))
echo "[seq] $J on ${nodes[*]} at $(date)"

# 1. the ceiling: resume to 40k steps
$H/scripts/study/denoise_multi.sh $J M1b_scratch_full 29763 --arch-from $A --steps 40000 --warmup 1000 \
  --lr 1e-2 --lr-head 1e-2 --val-every 2000 --ckpt-every 2000 --val-events 64 --cov-per-sig 2
echo "[seq] M1b done at $(date)"

# 2. fine-tune LR sweep (encoder LR; head 1e-2), one per node
i=0; for lr in 1e-4 3e-4 1e-3 3e-3; do
  $H/scripts/study/denoise_node.sh $J ${nodes[$i]} lr_ft_$lr $((29750 + i)) --arch-from $A --init $A --steps 3000 \
    --warmup 300 --lr $lr --lr-head 1e-2 --val-every 500 --ckpt-every 1000 --val-events 32 --cov-per-sig 2 &
  i=$((i + 1)); done; wait
best=$(for lr in 1e-4 3e-4 1e-3 3e-3; do echo "$(tail -n 1 $W/denoise/lr_ft_$lr/val.jsonl | python3 -c 'import sys,json; print(json.load(sys.stdin)["mse"])') $lr"; done | sort -g | head -1 | cut -d' ' -f2)
echo "[seq] fine-tune LR sweep done at $(date); best encoder LR $best"

# 3. label efficiency: scratch (lr 1e-2) vs fine-tuned (best) at N labelled events
runs=(); for n in 64 256 1024 4096 16384; do runs+=("scratch:1e-2:$n" "ft:$best:$n"); done
for ((r = 0; r < ${#runs[@]}; r += 4)); do
  i=0; for spec in "${runs[@]:r:4}"; do IFS=: read -r init lr n <<< "$spec"
    extra=(); [ "$init" = ft ] && extra=(--init $A)
    $H/scripts/study/denoise_node.sh $J ${nodes[$i]} le_${init}_N${n} $((29800 + r + i)) --arch-from $A "${extra[@]}" \
      --n-events $n --steps 4000 --warmup 300 --lr $lr --lr-head 1e-2 --val-every 250 --ckpt-every 1000 \
      --val-events 64 --cov-per-sig 2 &
    i=$((i + 1)); done; wait
  echo "[seq] label-efficiency round $((r / 4 + 1)) done at $(date)"
done
echo "[seq] all done at $(date)"
