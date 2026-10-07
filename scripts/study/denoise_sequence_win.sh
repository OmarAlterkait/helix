#!/bin/bash
# Second programme (windows recipe) on ONE 4-node interactive allocation: finish the
# fine-tuned full-label run (M4w), then the label-efficiency grid with whole noise
# windows (le2_*), scratch (lr 1e-2) vs fine-tuned (lr 3e-3), rounds of 4.
#   denoise_sequence_win.sh <jobid>
set -uo pipefail
J=$1
H=$(cd "$(dirname "$0")/../.." && pwd); W=${HELIX_STUDY_DIR:-/pscratch/sd/o/oalter/helix_work}
A=$W/probe/artifact_nz_pw16_s1
nodes=($(scontrol show hostnames $(squeue -j $J -h -o %N)))
echo "[seq2] $J on ${nodes[*]} at $(date)"
$H/scripts/study/denoise_multi.sh $J M4w_ft_full_win 29766 --arch-from $A --init $A --steps 40000 --warmup 1000 \
  --lr 3e-3 --lr-head 1e-2 --val-every 2000 --ckpt-every 2000 --val-events 64 --cov-per-sig 2 --win-per-event 48
echo "[seq2] M4w done at $(date)"
runs=(); for n in 64 256 1024 4096 16384; do runs+=("scratch:1e-2:$n" "ft:3e-3:$n"); done
for ((r = 0; r < ${#runs[@]}; r += 4)); do
  i=0; for spec in "${runs[@]:r:4}"; do IFS=: read -r init lr n <<< "$spec"
    extra=(); [ "$init" = ft ] && extra=(--init $A)
    $H/scripts/study/denoise_node.sh $J ${nodes[$i]} le2_${init}_N${n} $((29850 + r + i)) --arch-from $A "${extra[@]}" \
      --n-events $n --steps 4000 --warmup 300 --lr $lr --lr-head 1e-2 --val-every 250 --ckpt-every 1000 \
      --val-events 64 --cov-per-sig 2 --win-per-event 48 &
    i=$((i + 1)); done; wait
  echo "[seq2] label-efficiency round $((r / 4 + 1)) done at $(date)"
done
echo "[seq2] all done at $(date)"
