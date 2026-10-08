#!/bin/bash
# Label efficiency with the near-activity recipe (whole noise windows + near-activity
# cells and windows + presence head), from scratch vs fine-tuned, then every run
# without a near-evaluation row is scored with --near/--annot:
#   denoise_le3.sh <jobid>
set -uo pipefail
J=$1
H=$(cd "$(dirname "$0")/../.." && pwd); W=${HELIX_STUDY_DIR:-/pscratch/sd/o/oalter/helix_work}
A=$W/probe/artifact_nz_pw16_s1
nodes=($(scontrol show hostnames $(squeue -j $J -h -o %N)))
echo "[le3] $J on ${nodes[*]} at $(date)"
runs=(); for n in 64 256 1024 4096 16384; do runs+=("ft:3e-3:$n" "scratch:1e-2:$n"); done
for ((r = 0; r < ${#runs[@]}; r += 4)); do
  i=0; for spec in "${runs[@]:r:4}"; do IFS=: read -r init lr n <<< "$spec"
    extra=(); [ "$init" = ft ] && extra=(--init $A)
    $H/scripts/study/denoise_node.sh $J ${nodes[$i]} le3_${init}_N${n} $((29920 + r + i)) --arch-from $A "${extra[@]}" \
      --n-events $n --steps 4000 --warmup 300 --lr $lr --lr-head 1e-2 --val-every 250 --ckpt-every 1000 \
      --val-events 64 --cov-per-sig 2 --win-per-event 48 --near-any-per-sig 1 --win-near-per-event 24 --presence &
    i=$((i + 1)); done; wait
  echo "[le3] round $((r / 4 + 1)) done at $(date)"
done

ev() {   # ev <node> <gpu> <tag>...
  local n=$1 g=$2; shift 2
  for t in "$@"; do
    ck=$W/denoise/$t/best.pt; [ -f $ck ] || ck=$W/denoise/$t/final.pt
    [ -f $ck ] || { echo "[le3] no checkpoint for $t"; continue; }
    srun --jobid=$J -N1 -n1 -w $n --overlap --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE \
      CUDA_VISIBLE_DEVICES=$g $H/scripts/helix_run.sh python $H/scripts/eval_denoise.py --checkpoint $ck \
      --truth $W/resolution/truth_v2 --near $W/resolution/truth_v2_near --annot $W/denoise/noise_vs_hits.npz \
      --tag $t --out $W/denoise/results/dn_near.jsonl > $W/denoise/results/evalnear_$t.log 2>&1 \
      && echo "[le3] scored $t" || echo "[le3] FAILED $t"
  done
}
tags=(); for n in 64 256 1024 4096 16384; do tags+=(le3_ft_N$n le3_scratch_N$n le2_ft_N$n le2_scratch_N$n); done
tags+=(M4w_ft_full_win nr_ft1024_pres_s1 nr2_ft1024_pres)
k=0; for t in "${tags[@]}"; do
  ev ${nodes[$((k % 4))]} $(((k / 4) % 4)) $t &
  k=$((k + 1)); [ $((k % 16)) -eq 0 ] && wait
done; wait
echo "[le3] all done at $(date)"
