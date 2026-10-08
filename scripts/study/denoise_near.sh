#!/bin/bash
# Near-activity ablation + re-scoring on the near-activity evaluation, one 4-node allocation:
#   denoise_near.sh <jobid>
# Nodes 0-2: the floor-objective ablation's best arm (fine-tuned, 1,024 events, whole
# noise windows) plus near-activity sampling -- seeds 0 and 1, and with a presence head.
# Node 3: every existing checkpoint re-scored with --near/--annot, then the new arms.
set -uo pipefail
J=$1
H=$(cd "$(dirname "$0")/../.." && pwd); W=${HELIX_STUDY_DIR:-/pscratch/sd/o/oalter/helix_work}
A=$W/probe/artifact_nz_pw16_s1
nodes=($(scontrol show hostnames $(squeue -j $J -h -o %N)))
echo "[near] $J on ${nodes[*]} at $(date)"
common=(--arch-from $A --init $A --n-events 1024 --steps 4000 --warmup 300 --lr 3e-3 --lr-head 1e-2
        --val-every 250 --ckpt-every 1000 --val-events 64 --cov-per-sig 2 --win-per-event 48
        --near-any-per-sig 1 --win-near-per-event 24)
$H/scripts/study/denoise_node.sh $J ${nodes[0]} nr_ft1024 29901 "${common[@]}" &
$H/scripts/study/denoise_node.sh $J ${nodes[1]} nr_ft1024_s1 29902 "${common[@]}" --seed 1 &
$H/scripts/study/denoise_node.sh $J ${nodes[2]} nr_ft1024_pres 29903 "${common[@]}" --presence &

ev() {   # ev <gpu> <tag>... : score best.pt (final.pt for full runs) on the near evaluation
  local g=$1; shift
  for t in "$@"; do
    ck=$W/denoise/$t/best.pt; [ -f $ck ] || ck=$W/denoise/$t/final.pt
    srun --jobid=$J -N1 -n1 -w ${nodes[3]} --overlap --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE \
      CUDA_VISIBLE_DEVICES=$g $H/scripts/helix_run.sh python $H/scripts/eval_denoise.py --checkpoint $ck \
      --truth $W/resolution/truth_v2 --near $W/resolution/truth_v2_near --annot $W/denoise/noise_vs_hits.npz \
      --tag $t --out $W/denoise/results/dn_near.jsonl > $W/denoise/results/evalnear_$t.log 2>&1 \
      && echo "[near] scored $t" || echo "[near] FAILED $t"
  done
}
ev 0 ab_ft1024_win le_ft_N1024 &
ev 1 ab_ft1024_seed1 ab_ft1024_winpres &
ev 2 ab_ft1024_pres M1b_scratch_full &
ev 3 le_ft_N64 le_scratch_N1024 M2w_scratch_full_win &
wait
echo "[near] arms trained and existing runs scored at $(date)"
ev 0 nr_ft1024 & ev 1 nr_ft1024_s1 & ev 2 nr_ft1024_pres & wait
echo "[near] all done at $(date)"
