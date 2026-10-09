#!/bin/bash
# Output-model ablation on the fast protocol (fine-tuned, 1,024 labels, 4k steps,
# near-activity recipe, presence head), one 4-node allocation:
#   denoise_head.sh <jobid>
#   node 0: hurdle loss (charge regressed on charged cells only)        hd_cond
#   node 1: cell-resolution decoder                                     hd_dec
#   node 2: decoder + hurdle                                            hd_dec_cond
#   node 3: decoder + hurdle, seed 1                                    hd_dec_cond_s1
# then every arm, and the baseline (le3_ft_N1024, the same recipe with the MLP
# head), scored with --near/--annot.
set -uo pipefail
J=$1
H=$(cd "$(dirname "$0")/../.." && pwd); W=${HELIX_STUDY_DIR:-/pscratch/sd/o/oalter/helix_work}
A=$W/probe/artifact_nz_pw16_s1
# /pscratch OST 61 hangs (2026-10-09): train through overlays that route its files to CFS copies /
# exact regenerations, and write every new output to CFS (a new file could land on the hung OST)
O=/global/cfs/cdirs/m5238/users/oalter/ost61
R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; mkdir -p $R/denoise/results
export DENOISE_TRUTH=$O/denoise_truth
nodes=($(scontrol show hostnames $(squeue -j $J -h -o %N)))
echo "[head] $J on ${nodes[*]} at $(date)"
common=(--corpus-root $O/corpus --arch-from $A --init $A --n-events 1024 --steps 4000 --warmup 300 --lr 3e-3 --lr-head 1e-2
        --val-every 250 --ckpt-every 1000 --val-events 64 --cov-per-sig 2 --win-per-event 48
        --near-any-per-sig 1 --win-near-per-event 24 --presence --workers 8)
HELIX_STUDY_DIR=$R $H/scripts/study/denoise_node.sh $J ${nodes[0]} hd_cond 29981 "${common[@]}" --cond-charge &
HELIX_STUDY_DIR=$R $H/scripts/study/denoise_node.sh $J ${nodes[1]} hd_dec 29982 "${common[@]}" --head decoder &
HELIX_STUDY_DIR=$R $H/scripts/study/denoise_node.sh $J ${nodes[2]} hd_dec_cond 29983 "${common[@]}" --head decoder --cond-charge &
HELIX_STUDY_DIR=$R $H/scripts/study/denoise_node.sh $J ${nodes[3]} hd_dec_cond_s1 29984 "${common[@]}" --head decoder --cond-charge --seed 1 &
wait
echo "[head] arms trained at $(date)"

ev() {   # ev <node> <gpu> <tag>  (new arms on CFS; the baseline on /pscratch)
  local n=$1 g=$2 t=$3 ck=$R/denoise/$3/best.pt
  [ -f $ck ] || ck=$W/denoise/$3/best.pt
  srun --jobid=$J -N1 -n1 -w $n --overlap --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE \
    CUDA_VISIBLE_DEVICES=$g $H/scripts/helix_run.sh python $H/scripts/eval_denoise.py --checkpoint $ck --q0 2043.2578108909702 \
    --truth $W/resolution/truth_v2 --near $W/resolution/truth_v2_near --annot $W/denoise/noise_vs_hits.npz \
    --tag $t --out $R/denoise/results/dn_head.jsonl > $R/denoise/results/evalhead_$t.log 2>&1 \
    && echo "[head] scored $t" || echo "[head] FAILED $t"
}
ev ${nodes[0]} 0 hd_cond & ev ${nodes[1]} 0 hd_dec & ev ${nodes[2]} 0 hd_dec_cond &
ev ${nodes[3]} 0 hd_dec_cond_s1 & ev ${nodes[0]} 1 le3_ft_N1024 &
wait
echo "[head] all done at $(date)"
