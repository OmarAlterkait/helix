#!/bin/bash
# Label ablation on the fast protocol (fine-tuned, 1,024 labels, 4k steps, near-activity
# recipe, presence head, cell-resolution decoder), one 4-node allocation:
#   denoise_label.sh <jobid>
#   nodes 0/1: hits-truth labels      -- lb_dec_hits, lb_dec_cond_hits
#   nodes 2/3: complete-truth labels  -- lb_dec_ct,   lb_dec_cond_ct   (scripts/build_complete_corpus_truth.py;
#              started once all 790 complete-truth shards exist)
# then all four scored with 5 shared noise draws, complete-truth keys and near-activity windows.
# /pscratch OST 61 hangs: overlay inputs, CFS outputs.
set -uo pipefail
J=$1
H=$(cd "$(dirname "$0")/../.." && pwd); W=/pscratch/sd/o/oalter/helix_work; A=$W/probe/artifact_nz_pw16_s1
O=/global/cfs/cdirs/m5238/users/oalter/ost61; R=/global/cfs/cdirs/m5238/users/oalter/helix_runs
CT=/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete
mkdir -p $R/denoise/results
nodes=($(scontrol show hostnames $(squeue -j $J -h -o %N)))
echo "[label] $J on ${nodes[*]} at $(date)"
common=(--corpus-root $O/corpus --arch-from $A --init $A --n-events 1024 --steps 4000 --warmup 300 --lr 3e-3 --lr-head 1e-2
        --val-every 250 --ckpt-every 1000 --val-events 64 --cov-per-sig 2 --win-per-event 48
        --near-any-per-sig 1 --win-near-per-event 24 --presence --workers 8 --head decoder)
run() {  # run <node> <tag> <port> <truth root> [extra...]
  local n=$1 t=$2 p=$3 tr=$4; shift 4
  HELIX_STUDY_DIR=$R DENOISE_TRUTH=$tr $H/scripts/study/denoise_node.sh $J $n $t $p "${common[@]}" "$@"
}
run ${nodes[0]} lb_dec_hits 30051 $O/denoise_truth &
run ${nodes[1]} lb_dec_cond_hits 30052 $O/denoise_truth --cond-charge &
until [ $(ls $CT/run_*/truth_*.npz 2>/dev/null | grep -v tmp | wc -l) -ge 790 ]; do sleep 60; done
echo "[label] complete truth ready at $(date)"
run ${nodes[2]} lb_dec_ct 30053 $CT &
run ${nodes[3]} lb_dec_cond_ct 30054 $CT --cond-charge &
wait
echo "[label] arms trained at $(date)"
ev() {   # ev <node> <tag>
  local n=$1 t=$2
  srun --jobid=$J -N1 -n1 -w $n --overlap --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE \
    CUDA_VISIBLE_DEVICES=0 $H/scripts/helix_run.sh python $H/scripts/eval_denoise.py --checkpoint $R/denoise/$t/best.pt \
    --q0 2043.2578108909702 --truth $W/resolution/truth_v2 --near $W/resolution/truth_v2_near \
    --annot $W/denoise/noise_vs_hits.npz --draws 5 --complete-truth $CT \
    --tag $t --out $R/denoise/results/dn_label.jsonl > $R/denoise/results/evallabel_$t.log 2>&1 \
    && echo "[label] scored $t" || echo "[label] FAILED $t"
}
ev ${nodes[0]} lb_dec_hits & ev ${nodes[1]} lb_dec_cond_hits & ev ${nodes[2]} lb_dec_ct & ev ${nodes[3]} lb_dec_cond_ct &
wait
echo "[label] all done at $(date)"
