#!/bin/bash
# Support-threshold arms (complete-truth charge targets, presence and sampling support at c e-), fast
# protocol (fine-tuned, 1,024 labels, 4k steps, near-activity recipe, presence, decoder), one 4-node allocation:
#   denoise_support.sh <jobid>   -- sp_dec_c25, sp_dec_cond_c25 (c = 25 e-), sp_dec_c102, sp_dec_cond_c102 (c = 102 e-)
# compared with the label arms lb_* (hits truth; complete truth at c = 0); all scored with 5 shared noise draws,
# complete-truth keys and near-activity windows. /pscratch OST 61 hangs: overlay inputs, CFS outputs.
set -uo pipefail
J=$1
H=$(cd "$(dirname "$0")/../.." && pwd); W=/pscratch/sd/o/oalter/helix_work; A=$W/probe/artifact_nz_pw16_s1
O=/global/cfs/cdirs/m5238/users/oalter/ost61; R=/global/cfs/cdirs/m5238/users/oalter/helix_runs
CT=/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete
mkdir -p $R/denoise/results
nodes=($(scontrol show hostnames $(squeue -j $J -h -o %N)))
echo "[support] $J on ${nodes[*]} at $(date)"
common=(--corpus-root $O/corpus --arch-from $A --init $A --n-events 1024 --steps 4000 --warmup 300 --lr 3e-3 --lr-head 1e-2
        --val-every 250 --ckpt-every 1000 --val-events 64 --cov-per-sig 2 --win-per-event 48
        --near-any-per-sig 1 --win-near-per-event 24 --presence --workers 8 --head decoder)
run() {  # run <node> <tag> <port> <truth root> [extra...]
  local n=$1 t=$2 p=$3 tr=$4; shift 4
  HELIX_STUDY_DIR=$R DENOISE_TRUTH=$tr $H/scripts/study/denoise_node.sh $J $n $t $p "${common[@]}" "$@"
}
run ${nodes[0]} sp_dec_c25 30061 $CT --support-min-q 25 &
run ${nodes[1]} sp_dec_cond_c25 30062 $CT --support-min-q 25 --cond-charge &
# complete truth already built

run ${nodes[2]} sp_dec_c102 30063 $CT --support-min-q 102 &
run ${nodes[3]} sp_dec_cond_c102 30064 $CT --support-min-q 102 --cond-charge &
wait
echo "[support] arms trained at $(date)"
ev() {   # ev <node> <tag>
  local n=$1 t=$2
  srun --jobid=$J -N1 -n1 -w $n --overlap --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE \
    CUDA_VISIBLE_DEVICES=0 $H/scripts/helix_run.sh python $H/scripts/eval_denoise.py --checkpoint $R/denoise/$t/best.pt \
    --q0 2043.2578108909702 --truth $W/resolution/truth_v2 --near $W/resolution/truth_v2_near \
    --annot $W/denoise/noise_vs_hits.npz --draws 5 --complete-truth $CT \
    --tag $t --out $R/denoise/results/dn_support.jsonl > $R/denoise/results/evalsupport_$t.log 2>&1 \
    && echo "[support] scored $t" || echo "[support] FAILED $t"
}
ev ${nodes[0]} sp_dec_c25 & ev ${nodes[1]} sp_dec_cond_c25 & ev ${nodes[2]} sp_dec_c102 & ev ${nodes[3]} sp_dec_cond_c102 &
wait
echo "[support] all done at $(date)"
