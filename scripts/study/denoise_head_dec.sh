#!/bin/bash
# Decoder arms of the output-model ablation (after the per-token key/value fix), on the
# nodes of an allocation already running hd_cond, then all five evaluations:
#   denoise_head_dec.sh <jobid> <node_dec> <node_dec_cond> <node_dec_cond_s1>
set -uo pipefail
J=$1; N1=$2; N2=$3; N3=$4
H=$(cd "$(dirname "$0")/../.." && pwd); W=${HELIX_STUDY_DIR:-/pscratch/sd/o/oalter/helix_work}
A=$W/probe/artifact_nz_pw16_s1
O=/global/cfs/cdirs/m5238/users/oalter/ost61          # /pscratch OST 61 hangs: overlays + CFS outputs
R=/global/cfs/cdirs/m5238/users/oalter/helix_runs; mkdir -p $R/denoise/results
export DENOISE_TRUTH=$O/denoise_truth
common=(--corpus-root $O/corpus --arch-from $A --init $A --n-events 1024 --steps 4000 --warmup 300 --lr 3e-3 --lr-head 1e-2
        --val-every 250 --ckpt-every 1000 --val-events 64 --cov-per-sig 2 --win-per-event 48
        --near-any-per-sig 1 --win-near-per-event 24 --presence --workers 8 --head decoder)
echo "[dec] $J on $N1 $N2 $N3 at $(date)"
HELIX_STUDY_DIR=$R $H/scripts/study/denoise_node.sh $J $N1 hd_dec 29991 "${common[@]}" &
HELIX_STUDY_DIR=$R $H/scripts/study/denoise_node.sh $J $N2 hd_dec_cond 29992 "${common[@]}" --cond-charge &
HELIX_STUDY_DIR=$R $H/scripts/study/denoise_node.sh $J $N3 hd_dec_cond_s1 29993 "${common[@]}" --cond-charge --seed 1 &
until [ -f $R/denoise/hd_cond/final.pt ]; do sleep 30; done                 # the MLP-head arm, already running
wait
echo "[dec] arms trained at $(date)"
ev() {   # ev <node> <gpu> <tag>
  local n=$1 g=$2 t=$3 ck=$R/denoise/$3/best.pt
  [ -f $ck ] || ck=$W/denoise/$3/best.pt
  srun --jobid=$J -N1 -n1 -w $n --overlap --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE \
    CUDA_VISIBLE_DEVICES=$g $H/scripts/helix_run.sh python $H/scripts/eval_denoise.py --checkpoint $ck --q0 2043.2578108909702 \
    --truth $W/resolution/truth_v2 --near $W/resolution/truth_v2_near --annot $W/denoise/noise_vs_hits.npz \
    --tag $t --out $R/denoise/results/dn_head.jsonl > $R/denoise/results/evalhead_$t.log 2>&1 \
    && echo "[dec] scored $t" || echo "[dec] FAILED $t"
}
ev $N1 0 hd_cond & ev $N1 1 hd_dec & ev $N2 0 hd_dec_cond & ev $N3 0 hd_dec_cond_s1 & ev $N2 1 le3_ft_N1024 &
wait
echo "[dec] all done at $(date)"
