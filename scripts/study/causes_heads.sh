#!/bin/bash
# Floor causes, round 2 (m7_tests_results.md, "Independent audit"): one 2-node interactive allocation, resumable.
#   node A: head probes H1-H4    node B: C14 overflow bins (CPU), T13 learned linear filter, head probes H5-H6
# Head probes (t11_head_probe.py, 10k steps, each ending with the head-only convergence test):
#   H1 nofeat_raw          no encoder; raw arcsinh 3x3 slots                 (learnability from the log input)
#   H2 nofeat_amp          no encoder; + linear amplitudes of the 3x3 slots   (input encoding)
#   H3 m7_nbramp           frozen M7dc encoder + 3x3 slots + amplitudes      (the audit's missing control)
#   H4 m7_bright           frozen M7dc encoder, M7dc-shape head, loss on truth >= 16 k only   (loss share)
#   H5 pre_nbramp          frozen PRETRAINED FM + 3x3 slots + amplitudes     (does the FM add to the raw inputs?)
#   H6 nofeat_amp_bright   H2 with the loss on truth >= 16 k only
set -uo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd); M=/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7tests; O=$M/t14
mkdir -p $O; cd $H
declare -A ARGS=(
  [nofeat_raw]="--head fresh --no-feat --nbr-raw"
  [nofeat_amp]="--head fresh --no-feat --nbr-raw --nbr-amp"
  [m7_nbramp]="--head fresh --nbr-raw --nbr-amp"
  [m7_bright]="--head fresh --min-q 16000"
  [pre_nbramp]="--head fresh --encoder pretrained --nbr-raw --nbr-amp"
  [nofeat_amp_bright]="--head fresh --no-feat --nbr-raw --nbr-amp --min-q 16000" )
A_TAGS=(nofeat_raw nofeat_amp m7_nbramp m7_bright); B_TAGS=(pre_nbramp nofeat_amp_bright)
done_() { python3 -c "import json,sys; sys.exit(0 if 'converge' in json.load(open('$O/$1.json')) else 1)" 2>/dev/null; }
alldone() { for t in "${A_TAGS[@]}" "${B_TAGS[@]}"; do done_ $t || return 1; done; [ -f $M/t13.json ] && [ -f $M/c14.json ]; }
E=(env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE)
while ! alldone; do
  J=""; until [ -n "$J" ]; do
    J=$(timeout 21600 salloc --no-shell --qos=interactive --account=m5238_g --constraint="gpu&hbm80g" --nodes=2 --gpus-per-node=4 \
        --time=04:00:00 --job-name=hx-heads 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
    [ -z "$J" ] && sleep 30
  done
  NODES=($(scontrol show hostnames $(squeue -j $J -h -o %N)))
  echo "[heads] got $J (${NODES[*]}) at $(date)"
  S() { local node=$1 gpu=$2; shift 2; srun --jobid=$J -N1 -n1 -w $node --overlap -c 28 --gpus-per-node=4 \
        "${E[@]}" CUDA_VISIBLE_DEVICES=$gpu "$@"; }
  run_tags() { local node=$1; shift; local i=0
    for t in "$@"; do
      S $node $i scripts/helix_run.sh python baselines/uboone_sp/causes/t11_head_probe.py --tag $t ${ARGS[$t]} \
        --steps 10000 --accum 4 --workers 6 --converge --out $O >> $O/$t.log 2>&1 &
      i=$((i + 1))
    done; }
  todo=(); for t in "${A_TAGS[@]}" "${B_TAGS[@]}"; do done_ $t || todo+=($t); done
  h=$(( (${#todo[@]} + 1) / 2 ))                                   # unfinished probes, split over the two nodes
  run_tags ${NODES[0]} "${todo[@]:0:$h}"
  run_tags ${NODES[1]} "${todo[@]:$h}"
  [ -f $M/c14.json ] || S ${NODES[1]} 3 scripts/helix_run.sh python baselines/uboone_sp/causes/c14_overflow.py --out $M/c14.json \
    > $M/c14.log 2>&1 &
  [ -f $M/t13.json ] || S ${NODES[1]} 2 env XLA_PYTHON_CLIENT_PREALLOCATE=false scripts/helix_run.sh python \
    baselines/uboone_sp/causes/t13_wiener.py --fit 0-127 --out $M/t13.json > $M/t13.log 2>&1 &
  wait
  scancel $J; echo "[heads] released $J at $(date)"; sleep 30
done
echo "[heads] all done at $(date)"
