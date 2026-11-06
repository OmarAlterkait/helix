#!/bin/bash
# T11 continued (m7_tests_results.md): head probes on M7dc's frozen encoder, resumable across 1-node interactive
# allocations until every run has finished (its <tag>.json holds a 'converge' entry):
#   P5_nbr     covering + 3x3-neighbourhood raw slots, 2.5k steps (equal schedule to P2_fresh)
#   P5_long    the same, 10k steps          P4_long  larger decoder, 10k steps     P2_long  M7dc-shape, 10k steps
# Each ends with the head-only first-order convergence test (--converge).
set -uo pipefail
H=$(cd "$(dirname "$0")/../../.." && pwd); M=/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7tests/t11
cd $H
declare -A ARGS=( [P5_nbr]="--head fresh --nbr-raw --steps 2500" [P5_long]="--head fresh --nbr-raw --steps 10000"
                  [P4_long]="--head fresh --proj 256 --layers 4 --steps 10000" [P2_long]="--head fresh --steps 10000" )
TAGS=(P5_nbr P5_long P4_long P2_long)
done_() { python3 -c "import json,sys; sys.exit(0 if 'converge' in json.load(open('$M/$1.json')) else 1)" 2>/dev/null; }
while :; do
  todo=(); for t in "${TAGS[@]}"; do done_ $t || todo+=($t); done
  [ ${#todo[@]} -eq 0 ] && break
  J=""; until [ -n "$J" ]; do
    J=$(timeout 90 salloc --no-shell --qos=interactive --account=m5238_g --constraint="gpu&hbm80g" --nodes=1 --gpus-per-node=4 \
        --time=04:00:00 --job-name=hx-t11long 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
    [ -z "$J" ] && sleep 30
  done
  echo "[t11long] got $J at $(date): ${todo[*]}"
  i=0
  for t in "${todo[@]}"; do
    srun --jobid=$J -N1 -n1 --overlap -c 28 --gpus-per-node=4 env CUDA_VISIBLE_DEVICES=$i HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE \
      scripts/helix_run.sh python baselines/uboone_sp/causes/t11_head_probe.py --tag $t ${ARGS[$t]} --accum 4 --workers 6 \
      --converge --out $M >> $M/$t.log 2>&1 &
    i=$((i + 1))
  done
  wait
  scancel $J; echo "[t11long] released $J at $(date)"; sleep 30
done
echo "[t11long] all done at $(date)"
