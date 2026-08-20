#!/bin/bash
# Submit N chained links of the coefficient-FM stable phase.
#
# Each link depends on the previous with `afterany`, so it starts when that one
# ENDS — completed, preempted-past-requeue, or out of wall clock alike. Every
# link resumes from the last checkpoint, and a link that finds training already
# complete exits without claiming its GPUs, so over-submitting is cheap and
# under-submitting is what costs a night.
#
#   scripts/chain_coeff_fm_train.sh 5
#   scripts/chain_coeff_fm_train.sh 5 configs/pimm/coeff_fm_cooldown.py
#
# Add more links at any time — a new chain's first link simply resumes.
set -euo pipefail

N=${1:-4}
H=${HELIX_ROOT:-/sdf/group/neutrino/omara/helix-extraction}
CFG=${2:-$H/configs/pimm/coeff_fm_train_8run.py}
ACCT=${ACCOUNT:-mli:default}
# EXCLUDE=node[,node] for nodes known to be bad. A node whose GPUs are held
# still reports healthy to SLURM, so it keeps being offered: sdfampere010 failed
# links 2, 3, 4 and 5 in about a minute each with "CUDA-capable device(s) is/are
# busy or unavailable". The launcher's preflight now requeues instead of burning
# a link, but excluding a known-bad node saves the round trip.
EXC=${EXCLUDE:-}
LOGS=${LOGDIR:-/sdf/data/neutrino/omara/exp/_diag/trainlogs}
mkdir -p "$LOGS"

case "$CFG" in /*) ;; *) CFG="$H/$CFG" ;; esac
[ -f "$CFG" ] || { echo "FATAL: no config at $CFG"; exit 1; }

PREV=""
for i in $(seq 1 "$N"); do
  DEP=""
  [ -n "$PREV" ] && DEP="--dependency=afterany:$PREV"
  JID=$(sbatch --parsable $DEP ${EXC:+--exclude=$EXC} \
      --account="$ACCT" \
      --job-name="coeff8_$i" \
      --output="$LOGS/coeff8_${i}_%j.out" \
      --error="$LOGS/coeff8_${i}_%j.out" \
      --export=ALL,HELIX_ROOT=$H,CFG=$CFG \
      "$H/scripts/submit_coeff_fm_train.sh")
  printf "link %2d/%s -> job %s%s\n" "$i" "$N" "$JID" "${PREV:+  (after $PREV)}"
  PREV=$JID
done
echo
echo "logs: $LOGS"
echo "watch: squeue -u \$USER -n $(basename "${CFG%.py}") ; tail -f $LOGS/coeff8_1_*.out"
