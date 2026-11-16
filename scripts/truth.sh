#!/bin/bash
# The cells truth of a whole corpus run on a cell grid (scripts/build_truth.py), as a CPU job array.
#
#   scripts/truth.sh submit GRID OUT [CORPUS]     e.g.  scripts/truth.sh submit 1x4 /sdf/data/neutrino/omara/truth_1x4_d1m/cells
#   scripts/truth.sh task   GRID OUT CORPUS       (inside an array task: its block of shards)
#
# It reads the campaign's hits / step / labl at HELIX_SIM_ROOT, so it runs only at a site that
# has them: S3DF (/sdf/data/neutrino/omara/doraemon_2026_1M_lartpc/data/jaxtpc_wire) and NERSC
# (j000000 only, $SCRATCH/sim_d1m; helix/sites/nersc.yaml). CORPUS defaults to HELIX_CORPUS (the coeff shards whose
# /ident names the events). The build is numpy only: no GPU.
#
# Each task builds SHARDS_PER_TASK shards with WORKERS processes (one shard each); a shard whose
# file exists on GRID is skipped, so resubmitting redoes only what is missing. Commit first: the
# provenance records the helix commit and refuses nothing, but a dirty tree is recorded as such.
#
# To train or evaluate on it, give the dataset a root holding both modalities:
#   <root>/coeff -> the corpus run,  <root>/cells = OUT     (CoeffTPCDataset(data_root=<root>))
#
# Environment: SBATCH_ARGS (the CPU partition's account / partition / qos -- required, since the
# site profiles declare GPU job kinds only; at S3DF the 1 x 1 finetruth ran on milano),
# SHARDS_PER_TASK (20), WORKERS (8), MEM (64G), TIME (04:00:00), CONC (20 tasks at once),
# LOGS (OUT/../_logs).
set -euo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$H/scripts/helix_env.sh"
CMD=${1:?usage: truth.sh submit|task GRID OUT [CORPUS]}
GRID=${2:?grid, WIRESxTICKS (e.g. 1x4)}
OUT=${3:?output directory}
CORPUS=${4:-${HELIX_CORPUS:?no CORPUS and no HELIX_CORPUS}}
K=${SHARDS_PER_TASK:-20}
WORKERS=${WORKERS:-8}
export HELIX_INTERACTIVE=0 OMP_NUM_THREADS=1 HELIX_FORWARD_ENV="OMP_NUM_THREADS"

case "$CMD" in
  submit)
    [ -n "${SBATCH_ARGS:-}" ] || { echo "set SBATCH_ARGS to a CPU partition's --account/--partition (and --qos)"; exit 2; }
    [ -n "${HELIX_SIM_ROOT:-}" ] && [ -d "$HELIX_SIM_ROOT" ] || { echo "no HELIX_SIM_ROOT at site ${HELIX_SITE:-?}: the hits are not here"; exit 2; }
    last=$(ls "$CORPUS" | sed -n 's/.*_coeff_\([0-9]*\)\.h5$/\1/p' | sort -n | tail -1)
    [ -n "$last" ] || { echo "$CORPUS: no *_coeff_<k>.h5"; exit 2; }
    n=$(( 10#$last / K + 1 ))
    LOGS=${LOGS:-$(dirname "$OUT")/_logs}
    mkdir -p "$OUT" "$LOGS"
    # shellcheck disable=SC2086
    sbatch --parsable $SBATCH_ARGS --cpus-per-task="$WORKERS" --mem="${MEM:-64G}" --time="${TIME:-04:00:00}" \
        --array="0-$((n - 1))%${CONC:-20}" --job-name="truth_$GRID" --output="$LOGS/truth_${GRID}_%A_%a.out" \
        --wrap="bash $H/scripts/truth.sh task $GRID $OUT $CORPUS"
    echo "$n tasks of $K shards (0..$((10#$last))) -> $OUT"
    ;;
  task)
    a=$(( SLURM_ARRAY_TASK_ID * K )); b=$(( a + K - 1 ))
    "$H/scripts/helix_run.sh" python "$H/scripts/build_truth.py" --corpus "$CORPUS" --grid "$GRID" --out "$OUT" \
        --shards "$a-$b" --workers "$WORKERS"
    ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
