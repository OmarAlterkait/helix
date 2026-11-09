#!/bin/bash
# A corpus from one source through helix.build (`python -m helix.build SOURCE`: sim, sbnd, goop, a
# module:Class, or the path of any script ending in helix.build.main(Source)), on the site's scheduler (helix/sites/<site>.yaml,
# scheduler.corpus) and in its container (scripts/helix_run.sh).
#
#   scripts/corpus.sh plan SOURCE OUT RUN_DIR...  -> OUT/_calib/CHUNKS.txt (one task per line) and RUNS.txt;
#                                                 sim: RUN_DIR is a campaign group, e.g. $HELIX_SIM_ROOT/j00*
#   scripts/corpus.sh calibrate SOURCE OUT        one job: a norm_sigma table per run (CAL_EVENTS spread over
#                                                 the run), their mean -> OUT/_calib/norm_sigma_global.npy
#   scripts/corpus.sh build SOURCE OUT            job arrays over CHUNKS.txt, after the calibration job when one
#                                                 is pending (CAL_JOB); a task whose shards exist does nothing,
#                                                 so resubmitting `build` redoes only what is missing
#   scripts/corpus.sh all SOURCE OUT RUN_DIR...   the three in order
#
# Every shard of a corpus embeds the one global table, so the calibration must finish before the build.
# One GPU type per corpus: the DSP's float reductions differ between generations (turing and A100
# disagree on 0.016% of kept coefficients), so the site pins the corpus partition.
#
# Environment: EVENTS_PER_TASK (500; sim: one sensor file per task), EVENTS_PER_SHARD (sim 200, else 100;
# sim: a divisor of the 1000 events of a task, since shard k holds campaign events k*EPS..(k+1)*EPS-1),
# CAL_EVENTS (100 per run), CONFIG (chain overrides, json/yaml), BUILD_ARGS (more build options for every
# shard, e.g. --no-optical), CONC (16 tasks at once per array),
# ARRAY_MAX (100: the largest array the scheduler accepts), LOGS (OUT/_logs).
set -euo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$H/scripts/helix_env.sh"
CMD=${1:?usage: corpus.sh plan|calibrate|build|all|task SOURCE OUT [RUN_DIR...]}
SOURCE=${2:?source: sim, sbnd, goop, module:Class or a build script}
OUT=${3:?output corpus root}
shift 3
CAL=$OUT/_calib
SRC=$SOURCE                 # what the jobs this submits are told, to rebuild BUILD from
if [ -f "$SOURCE" ]; then SRC=$(realpath "$SOURCE"); BUILD=("$SRC"); SOURCE=$(basename "$SOURCE" .py); SOURCE=${SOURCE#build_}
else BUILD=(-m helix.build "$SOURCE"); SOURCE=${SOURCE##*:}; fi
LOGS=${LOGS:-$OUT/_logs}
[ "$SOURCE" = sim ] && EPS=${EVENTS_PER_SHARD:-200} || EPS=${EVENTS_PER_SHARD:-100}
RUN=("$H/scripts/helix_run.sh" python)
CFG=(); [ -n "${CONFIG:-}" ] && CFG=(--config "$CONFIG")
SB=(--account="$HELIX_SLURM_CORPUS_ACCOUNT" --gpus=1)
[ -n "${HELIX_SLURM_CORPUS_PARTITION:-}" ] && SB+=(--partition="$HELIX_SLURM_CORPUS_PARTITION")
[ -n "${HELIX_SLURM_CORPUS_QOS:-}" ] && SB+=(--qos="$HELIX_SLURM_CORPUS_QOS")
[ -n "${HELIX_SLURM_CORPUS_CONSTRAINT:-}" ] && SB+=(--constraint="$HELIX_SLURM_CORPUS_CONSTRAINT")
read -r -a EXTRA <<< "${HELIX_SLURM_CORPUS_EXTRA:-}"
SB+=("${EXTRA[@]}")
export HELIX_INTERACTIVE=0 OMP_NUM_THREADS=2 HELIX_FORWARD_ENV="OMP_NUM_THREADS"

plan() {
  mkdir -p "$CAL"
  "${RUN[@]}" "${BUILD[@]}" plan "$OUT" "$@" --events-per-task "${EVENTS_PER_TASK:-500}" \
      --events-per-shard "$EPS" > "$CAL/CHUNKS.txt"
  for d in "$@"; do basename "$d"; done > "$CAL/RUNS.txt"
  echo "$(wc -l < "$CAL/CHUNKS.txt") tasks over $# runs -> $CAL/CHUNKS.txt"
}

calibrate_job() {          # inside the job; a run's inputs are those its tasks in CHUNKS.txt read
  for r in $(cat "$CAL/RUNS.txt"); do
    [ -s "$CAL/$r.npy" ] && { echo "$r calibrated"; continue; }
    mapfile -t inputs < <(awk -v r="$r" '{ n = split($1, a, "/") } a[n - 1] == r { for (i = 6; i <= NF; i++) print $i }' \
        "$CAL/CHUNKS.txt" | sort -u)
    "${RUN[@]}" "${BUILD[@]}" --input "${inputs[@]}" --spread "${CAL_EVENTS:-100}" --device cuda \
        --no-optical "${CFG[@]}" --calibrate "$CAL/$r.npy" | tail -3
  done
  "${RUN[@]}" - "$CAL" <<'PY'
import os, sys
import numpy as np
cal = sys.argv[1]
runs = open(os.path.join(cal, "RUNS.txt")).read().split()
A = np.stack([np.load(os.path.join(cal, f"{r}.npy")) for r in runs]); g = A.mean(0).astype(np.float32)
np.save(os.path.join(cal, "norm_sigma_global.npy"), g)
rel = np.abs(A - g) / np.maximum(g, 1e-9)
print(f"{len(runs)} runs, table {g.shape}; run-to-run deviation median {100 * np.median(rel):.2f}% max {100 * rel.max():.2f}%")
PY
}

calibrate() {
  mkdir -p "$LOGS"
  CAL_JOB=$(sbatch --parsable "${SB[@]}" --cpus-per-task=8 --mem=48G --time=02:00:00 --job-name=calib_$SOURCE \
      --output="$LOGS/calib_${SOURCE}_%j.out" --wrap="bash $H/scripts/corpus.sh calibrate_job $SRC $OUT")
  echo "calibration job $CAL_JOB"
}

task() {                   # inside an array task: line OFFSET + SLURM_ARRAY_TASK_ID of CHUNKS.txt
  read -r prefix first count eps shard inputs < <(sed -n "$((${OFFSET:-0} + SLURM_ARRAY_TASK_ID + 1))p" "$CAL/CHUNKS.txt")
  mkdir -p "$(dirname "$prefix")"
  echo "$prefix: events $first..$((first + count - 1)) -> shards from $shard"
  # shellcheck disable=SC2086
  "${RUN[@]}" "${BUILD[@]}" --input $inputs --output "$prefix" --device cuda --first "$first" \
      --count "$count" --events-per-shard "$eps" --shard-start "$shard" --norm-sigma "$CAL/norm_sigma_global.npy" \
      --skip-complete "${CFG[@]}" ${BUILD_ARGS:-} > "$prefix.summary_$(printf %04d "$shard").json"
  tail -c 400 "$prefix.summary_$(printf %04d "$shard").json"; echo
}

build() {
  [ -s "$CAL/norm_sigma_global.npy" ] || [ -n "${CAL_JOB:-}" ] || { echo "no $CAL/norm_sigma_global.npy: calibrate first"; exit 1; }
  mkdir -p "$LOGS"
  n=$(wc -l < "$CAL/CHUNKS.txt"); max=${ARRAY_MAX:-100}
  for ((o = 0; o < n; o += max)); do
    last=$(( n - o < max ? n - o - 1 : max - 1 ))
    sbatch --parsable "${SB[@]}" ${CAL_JOB:+--dependency=afterok:$CAL_JOB} --cpus-per-task=8 --mem=48G \
        --time=02:00:00 --array="0-$last%${CONC:-16}" --job-name="corpus_$SOURCE" \
        --output="$LOGS/corpus_${SOURCE}_%A_%a.out" --export=ALL,OFFSET=$o \
        --wrap="bash $H/scripts/corpus.sh task $SRC $OUT"
  done
}

case "$CMD" in
  plan) plan "$@" ;;
  calibrate) calibrate ;;
  calibrate_job) calibrate_job ;;
  build) build ;;
  task) task ;;
  all) plan "$@"; calibrate; build ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
