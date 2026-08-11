#!/bin/bash
# Submit the coeff-corpus build as a SLURM job array: ~160k events over 8 runs.
#
# TWO-PHASE BY NECESSITY. Every shard of the corpus must embed the SAME
# norm_sigma — the reader refuses to open a corpus whose shards disagree — so the
# frozen table has to exist before any build job starts. A naively parallel array
# would race the bootstrap: jobs reading a half-written .npy fail with
# "EOF reading array header". Phase 1 is therefore run once, by hand, and its
# output is an INPUT to this script.
#
#   Phase 1 (once, interactive):  scripts/calibrate_norm_sigma.sh
#   Phase 2 (this script):        sbatch scripts/submit_coeff_corpus.sh
#   Phase 3 (after, per run):     python -m pimm_data.coeff_verify <run dir> \
#                                     --dataset-name sim_wire --expect <N>
#
# Sizing rationale:
#   SERIAL mode, one job per SOURCE FILE. Loader mode globs the whole run and
#   indexes all 100 shard files before yielding an event: 6433 ms/file on a COLD
#   run (~11 min/job), against 4 ms/file warm. Every job would pay it again --
#   ~57 h of redundant indexing across 8 runs, more than the real work. Serial
#   opens only its own file. (Loader was chosen earlier because serial reused
#   noise seeds across shards; that is fixed -- the seed now folds in run+shard.)
#
#   200 events/job     : one source file. Build memory is ~4x the shard's on-disk
#                        size -- build_corpus_stream accumulates every CoeffEvent,
#                        the writer concatenates a copy, audit_shard reads it back.
#                        1000 events needed ~19 GB and was OOM-killed at 16.
#   --mem 12G          : ~5 GB measured at 250 events, linear, plus headroom.
#   ~2.2 min/job       : ~2 s warmup + 200 x 0.66 s. TORCH, not jax: torch has
#                        essentially no warmup where jax pays ~35 s of JIT per
#                        process, so at 200 events/job torch wins outright
#                        (134 s vs 149 s). jax leads only in steady state, by
#                        1.16x, which does not repay its warmup until ~361
#                        events. Pass --backend jax to override.
#   800 jobs           : 8 runs x 100 files x 200 events = 160,000 events.
#
#SBATCH --job-name=coeff_corpus
#SBATCH --output=logs/coeff_%A_%a.out
#SBATCH --error=logs/coeff_%A_%a.out
#SBATCH --array=0-799%32
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
set -euo pipefail

# Default to THIS checkout, resolved from the script's own location. It used to
# name /sdf/group/neutrino/omara/helix-consolidate, which exists — a divergent
# branch 32 commits behind, missing the MAD median fix, the packaged noise
# spectrum, the m113 anchoring and the whole pimm integration. A default-invoked
# corpus build silently used superseded DSP, and nothing in the output said so.
H=${HELIX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SRC=${SRC_ROOT:-/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor}
OUT=${OUT_ROOT:-/sdf/data/neutrino/omara/coeff_tpc}
NORM=${NORM_SIGMA:-$OUT/_calib/norm_sigma_global.npy}
EVENTS_PER_SHARD=${EVENTS_PER_SHARD:-200}   # = one source file
SHARDS_PER_RUN=${SHARDS_PER_RUN:-100}      # = files per run
RUNS_FILE=${RUNS_FILE:-$OUT/_calib/RUNS.txt}

export XLA_PYTHON_CLIENT_PREALLOCATE=false     # else JAX grabs 8.4 GB and torch OOMs

[ -s "$NORM" ] || { echo "FATAL: frozen norm_sigma missing at $NORM — run phase 1 first"; exit 1; }

mapfile -t RUNS < <(tr ' ' '\n' < "$RUNS_FILE" | grep -v '^$')
IDX=${SLURM_ARRAY_TASK_ID:-0}
RUN=${RUNS[$(( IDX / SHARDS_PER_RUN ))]}
K=$(( IDX % SHARDS_PER_RUN ))
SHARD=$(printf "%04d" "$K")

echo "task $IDX -> run=$RUN source file $SHARD (all $EVENTS_PER_SHARD events)"

cd "$H"
python scripts/build_coeff_corpus.py \
  --shard "$SRC/$RUN/sim_wire_sensor_$SHARD.h5" \
  --out   "$OUT/$RUN" \
  --dataset-name sim_wire --run "$RUN" \
  --file-index "$K" --event-start 0 --events "$EVENTS_PER_SHARD" \
  --mode serial --backend torch \
  --norm-sigma "$NORM"

# --file-index MUST match --shard's numeric suffix: serial mode derives both the
# output filename and /ident/source_file from --file-index, so a mismatch
# mislabels provenance AND overwrites another shard's output. The builder refuses
# unless they agree (--allow-index-mismatch overrides, deliberately).
