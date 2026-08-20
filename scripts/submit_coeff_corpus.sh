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
#   100 jobs/run       : 100 files x 200 events = 20,000 events. Eight such
#                        submissions (RUN_INDEX 0..7) give the 160,000-event
#                        corpus; they cannot be one array here (see --array).
#
#SBATCH --job-name=coeff_corpus
#SBATCH --output=logs/coeff_%A_%a.out
#SBATCH --error=logs/coeff_%A_%a.out
# 0-99, ONE RUN PER SUBMISSION — not 0-799. This cluster's MaxArraySize is 100
# (`scontrol show config`), so an 0-799 array is rejected outright with
# "Invalid job array specification"; it stood here unsubmittable, and run
#_0027575715 must have been built some other way. Select the run with
# RUN_INDEX (its 0-based position in RUNS.txt):
#
#   RUN_INDEX=1 sbatch --export=ALL,RUN_INDEX,CONTAINER=... scripts/submit_coeff_corpus.sh
#
#SBATCH --array=0-99%32
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
# PIN THE GPU ARCHITECTURE. The DSP is architecture-sensitive: float reduction
# order differs between GPU generations, which flips a small number of gate and
# threshold decisions. Measured on sim_wire_sensor_0000.h5, identical code and
# inputs: 2080 Ti gives 58,411,720 surviving coefficients, A100 gives
# 58,421,269 — 0.016% apart, and NOT bit-comparable. Both are individually
# deterministic (two A100 rebuilds are bit-identical).
#
# Without this line an 800-job array takes whatever the pool offers, so one
# corpus could be built across several architectures and nothing on disk would
# say so. run_0027575715 was built entirely on turing and reproduces
# bit-for-bit there, which is the only reason that corpus is coherent.
#SBATCH --partition=turing
set -euo pipefail

# Default to THIS checkout, resolved from the script's own location. It used to
# name /sdf/group/neutrino/omara/helix-consolidate, which exists — a divergent
# branch 32 commits behind, missing the MAD median fix, the packaged noise
# spectrum, the m113 anchoring and the whole pimm integration. A default-invoked
# corpus build silently used superseded DSP, and nothing in the output said so.
# Resolving the checkout: HELIX_ROOT, else the submission directory, else this
# script's own location — in that order, and the order is the whole point.
#
# ${BASH_SOURCE[0]} is RIGHT under `srun bash scripts/...` and WRONG under
# `sbatch`, which copies the script to /var/spool/slurmd/scripts/ and runs the
# copy. `cd "$H"` then landed on /var/spool and every task died with
#   can't open file '/var/spool/slurmd/scripts/build_coeff_corpus.py'
# after burning its allocation. An interactive test cannot catch this: the two
# launchers disagree about what BASH_SOURCE means. (The same trap as the pimm
# configs' __file__, documented in coeff_fm_train.py for the same reason.)
#
# SLURM_SUBMIT_DIR is where `sbatch` was invoked, which for this script is the
# checkout. It is only trusted if it actually looks like one.
if [ -n "${HELIX_ROOT:-}" ]; then
  H=$HELIX_ROOT
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/build_coeff_corpus.py" ]; then
  H=$SLURM_SUBMIT_DIR
else
  H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
# Checked HERE, not left to fail 100 times inside the array. The builder is the
# one file every task needs; if it is not under $H, nothing downstream can work.
[ -f "$H/scripts/build_coeff_corpus.py" ] || {
  echo "FATAL: no scripts/build_coeff_corpus.py under H=$H."
  echo "       Set HELIX_ROOT to the checkout, or sbatch from it."
  exit 1; }
SRC=${SRC_ROOT:-/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor}
# The R1 root, matching the DEFAULT gate below. It used to name .../coeff_tpc,
# which is the PRE-R1 corpus: `--tau` unset means DetectorConfig's 0.05, so the
# ordinary default invocation built r1-gated shards and wrote them into the tree
# whose shards were built with the legacy magnitude-only rule. Nothing in the
# output said so, and the reader's cross-shard check passed — the two corpora
# share band_lengths, gids, n_wires, norm_sigma and sigma_norm exactly, and
# differ only in basis_digest and removal_json, which it did not compare (fixed
# in pimm_data/readers/coeff_tpc.py). Set OUT_ROOT explicitly to extend the old
# corpus, and pass `--tau none` when you do.
OUT=${OUT_ROOT:-/sdf/data/neutrino/omara/coeff_tpc_r1}
NORM=${NORM_SIGMA:-$OUT/_calib/norm_sigma_global.npy}
EVENTS_PER_SHARD=${EVENTS_PER_SHARD:-200}   # = one source file
SHARDS_PER_RUN=${SHARDS_PER_RUN:-100}      # = files per run
RUNS_FILE=${RUNS_FILE:-$OUT/_calib/RUNS.txt}
# Must match the kgate the frozen norm_sigma was calibrated at (see
# calibrate_norm_sigma.sh). Empty -> DetectorConfig's default of 3.0, which is
# what run_0027575715 was built with and which leaves block-wide coherent strips.
KGATE=${KGATE:-}
KG_ARG=""; [ -n "$KGATE" ] && KG_ARG="--kgate $KGATE"
# The build needs torch/pywt/h5py, which the bare login/compute python does not
# have. Set CONTAINER to run each task inside an image instead; unset keeps the
# original bare-metal behaviour.
CONTAINER=${CONTAINER:-}
if [ -n "$CONTAINER" ]; then
  PY=(singularity exec --nv -B /sdf,/lscratch "$CONTAINER" python3)
  export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$H:${PIMM_DATA_SRC:-/sdf/group/neutrino/omara/pimm-data/src}"
  export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
else
  PY=(python)
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false     # else JAX grabs 8.4 GB and torch OOMs

[ -s "$NORM" ] || { echo "FATAL: frozen norm_sigma missing at $NORM — run phase 1 first"; exit 1; }
# Named explicitly rather than left to fail inside `mapfile`, which would print
# a bare "no such file" naming neither the corpus nor the phase that writes it.
# Each corpus root carries its OWN _calib: norm_sigma is computed from the GATED
# coefficients, so coeff_tpc and coeff_tpc_r1 have genuinely different tables and
# borrowing one for the other silently mis-normalises the whole corpus.
[ -s "$RUNS_FILE" ] || { echo "FATAL: run list missing at $RUNS_FILE (corpus root $OUT) — run phase 1 first"; exit 1; }

mapfile -t RUNS < <(tr ' ' '\n' < "$RUNS_FILE" | grep -v '^$')
# RUN_INDEX picks the run; the array index picks the file within it. Kept as two
# numbers rather than one flat index because MaxArraySize=100 makes the flat form
# unsubmittable past run 0 — and because "build run 3" is the operation people
# actually want, so it should be the thing they type.
RUN_INDEX=${RUN_INDEX:-0}
IDX=$(( RUN_INDEX * SHARDS_PER_RUN + ${SLURM_ARRAY_TASK_ID:-0} ))
RI=$(( IDX / SHARDS_PER_RUN ))
[ "$RI" -lt "${#RUNS[@]}" ] || {
  echo "FATAL: RUN_INDEX=$RUN_INDEX resolves to run slot $RI but $RUNS_FILE lists ${#RUNS[@]} runs"; exit 1; }
RUN=${RUNS[$RI]}
K=$(( IDX % SHARDS_PER_RUN ))
SHARD=$(printf "%04d" "$K")

echo "task $IDX -> run=$RUN source file $SHARD (all $EVENTS_PER_SHARD events)"

cd "$H"
"${PY[@]}" scripts/build_coeff_corpus.py \
  --shard "$SRC/$RUN/sim_wire_sensor_$SHARD.h5" \
  --out   "$OUT/$RUN" \
  --dataset-name sim_wire --run "$RUN" \
  --file-index "$K" --event-start 0 --events "$EVENTS_PER_SHARD" \
  --mode serial --backend torch \
  $KG_ARG \
  --norm-sigma "$NORM"

# --file-index MUST match --shard's numeric suffix: serial mode derives both the
# output filename and /ident/source_file from --file-index, so a mismatch
# mislabels provenance AND overwrites another shard's output. The builder refuses
# unless they agree (--allow-index-mismatch overrides, deliberately).
