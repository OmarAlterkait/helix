#!/bin/bash
# Phase 1: compute the ONE frozen global norm_sigma the whole corpus shares.
#
# norm_sigma is the cross-event normalisation table the tokenizer divides by.
# CoeffTPCReader refuses to open a corpus whose shards disagree on it, so it must
# be identical everywhere — which means it has to represent every run in the
# corpus, not run 0's first shard. Each run contributes an equal sample; the
# grand mean is their average.
#
# torch backend: no JIT warmup, so a short 100-event calibration pass does not
# pay jax's ~35 s compile eight times over.
#
# SERIAL mode, not loader: loader globs the whole run and indexes all 100 shard
# files before the first event. On a COLD run that measured 6433 ms/file -> ~11
# minutes of stalling per invocation (4 ms/file once warm, a 1600x difference).
# Serial opens only the file it was given.
#
# The spread BETWEEN per-run tables is also the electronics-drift measurement:
# shard-to-shard sampling noise is ~0.08%/0.99% (median/max) at 50 events, so
# run-to-run spread materially above that is real drift, not sampling.
#
#   ./calibrate_norm_sigma.sh [events_per_run]        # default 100
#
# Writes <OUT>/_calib/<run>.npy per run plus norm_sigma_global.npy, which is the
# --norm-sigma input to every build job.
set -euo pipefail

# Default to THIS checkout, resolved from the script's own location. It used to
# name /sdf/group/neutrino/omara/helix-consolidate, which exists — a divergent
# branch 32 commits behind, missing the MAD median fix, the packaged noise
# spectrum, the m113 anchoring and the whole pimm integration. A default-invoked
# corpus build silently used superseded DSP, and nothing in the output said so.
H=${HELIX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SRC=${SRC_ROOT:-/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor}
OUT=${OUT_ROOT:-/sdf/data/neutrino/omara/coeff_tpc}
CALIB=$OUT/_calib
N=${1:-100}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

[ -s "$CALIB/RUNS.txt" ] || { echo "FATAL: $CALIB/RUNS.txt missing (one run name per line or space-separated)"; exit 1; }
RUNS=$(tr '\n' ' ' < "$CALIB/RUNS.txt")

mkdir -p "$CALIB"
cd "$H"
for r in $RUNS; do
  [ -s "$CALIB/$r.npy" ] && { echo "=== $r already calibrated, skipping"; continue; }
  echo "=== calibrating $r ($N events)"
  python scripts/build_coeff_corpus.py \
    --shard "$SRC/$r/sim_wire_sensor_0000.h5" --out "$CALIB/$r" \
    --dataset-name sim_wire --run "$r" --file-index 0 \
    --event-start 0 --events "$N" --mode serial --backend torch \
    --calibrate --save-norm-sigma "$CALIB/$r.npy" 2>&1 | tail -2
done

python - "$CALIB" "$RUNS" << 'PY'
import sys, glob, os
import numpy as np
calib, runs = sys.argv[1], sys.argv[2].split()
tabs = {r: np.load(os.path.join(calib, f"{r}.npy")) for r in runs}
A = np.stack(list(tabs.values()))
g = A.mean(axis=0).astype(np.float32)
np.save(os.path.join(calib, "norm_sigma_global.npy"), g)

# run-to-run spread, as a fraction of the grand mean
rel = np.abs(A - g) / np.maximum(g, 1e-9)
print(f"\nper-run norm_sigma tables: {A.shape[0]} runs, table {g.shape}")
print(f"run-to-run deviation from the global table: "
      f"median {100*np.median(rel):.2f}%  max {100*rel.max():.2f}%")
print("  (shard-to-shard sampling noise alone is ~0.08% median / 0.99% max at 50 "
      "events, so materially larger spread here is real drift)")
worst = np.unravel_index(rel.argmax(), rel.shape)
print(f"  worst cell: run {runs[worst[0]]}  gid_row {worst[1]}  band {worst[2]}")
print(f"\nwrote {os.path.join(calib, 'norm_sigma_global.npy')}")
PY
