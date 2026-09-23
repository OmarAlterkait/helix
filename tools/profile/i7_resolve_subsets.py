"""I7 — resolve N_TRAIN_EVENTS for cooldown run-subsets.

coeff_fm_cooldown.py sets the cooldown LENGTH by annealing one epoch over the
first K runs, because `epoch` is an integer. K=3 gives 57,059 train events =
14,264 steps = 12.7% of the 112,679-step stable phase. The WSD literature puts
the optimum at 10-20% and the measured cooldown was still rising when it ended,
so the ladder needs arms above 12.7%. This resolves the identity-split train
count for each K so those arms can be written the same way the existing one was
-- resolved, then stated, not guessed.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prof"))
from common import emit
from helix.data import CoeffTPCDataset
from helix.data.identity import corpus_runs
from helix.paths import root

ROOT = str(root("HELIX_CORPUS").parent)
RUNS = corpus_runs(ROOT)
HOLDOUT = dict(seed=0, fractions=dict(train=0.95, val=0.03, probe=0.02))
STABLE = 112_679
WORLD = 4
print(f"corpus root {ROOT}\n{len(RUNS)} runs: {RUNS}\n", flush=True)
R = {"runs": list(RUNS), "stable_steps": STABLE, "world": WORLD, "subsets": {}}
print(f"{'K':>3s} {'train events':>13s} {'steps':>8s} {'% of stable':>12s}")
for K in range(1, len(RUNS) + 1):
    ds = CoeffTPCDataset(data_root=ROOT, split=list(RUNS[:K]),
                         dataset_name="sim_wire", modalities=("coeff", "coeff_clean"),
                         transform=None, holdout=HOLDOUT, split_role="train")
    n = len(ds)
    steps = n // WORLD
    R["subsets"][K] = dict(runs=list(RUNS[:K]), n_train=n, steps=steps,
                           pct_of_stable=100 * steps / STABLE)
    print(f"{K:3d} {n:13,d} {steps:8,d} {100*steps/STABLE:11.1f}%", flush=True)
print("\n  the existing cooldown is K=3. The 10-20% band and 'still rising at")
print("  the end' both argue for the next arms up.")
emit("i7_subsets", R)
