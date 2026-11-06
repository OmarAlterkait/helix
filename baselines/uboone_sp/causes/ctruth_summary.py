#!/usr/bin/env python3
"""Summarise the complete-truth build: validation table (validate_*.json) and, per run, shards done,
events, unplaced deposit-charge fraction (tracks without hits; left out by definition)."""
import glob, json, os, sys
import numpy as np
O = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{O}/_validate/run_*/validate_*.json"))]
print("| run | shard | events | thr-rebuild vs hits truth, rel L2 per (event, plane): median / p95 | thr / hits charge | complete / hits charge | unplaced deposit charge |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['run']} | {r['shard']} | {r['n_events']} | {r['thr_vs_hits_rel_l2_median']:.4f} / {r['thr_vs_hits_rel_l2_p95']:.4f} | {r['thr_over_hits']:.5f} | {r['complete_over_hits']:.4f} | {r['unplaced_frac']:.3f} |")
tot = dict(shards=0, events=0)
print("\n| run | shards done | events | unplaced fraction (charge-weighted; min-max over shards) |\n|---|---|---|---|")
for d in sorted(glob.glob(f"{O}/run_*")):
    fs = sorted(f for f in glob.glob(f"{d}/truth_*.npz") if not f.endswith(".tmp.npz"))
    qt = qu = 0.0; ne = 0; fr = []
    for f in fs:
        z = np.load(f); qt += z["q_total"].sum(); qu += z["q_unplaced"].sum(); ne += len(z["event"])
        fr.append(z["q_unplaced"].sum() / max(z["q_total"].sum(), 1e-9))
    tot["shards"] += len(fs); tot["events"] += ne
    print(f"| {os.path.basename(d)} | {len(fs)} | {ne} | {qu / max(qt, 1e-9):.3f} ({min(fr) if fr else 0:.3f}-{max(fr) if fr else 0:.3f}) |")
print("\ntotal", tot)
