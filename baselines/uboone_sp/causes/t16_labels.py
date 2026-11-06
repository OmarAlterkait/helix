#!/usr/bin/env python3
"""T16: are the networks' training labels the same per-cell charge as the truth the linear maps are scored on?

The networks train and are scored on the corpus complete truth (denoise_truth_complete, the training label); the
inversions / learned linear filter (floor.py, t13_wiener.py) are scored on causes/build.py's rebuild of complete
truth from the step files (floor outputs truth_g<g>). If the two disagree per cell by ~2%, the networks' flat ~2.5%
floor would partly be label disagreement. Per plane and truth bin (rebuild truth), on the 48 floor test events:
hw68 / median of log(label / rebuild) and the fraction of rebuild-bright cells with no label at all.

    scripts/helix_run.sh python baselines/uboone_sp/causes/t16_labels.py --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
FLOOR = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/bright/floor"
BINS = [(4e3, 16e3), (16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import denoise_products as DP
    shapes = DP._shapes()
    acc = {}
    for ev in range(260, 308):
        f = os.path.join(FLOOR, f"ev{ev:03d}.npz")
        if not os.path.exists(f):
            continue
        Z = np.load(f)
        shard, event, _ = DP._event_inputs(ev, None)
        ck, cq = DP._corpus_truth(DP.COMPLETE, shard, event)
        L = DP._maps(ck, cq, shapes)
        for g, sh in shapes.items():
            Tr = Z[f"truth_g{g}"]
            n0, n1 = min(sh[0], Tr.shape[0]), min(sh[1], Tr.shape[1])
            tr, lab = Tr[:n0, :n1].astype(np.float64), L[g][:n0, :n1]
            for k, (lo, hi) in enumerate(BINS):
                m = (tr >= lo) & (tr < hi)
                r = acc.setdefault(("UVY"[g % 3], k), [[], 0, 0])
                r[0].append(np.log(np.maximum(lab[m], 1.0) / tr[m])); r[1] += int((lab[m] <= 0).sum()); r[2] += int(m.sum())
        print(f"ev{ev}", flush=True)
    res = {}
    for (pl, k), (lr, nz, n) in sorted(acc.items()):
        x = np.concatenate(lr); ok = np.isfinite(x)
        a16, a50, a84 = np.percentile(x[ok], [16, 50, 84])
        res[f"{pl}_{BINS[k][0]:.0f}"] = dict(hw68=float((a84 - a16) / 2), med=float(a50), no_label=nz / max(n, 1), n=n)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(k, {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()})


if __name__ == "__main__":
    main()
