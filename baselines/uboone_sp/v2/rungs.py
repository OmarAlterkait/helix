#!/usr/bin/env python3
"""Perfect-knowledge rungs in the Wire-Cell port, fitted vs exact response.

    scripts/helix_run.sh python baselines/uboone_sp/v2/rungs.py --root /pscratch/sd/o/oalter/helix_work/uboone_sp/v2/rungs \
        --out baselines/uboone_sp/v2/results/rungs.json

<root>/<input>_<kernel>/ are classical.py outputs (--settings NONE --full qor,g,zlf,zw) on the
128 test events, inputs syn (R0syn: hits (*) the same kernel), clean (R0: stored clean
sensor), inc (R1), gate (R2). Per rung:
  oracle-ROI charge (qor) and no-ROI Gaus_wide charge (g): per-cell median predicted /
  true and 68% half-width by true-charge bin, total charge recovery, per plane;
  detection (noisy inputs only): per-plane near / far 1%-FPR efficiency at 0.1-0.2 MeV
  of the no-ROI significance zlf (the classical statistic of the tuned chain on U/V).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
QB = (0, 1000, 2000, 4000, 8000, 16000, 64000, 1e9)


def fidelity(d):
    rows = {}
    for p in range(3):
        T, X = {"qor": [], "g": []}, {"qor": [], "g": []}
        tot = {"qor": [0.0, 0.0], "g": [0.0, 0.0]}
        for f in sorted(glob.glob(os.path.join(d, f"ev*_g{p}.npz")) + glob.glob(os.path.join(d, f"ev*_g{p + 3}.npz"))):
            z = np.load(f)
            t = z["full_truth"]
            for k in ("qor", "g"):
                x = z[f"full_{k}"]
                m = t > 0
                T[k].append(t[m]); X[k].append(x[m])
                tot[k][0] += x.sum(); tot[k][1] += t.sum()
        for k in ("qor", "g"):
            t, x = np.concatenate(T[k]), np.concatenate(X[k])
            r = {}
            for lo, hi in zip(QB[:-1], QB[1:]):
                m = (t > lo) & (t <= hi)
                if m.sum() >= 20:
                    q = x[m] / t[m]
                    a, b, c = np.percentile(q, [16, 50, 84])
                    r[f"{lo:g}-{hi:g}"] = dict(median=float(b), hw68=float((c - a) / 2))
            r["total"] = float(tot[k][0] / tot[k][1])
            rows[f"{k}|{'UVY'[p]}"] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import score as SC
    A = np.load(SC.ANNOT)
    ann = {k: (float(s) if pk > 0 else 0.0, bool(fo <= 0.25 * q_))
           for k, s, pk, fo, q_ in zip(zip(A["ev"].tolist(), A["g"].tolist(), A["w0"].tolist(), A["t0"].tolist()),
                                       A["snr"], A["peak"], A["foreign"], A["q"])}
    res = {}
    for d in sorted(glob.glob(os.path.join(a.root, "*_*"))):
        tag = os.path.basename(d)
        if len(glob.glob(os.path.join(d, "ev*_g*.npz"))) < 768:
            print("incomplete", tag); continue
        r = dict(fidelity=fidelity(d))
        if tag.split("_")[0] in ("inc", "gate"):
            M, wq, st, en, V = SC.load_split(SC.TEST, d, None, ann)
            T = SC.Tab(M)
            for k in ("zlf", "zw", "g"):
                o = SC.effs(T, SC.win_scores(V[k], st, en))
                r[f"det_{k}"] = {x: o.get(x) for x in ("near_pp_0.1-0.2", "far_pp_0.1-0.2", "near_U_0.1-0.2", "near_V_0.1-0.2",
                                                       "near_Y_0.1-0.2", "near_pp_snr_5-7", "near_pp_snr_7-10")}
        res[tag] = r
        print(tag, json.dumps({k: v for k, v in r.items() if k.startswith("det")})[:300], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
