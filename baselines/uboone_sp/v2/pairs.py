#!/usr/bin/env python3
"""Two-deposit separation on the eval's 'pair' windows, de-confounded from detection.

helix.probe.resolution's dip is 1 ('one blob') whenever the weaker deposit's peak is
<= 0, so for a thresholded output (ROI charge) it measures detection, and for a
clipped noisy output (Gaus_wide cell sums) a noise valley at 0 reads as 'resolved'.
Here, per method and plane (128 test events):
  both      fraction of pair windows where BOTH deposits' peaks (max over the 3x3 cells
            around each deposit's centroid cell, as the eval) exceed the method's own
            far 1%-FPR window threshold for that plane
  resolved  among those, fraction with valley / weaker peak < 0.5 (valley = min over
            the cells on the segment between the two centroid cells, endpoints excluded)
  eval_dip  the eval's median dip, for reference
    scripts/helix_run.sh python baselines/uboone_sp/v2/pairs.py --out baselines/uboone_sp/v2/results/pairs.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CLS = "/pscratch/sd/o/oalter/helix_work/uboone_sp/v2/cls"
MODEL = "/pscratch/sd/o/oalter/helix_work/uboone_sp/v2/model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(HERE, "results", "benchmark.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import score as SC
    from helix.probe.resolution import SEP_BINS, FW, FT
    B = json.load(open(a.bench))
    picks = {f: {int(k): v for k, v in d.items()} for f, d in B["selection"]["pick"].items()}
    M, wq, st, en, V = SC.load_split(SC.TEST, CLS, MODEL, {})
    T = SC.Tab(M)
    wplane = np.repeat(T.g, en - st)
    meth = {}
    for fam in ("default", "tuned_q", "tuned_any"):
        v = np.zeros(len(wq))
        for p in range(3):
            v[wplane == p] = V[picks[fam][p]][wplane == p]
        meth[f"cls_{fam}"] = v
    meth["cls_noroi_gauss"] = V["g"]
    for m in SC.MODEL_STATS:
        meth[m] = V[m]
    meth["truth"] = wq
    res = {}
    shape = (16, 8)
    for name, v in meth.items():
        s = SC.win_scores(v, st, en)
        thr = {p: (np.quantile(s[(T.g == p) & (T.kind == "bg")], 0.99) if name != "truth" else 0.0) for p in range(3)}
        rows = {}
        for n in np.nonzero(T.kind == "pair")[0]:
            m = M[n]
            P = v[st[n]:en[n]].reshape(shape)
            (wa, ta), (wb, tb) = m["cen"]
            ca, cb = (int(wa // FW), int(ta // FT)), (int(wb // FW), int(tb // FT))
            dw, dt = abs(ca[0] - cb[0]), abs(ca[1] - cb[1])
            nstep = max(dw, dt)
            if nstep < 2:
                continue
            axis, dist = ("w", dw * FW) if dw * FW / 32 >= dt * FT / 128 else ("t", dt * FT)
            bins = SEP_BINS[axis]
            key = f"{axis}<={next((b for b in bins if dist <= b), bins[-1])}"
            c = lambda i, j: P[min(max(i, 0), 15), min(max(j, 0), 7)]
            pa = max(c(ca[0] + u, ca[1] + w_) for u in (-1, 0, 1) for w_ in (-1, 0, 1))
            pb = max(c(cb[0] + u, cb[1] + w_) for u in (-1, 0, 1) for w_ in (-1, 0, 1))
            line = [(round(ca[0] + (cb[0] - ca[0]) * k / nstep), round(ca[1] + (cb[1] - ca[1]) * k / nstep)) for k in range(1, nstep)]
            valley = min(c(i, j) for i, j in line)
            lo = min(pa, pb)
            both = lo > thr[m["g"] % 3]
            r = rows.setdefault(f"{'UVY'[m['g'] % 3]}|{key}", [0, 0, 0])
            r[0] += 1; r[1] += int(both); r[2] += int(both and valley / lo < 0.5)
        for k, (n_, b_, r_) in rows.items():
            res[f"{name}|{k}"] = dict(n=n_, both=b_ / n_, resolved_if_both=(r_ / b_ if b_ else float("nan")), both_and_resolved=r_ / n_)
        tot = {}
        for k, (n_, b_, r_) in rows.items():
            sep = k.split("|")[1]
            t = tot.setdefault(sep, [0, 0, 0]); t[0] += n_; t[1] += b_; t[2] += r_
        for sep, (n_, b_, r_) in tot.items():
            res[f"{name}|all|{sep}"] = dict(n=n_, both=b_ / n_, resolved_if_both=(r_ / b_ if b_ else float("nan")), both_and_resolved=r_ / n_)
    json.dump(res, open(a.out, "w"), indent=1)
    for name in meth:
        print(name, [(s, round(res[f"{name}|all|{s}"]["both"], 2), round(res[f"{name}|all|{s}"]["resolved_if_both"], 2))
                     for s in ("w<=4", "w<=8", "w<=16", "w<=24", "t<=32", "t<=64", "t<=96") if f"{name}|all|{s}" in res])


if __name__ == "__main__":
    main()
