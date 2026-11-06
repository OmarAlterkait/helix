#!/usr/bin/env python3
"""T0 of results/m7_tests_plan.md: independent recomputation of the high-charge comparison (own code,
raw inputs only: model_dump2 dumps, products params, floor.py maps; nothing from m7diag_*).

    scripts/helix_run.sh python baselines/uboone_sp/causes/t0_verify.py --events 260-307 --out <json>

T0a alignment scan, T0b per-truth-bin median / hw68 of pred/truth for model calo and corpus Tikhonov (all
lambdas), T0c the same with cells binned by the inversion's value and by the model's value.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
FLOOR = f"{R}/bright/floor"
BINS = [(16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]
LAMS = ("0.0001", "0.001", "0.01")
PL = "UVY"


def hw(r):
    if len(r) < 20:
        return dict(n=int(len(r)))
    q16, q50, q84 = np.percentile(r, [16, 50, 84])
    return dict(n=int(len(r)), med=float(q50), hw68=float((q84 - q16) / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-307")
    ap.add_argument("--models", default="M7dc:products/model_test_m7,M7d:products/model_test_m7")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import denoise_products as DP
    lo, hi = map(int, a.events.split("-"))
    evs = [e for e in range(lo, hi + 1) if os.path.exists(f"{FLOOR}/ev{e:03d}.npz")]
    models = [m.split(":") for m in a.models.split(",")]
    prm = {m: json.load(open(f"{R}/products/params_{m}.json"))[m] for m, _ in models}
    # accumulators: per plane, arrays of (Tf, model calo..., tik...) over Tf >= 16k cells
    rows = {p: [] for p in range(3)}
    shifts = [(dw, dt) for dw in range(-2, 3) for dt in range(-3, 4)]
    corr = {m: {p: np.zeros((len(shifts), 3)) for p in range(3)} for m, _ in models}   # sum xy, xx, yy
    for ev in evs:
        F = np.load(f"{FLOOR}/ev{ev:03d}.npz")
        Z = {d: np.load(f"{R}/{d}/ev{ev:03d}.npz") for _, d in models}
        for g in range(6):
            p = g % 3
            Tf = F[f"truth_g{g}"].astype(np.float64)
            nw, nt = Tf.shape
            cols = [Tf]
            for m, d in models:
                z = Z[d]; to = int(z["toff"])
                q, pp, cov = z[f"q_{m}_g{g}"].astype(np.float64), z[f"p_{m}_g{g}"].astype(np.float64), z[f"cov_g{g}"]
                calo_full = DP.products_maps(q, pp, cov, p, prm[m])["calo"]
                # T0a: model shifted against truth (truth window fixed, model window moved)
                sel = Tf >= 1000
                x = np.log1p(Tf[sel] / 2043.26)
                P = np.pad(calo_full, 4)
                for i, (dw, dt) in enumerate(shifts):
                    M = P[4 + dw:4 + dw + nw, 4 + to + dt:4 + to + dt + nt]
                    y = np.log1p(M[sel] / 2043.26)
                    corr[m][p][i] += [(x * y).sum(), (x * x).sum(), (y * y).sum()]
                cols.append(calo_full[:nw, to:to + nt])
            for lam in LAMS:
                cols.append(F[f"corpus_tik{lam}_g{g}"].astype(np.float64))
            S = np.stack([c.ravel() for c in cols], 1)
            rows[p].append(S[S[:, 0] >= 1000])
    names = ["Tf"] + [f"{m}_calo" for m, _ in models] + [f"tik{l}" for l in LAMS]
    out = dict(events=evs, names=names, align={}, truth_binned={}, model_binned={}, tik_binned={})
    for m, _ in models:
        out["align"][m] = {}
        for p in range(3):
            c = corr[m][p]
            r = c[:, 0] / np.sqrt(np.maximum(c[:, 1] * c[:, 2], 1e-30))
            best = int(np.argmax(r))
            out["align"][m][PL[p]] = dict(best_shift=shifts[best], r_best=float(r[best]),
                                          r_00=float(r[shifts.index((0, 0))]),
                                          r_next=float(np.sort(r)[-2]))
    for p in range(3):
        A = np.concatenate(rows[p])
        Tf = A[:, 0]
        for key, sel_col in (("truth_binned", 0), ("model_binned", 1), ("tik_binned", names.index("tik0.001"))):
            v = A[:, sel_col]
            res = {}
            for lo_, hi_ in BINS:
                b = (v >= lo_) & (v < hi_) & (Tf > 0)
                res[f"{lo_:.0f}-{hi_:.0f}"] = {n: hw(A[b, j] / Tf[b] - 1) for j, n in enumerate(names) if j > 0}
            out[key][PL[p]] = res
    json.dump(out, open(a.out, "w"), indent=1)
    for p in PL:
        print(p, "align", {m: out["align"][m][p] for m, _ in models})
        for b, d in out["truth_binned"][p].items():
            print(" ", p, b, {n: (x.get("med", 0), x.get("hw68", 0), x["n"]) for n, x in d.items()})


if __name__ == "__main__":
    main()
