#!/usr/bin/env python3
"""Aggregate t2t3_model.py output (T2a gain response, T2b fp32, T2c jitter, T3a train vs test) and T1g
(model error by the cell's position inside its covering tokens).

    scripts/helix_run.sh python baselines/uboone_sp/causes/t2t3_summary.py --dir <m7tests> --out <json>

Error per cell: e = y - log1p(T / Q0) (log space; at high charge ~ relative error), T = training-label complete
truth. Bins of T: 4-16k, 16-64k, 64-256k, >= 256k. hw68 = half the 16-84% spread.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
Q0 = 2043.2578108909702
BINS = [(4e3, 16e3), (16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]
GAINS = (-0.02, -0.01, 0.005, 0.01, 0.02, 0.05)
PL = "UVY"


def st(x):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return dict(n=int(len(x)))
    a, m, b = np.percentile(x, [16, 50, 84])
    return dict(n=int(len(x)), med=float(m), hw68=float((b - a) / 2))


def load(d):
    Z = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(d, "*.npz")))]
    keys = list(Z[0].keys())
    return {k: np.concatenate([z[k] for z in Z]) for k in keys}, len(Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp32", action="store_true", help="T2 variants from t2t3_fp32_test (reference y = fp32)")
    a = ap.parse_args()
    from helix.model.tokenize import PatchConfig, tau_of_tick
    from helix.probe.resolution import FT, FW, unkey
    cfg = PatchConfig(cell_t="grid_center")
    res = {}
    for s in ("test", "train"):
        D, nev = load(os.path.join(a.dir, f"t2t3_{s}"))
        g, fw, ft = unkey(D["keys"]); p = g % 3
        T = D["Tq"]; yT = np.log1p(T / Q0); cov = D["covered"]
        out = dict(n_events=nev)
        for pl in range(3):
            for lo, hi in BINS:
                m = (p == pl) & (T >= lo) & (T < hi) & cov
                k = f"{PL[pl]}_{lo:.0f}"
                r = dict(base=st(D["y_M7dc_base"][m] - yT[m]), fp32=st(D["y_M7dc_fp32"][m] - yT[m]),
                         M7d=st(D["y_M7d_base"][m] - yT[m]))
                y0 = D["y_M7dc_base"][m]; q0 = Q0 * np.expm1(np.maximum(y0, 0))
                for e in GAINS:
                    dy = D[f"y_M7dc_g{e:+g}"][m] - y0
                    exp_ = np.log1p(q0 * (1 + e) / Q0) - np.log1p(q0 / Q0)
                    r[f"gain{e:+g}"] = dict(slope=st(dy / exp_), resid=st(dy - exp_))
                J = np.stack([D[f"y_M7dc_j{j}"][m] for j in range(3)])
                r["jitter_sd"] = st(J.std(0, ddof=1))
                r["jitter_vs_base"] = st((J - y0[None]).ravel())
                out[k] = r
        if s == "test":
            # T1g: position inside the covering tokens (the cell's centre pixel, as helix.probe.resolution.cell_inputs)
            import h5py
            item = json.load(open(os.path.join(a.dir, "t2t3_test", "items.json")))[0]
            with h5py.File(item[1], "r") as fh:
                bl = fh["config"]["band_lengths"][:]
            w, t = fw * FW, ft * FT + FT // 2
            pos = {"wire": (w % cfg.pw) // FW}
            for b, nm in enumerate(("A4", "D4", "D3", "D2")):
                tau = tau_of_tick(t.astype(np.float64), g, np.full(len(g), b), bl, cfg)
                pos[nm] = (np.asarray(tau, np.int64) % cfg.pt)
            e = D["y_M7dc_base"] - yT
            tg = {}
            for pl in range(3):
                for lo, hi in ((16e3, 64e3), (64e3, np.inf)):
                    m0 = (p == pl) & (T >= lo) & (T < hi) & cov
                    for nm, v in pos.items():
                        tg[f"{PL[pl]}_{lo:.0f}_{nm}"] = {int(i): st(e[m0 & (v == i)]) for i in np.unique(v[m0])}
            out["T1g"] = tg
        res[s] = out
    json.dump(res, open(a.out, "w"), indent=1)
    for s in ("test", "train"):
        print("==", s, res[s]["n_events"], "events")
        for k, r in res[s].items():
            if k in ("n_events", "T1g"):
                continue
            print(k, "base", round(r["base"].get("med", 0), 4), round(r["base"].get("hw68", 0), 4), r["base"]["n"],
                  "fp32", round(r["fp32"].get("hw68", 0), 4), "M7d", round(r["M7d"].get("hw68", 0), 4),
                  "gain+1% slope", round(r["gain+0.01"]["slope"].get("med", 0), 3), round(r["gain+0.01"]["slope"].get("hw68", 0), 3),
                  "gain+5% resid", round(r["gain+0.05"]["resid"].get("hw68", 0), 4),
                  "jit sd", round(r["jitter_sd"].get("med", 0), 5))


if __name__ == "__main__":
    main()
