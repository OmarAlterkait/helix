#!/usr/bin/env python3
"""Summaries of T2d (partial-scaling response) and T8 (noise-draw decomposition of the bright error).

    scripts/helix_run.sh python baselines/uboone_sp/causes/t2d_t8_summary.py [t2d] [t8]
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "/global/u1/o/oalter/nu/helix")
from helix.model.tokenize import PatchConfig  # noqa: E402
from helix.probe.resolution import FT, FW, unkey  # noqa: E402

M = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7tests"
Q0 = 2043.2578108909702
EPS = 0.05
BINS = [(16e3, 64e3), (64e3, np.inf)]


def load(d):
    Z = [dict(np.load(f)) for f in sorted(glob.glob(d + "/ev*.npz"))]
    return {k: np.concatenate([z[k] for z in Z]) for k in Z[0]}, len(Z)


def med_hw(x):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return (float("nan"), float("nan"), int(len(x)))
    a, m, b = np.percentile(x, [16, 50, 84])
    return (round(float(m), 3), round(float((b - a) / 2), 3), int(len(x)))


def t2d(sub="t2d"):
    D, n = load(M + "/" + sub)
    g, fw, ft = unkey(D["keys"]); p = g % 3; T = D["Tq"]; ok = D["covered"]
    y0 = D["y_base"]; q0 = Q0 * np.expm1(np.maximum(y0, 0))
    lin = np.log1p(q0 * (1 + EPS) / Q0) - np.log1p(q0 / Q0)
    w = fw * FW + 1; t = ft * FT + FT // 2
    regions = {
        "all": (np.ones_like(ok), None),
        "wstrip": (((w // 64) % 2 == 0) & (w % 64 >= 16) & (w % 64 < 48), ((w // 64) % 2 == 1) & (w % 64 >= 16) & (w % 64 < 48)),
        "tblock": (((t // 512) % 2 == 0) & (t % 512 >= 128) & (t % 512 < 384), ((t // 512) % 2 == 1) & (t % 512 >= 128) & (t % 512 < 384)),
    }
    for gs in range(6):
        regions[f"plane{gs}"] = (g == gs, g != gs)
    for v in range(2):
        regions[f"vol{v}"] = (g // 3 == v, g // 3 != v)
    for b in range(4):
        regions[f"band{b}"] = (np.ones_like(ok), None)
    out = {"n_events": n}
    for name, (ins, ctl) in regions.items():
        s = (D[f"y_{name}"] - y0) / lin
        r = {}
        for pl in range(3):
            for lo, hi in BINS:
                m = ok & (p == pl) & (T >= lo) & (T < hi)
                r[f"{'UVY'[pl]}_{lo:.0f}"] = dict(inside=med_hw(s[m & ins]), control=med_hw(s[m & ctl]) if ctl is not None else None)
        out[name] = r
        print(name, {k: (v["inside"][:2], v["control"][:2] if v["control"] else None) for k, v in r.items()})
    # band responses should add up to 'all' if the response is linear in the per-band scale
    tot = sum((D[f"y_band{b}"] - y0) for b in range(4)) / lin
    out["band_sum"] = {f"{'UVY'[pl]}_{lo:.0f}": med_hw(tot[ok & (p == pl) & (T >= lo) & (T < hi)]) for pl in range(3) for lo, hi in BINS}
    print("band_sum", out["band_sum"])
    # bright accuracy of the base prediction (training-label complete truth)
    yT = np.log1p(T / Q0)
    out["accuracy"] = {f"{'UVY'[pl]}_{lo:.0f}": med_hw((y0 - yT)[ok & (p == pl) & (T >= lo) & (T < hi)]) for pl in range(3) for lo, hi in BINS}
    print("accuracy (med, hw68, n)", out["accuracy"])
    json.dump(out, open(M + f"/{sub}_summary.json", "w"), indent=1)


def t8():
    D, n = load(M + "/t8")
    g, _, _ = unkey(D["keys"]); p = g % 3; T = D["Tq"]; lT = np.log(T)
    K = len([k for k in D if k.startswith("y_d")])
    est = {"model": np.stack([np.log(np.maximum(Q0 * np.expm1(np.maximum(D[f"y_d{k}"], 0)), 1.0)) for k in range(K)])}
    for lam in ("0.0001", "0.001"):
        est[f"tik{lam}"] = np.stack([np.log(np.maximum(D[f"tik{lam}_d{k}"], 1.0)) for k in range(K)])
    out = {"n_events": n, "K": K}
    for pl in range(3):
        for lo, hi in [(16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]:
            m = (p == pl) & (T >= lo) & (T < hi) & np.isfinite(est["model"]).all(0)
            r = {}
            for nm, E in est.items():
                e = E[:, m] - lT[m][None]
                v_n = float(e.var(0, ddof=1).mean())
                s = e.mean(0)
                v_s = float(s.var() - v_n / K)
                a16, a84 = np.percentile(e.ravel(), [16, 84])
                # robust: hw68 of the draw-mean error (systematic + noise/sqrt(K)) and of the per-draw deviation
                # from it (noise; x sqrt(K/(K-1)) for the mean's own share)
                h = lambda x: float(np.subtract(*np.percentile(x, [84, 16])) / 2)
                hn = h((e - s[None]).ravel()) * np.sqrt(K / (K - 1))
                hs2 = h(s) ** 2 - hn ** 2 / K
                r[nm] = dict(rms_noise=float(np.sqrt(v_n)), rms_sys=float(np.sqrt(max(v_s, 0))), share_sys=float(v_s / (v_s + v_n)),
                             hw_noise=hn, hw_sys=float(np.sqrt(max(hs2, 0))), share_sys_hw=float(max(hs2, 0) / (max(hs2, 0) + hn ** 2)),
                             err_hw68=float((a84 - a16) / 2), n=int(m.sum()))
            out[f"{'UVY'[pl]}_{lo:.0f}"] = r
            print(f"{'UVY'[pl]} {lo/1e3:>4.0f}k+", {k: dict(err=round(v["err_hw68"], 4), sys=round(v["hw_sys"], 4), noise=round(v["hw_noise"], 4), share_sys=round(v["share_sys_hw"], 2)) for k, v in r.items()})
    json.dump(out, open(M + "/t8_summary.json", "w"), indent=1)


if __name__ == "__main__":
    for a in sys.argv[1:] or ["t2d", "t8"]:
        if a.startswith("t2d"):
            t2d(a)
        else:
            t8()
