#!/usr/bin/env python3
"""T2a / T2c in fp32 (t2t3_model.py --fp32-variants): the response of M7dc's charge output to a global input
gain, and to 1e-3 random input jitter, per plane and truth bin.

    scripts/helix_run.sh python baselines/uboone_sp/causes/t2_fp32_summary.py

slope = dy / dy_lin, dy_lin = log1p(q (1+e) / Q0) - log1p(q / Q0), q = the fp32 prediction: 1 for a map that is
linear in the input amplitude. resid = dy - dy_lin (log units ~ relative charge). jitter: per-cell sd of y over
3 jittered inputs, and that sd relative to the cell's error |y - log1p(T/Q0)|.
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "/global/u1/o/oalter/nu/helix")
from helix.probe.resolution import unkey  # noqa: E402

M = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7tests"
Q0 = 2043.2578108909702
GAINS = (-0.02, -0.01, 0.005, 0.01, 0.02, 0.05)
BINS = [(4e3, 16e3), (16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]


def st(x):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return dict(n=int(len(x)))
    a, m, b = np.percentile(x, [16, 50, 84])
    return dict(n=int(len(x)), med=float(m), hw68=float((b - a) / 2), mean=float(x.mean()))


Z = [dict(np.load(f)) for f in sorted(glob.glob(M + "/t2_fp32_test/*.npz"))]
D = {k: np.concatenate([z[k] for z in Z]) for k in Z[0]}
g, _, _ = unkey(D["keys"]); p = g % 3
T = D["Tq"]; yT = np.log1p(T / Q0); cov = D["covered"]
y0 = D["y_M7dc_fp32"]; q0 = Q0 * np.expm1(np.maximum(y0, 0))
out = {}
for pl, nm in enumerate("UVY"):
    for lo, hi in BINS:
        m = (p == pl) & (T >= lo) & (T < hi) & cov
        r = dict(err=st(y0[m] - yT[m]))
        for e in GAINS:
            dy = D[f"y_M7dc_g{e:+g}"][m] - y0[m]
            lin = np.log1p(q0[m] * (1 + e) / Q0) - np.log1p(q0[m] / Q0)
            r[f"g{e:+g}"] = dict(slope=st(dy / lin), resid=st(dy - lin))
        J = np.stack([D[f"y_M7dc_j{k}"][m] for k in range(3)])
        sd = J.std(0, ddof=1)
        r["jitter_sd"] = st(sd)
        r["jitter_sd_over_err"] = st(sd / np.maximum(np.abs(y0[m] - yT[m]), 1e-6))
        out[f"{nm}_{lo:.0f}"] = r
        print(f"{nm} {lo/1e3:>4.0f}k+ n={m.sum():6d} err hw68 {r['err']['hw68']:.4f} | "
              + " ".join(f"g{e:+g}: slope {r[f'g{e:+g}']['slope']['med']:.3f}±{r[f'g{e:+g}']['slope']['hw68']:.3f} resid {r[f'g{e:+g}']['resid']['hw68']:.4f}"
                         for e in (-0.02, 0.01, 0.05))
              + f" | jitter sd med {r['jitter_sd']['med']:.5f}")
json.dump(out, open(M + "/t2_fp32_summary.json", "w"), indent=1)
