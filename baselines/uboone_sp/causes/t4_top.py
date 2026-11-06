"""T4a/b: truth-binned median and log-space slope at the top end, training vs test events (t2t3_model.py output)."""
import glob, sys
import numpy as np
sys.path.insert(0, "/global/u1/o/oalter/nu/helix")
from helix.probe.resolution import unkey
M = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7tests"; Q0 = 2043.2578108909702
for s in ("test", "train"):
    Z = [dict(np.load(f)) for f in sorted(glob.glob(f"{M}/t2t3_{s}/*.npz"))]
    D = {k: np.concatenate([z[k] for z in Z]) for k in Z[0]}
    T = D["Tq"]; ok = D["covered"]
    for v in ("base", "fp32"):
        q = Q0 * np.expm1(np.maximum(D[f"y_M7dc_{v}"], 0))
        line = []
        for lo, hi in ((256e3, 1e6), (1e6, 2e6), (2e6, np.inf)):
            m = ok & (T >= lo) & (T < hi)
            line.append(f"{lo/1e6:g}-{hi/1e6:g}M n={m.sum()} med={np.median(q[m] / T[m]):.4f}" if m.sum() else f"{lo/1e6:g}M n=0")
        m = ok & (T >= 256e3)
        b = np.polyfit(np.log(T[m]), np.log(np.maximum(q[m], 1)), 1)[0]
        print(s, v, " | ".join(line), f"| log-slope >=256k {b:.3f}")
