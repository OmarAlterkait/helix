"""T8b: does the deterministic (draw-repeating) bright error depend on event size (token count / total charge)?"""
import glob, os, sys
import numpy as np
sys.path.insert(0, "/global/u1/o/oalter/nu/helix")
from helix.probe.resolution import unkey
M = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7tests"; Q0 = 2043.2578108909702
rows = []
for f in sorted(glob.glob(M + "/t8/ev*.npz")):
    D = np.load(f); T = D["Tq"]; g, _, _ = unkey(D["keys"])
    E = np.stack([np.log(np.maximum(Q0 * np.expm1(np.maximum(D[f"y_d{k}"], 0)), 1.0)) for k in range(5)]) - np.log(T)[None]
    s = E.mean(0); m = (T >= 64e3) & np.isfinite(s)
    a16, a84 = np.percentile(s[m], [16, 84])
    rows.append((int(os.path.basename(f)[2:5]), float(T.sum()), int(len(T)), int(m.sum()), float(np.median(s[m])), float((a84 - a16) / 2)))
R = np.array(rows)
print("ev total_charge n_bright n_64k med_sys hw_sys"); [print(*[round(x, 4) if isinstance(x, float) else x for x in r]) for r in rows]
for j, nm in ((1, "total charge"), (2, "n bright cells")):
    print(f"corr(hw_sys, {nm}) = {np.corrcoef(R[:, 5], R[:, j])[0, 1]:.3f}   corr(med_sys, {nm}) = {np.corrcoef(R[:, 4], R[:, j])[0, 1]:.3f}")
