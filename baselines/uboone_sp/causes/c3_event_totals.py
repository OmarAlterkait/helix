"""C3: does the event generator make charge in one part of the event predict the rest? Per event, total complete-truth
charge per TPC volume and per plane (training-label truth, denoise_truth_complete), over many events from every run:
the spread of the event total, and the correlation between volumes / planes. A negative correlation (fixed total
energy split between volumes) would make 'brighter elsewhere => dimmer here' the optimal masked-prediction prior."""
import glob, os, sys
import numpy as np
sys.path.insert(0, "/global/u1/o/oalter/nu/helix")
from helix.probe.resolution import unkey
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
rows = []
for run in sorted(d for d in os.listdir(CT) if d.startswith("run_")):
    for f in sorted(glob.glob(f"{CT}/{run}/truth_*.npz"))[:6]:          # 6 shards x ~200 events per run
        z = np.load(f)
        off, keys, q = z["offsets"], z["keys"], z["q"]
        g = unkey(keys)[0]
        for i in range(len(off) - 1):
            a, b = off[i], off[i + 1]
            s = np.bincount(g[a:b], weights=q[a:b], minlength=6)
            rows.append([hash(run) % 1000] + s.tolist())
A = np.array(rows); S = A[:, 1:]
v0, v1 = S[:, :3].sum(1), S[:, 3:].sum(1); tot = v0 + v1
print(f"events {len(A)}")
print(f"event total charge (e-): median {np.median(tot):.3g}  16-84% {np.percentile(tot,16):.3g} - {np.percentile(tot,84):.3g}  "
      f"(relative spread hw68/median {np.subtract(*np.percentile(tot,[84,16]))/2/np.median(tot):.2f})")
print(f"corr(vol0, vol1) = {np.corrcoef(v0, v1)[0,1]:.3f}   corr(log vol0, log vol1) = {np.corrcoef(np.log1p(v0), np.log1p(v1))[0,1]:.3f}")
f0 = v0 / np.maximum(tot, 1)
print(f"fraction of the event's charge in vol0: median {np.median(f0):.2f}, 16-84% {np.percentile(f0,16):.2f}-{np.percentile(f0,84):.2f}")
print("corr between planes (U,V,Y of vol0 then vol1):"); print(np.round(np.corrcoef(S.T), 3))
