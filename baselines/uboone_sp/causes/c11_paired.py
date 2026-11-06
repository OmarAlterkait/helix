"""C11 paired: near the cathode, |log error| per cell with full vs per-volume encoding, paired on the same cells,
event-bootstrap (2000 resamples) of the mean difference and of the hw68 ratio, per plane and charge bin."""
import sys
import numpy as np
D = dict(np.load(sys.argv[1])); Q0 = 2043.2578108909702
yT = np.log1p(D["T"] / Q0); p = D["g"] % 3
ok = np.isfinite(D["full"]) & np.isfinite(D["split"])
rng = np.random.default_rng(0)
hw = lambda x: np.subtract(*np.percentile(x, [84, 16])) / 2
for pl in range(3):
    for lo, hi in ((16e3, 64e3), (64e3, np.inf)):
        m = ok & (p == pl) & (D["T"] >= lo) & (D["T"] < hi)
        ef, es, ev = np.abs(D["full"][m] - yT[m]), np.abs(D["split"][m] - yT[m]), D["ev"][m]
        evs = np.unique(ev); idx = {e: np.nonzero(ev == e)[0] for e in evs}
        d0 = (es - ef).mean(); r0 = hw(D["split"][m] - yT[m]) / hw(D["full"][m] - yT[m])
        bd, br = [], []
        for _ in range(2000):
            sel = np.concatenate([idx[e] for e in rng.choice(evs, len(evs))])
            bd.append((es[sel] - ef[sel]).mean())
            br.append(hw(D["split"][m][sel] - yT[m][sel]) / hw(D["full"][m][sel] - yT[m][sel]))
        print(f"{'UVY'[pl]} {lo/1e3:.0f}k+ n={m.sum()} events={len(evs)}  mean|e| split-full = {d0:+.5f} "
              f"[{np.percentile(bd, 2.5):+.5f}, {np.percentile(bd, 97.5):+.5f}]  hw68 split/full = {r0:.3f} "
              f"[{np.percentile(br, 2.5):.3f}, {np.percentile(br, 97.5):.3f}]")
