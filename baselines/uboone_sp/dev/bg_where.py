import json, sys, numpy as np
d = sys.argv[1]
for r in sys.argv[2:]:
    W = [w for w in json.load(open(f"{d}/windows_{r}.json")) if w["kind"] == "bg"]
    s = np.array([w["score"] for w in W]); t0 = np.array([w["t0"] for w in W]); w0 = np.array([w["w0"] for w in W]); g = np.array([w["g"] % 3 for w in W])
    hi = s >= np.quantile(s, 0.95)
    print(f"{r}: n bg {len(W)}; top-5% scores >= {np.quantile(s,0.95):.0f}: t0 pct [0,25,50,75,100] {np.percentile(t0[hi],[0,25,50,75,100])} (all bg: {np.percentile(t0,[25,50,75])}); "
          f"w0 median {np.median(w0[hi]):.0f}; planes U/V/Y {[(g[hi]==p).sum() for p in range(3)]}; frac t0<160 among top {np.mean(t0[hi]<160):.2f} vs all {np.mean(t0<160):.2f}; frac t0>4150 top {np.mean(t0[hi]>4150):.2f} vs all {np.mean(t0>4150):.2f}")
