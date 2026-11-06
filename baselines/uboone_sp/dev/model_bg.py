import json, numpy as np
W = json.load(open("/pscratch/sd/o/oalter/helix_work/denoise/results/dn_ab_ft1024_win_trained_windows.json"))
bg = [w for w in W if w["kind"] == "bg"]
s = np.array([w["score"] for w in bg]); t0 = np.array([w["t0"] for w in bg]); g = np.array([w["g"] % 3 for w in bg])
thr = np.quantile(s, 0.99); hi = s > np.quantile(s, 0.95)
print(f"model bg q50 {np.median(s):.0f} q99 {thr:.0f}; top-5%: frac t0<160 {np.mean(t0[hi]<160):.2f} vs all {np.mean(t0<160):.2f}; planes {[(g[hi]==p).sum() for p in range(3)]}; q99 excluding t0<160: {np.quantile(s[t0>=160],0.99):.0f}")
for p in range(3): print("  plane", "UVY"[p], "bg q50/q99", np.quantile(s[g==p],[0.5,0.99]).round(0))
