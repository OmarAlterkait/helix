import json, sys, numpy as np
d = sys.argv[1]
for r in sys.argv[2:]:
    W = json.load(open(f"{d}/windows_{r}.json"))
    for p in range(3):
        bg = np.array([w["score"] for w in W if w["kind"] == "bg" and w["g"] % 3 == p])
        bgt = np.array([w["t0"] for w in W if w["kind"] == "bg" and w["g"] % 3 == p])
        print(f"{r} {'UVY'[p]}: bg n {len(bg)} frac>0 {np.mean(bg>0):.3f} q90/95/98/99 {np.quantile(bg,[0.9,0.95,0.98,0.99]).round(0)}; excluding t0<160: q99 {np.quantile(bg[bgt>=160],0.99):.0f}")
        for eb in range(5):
            s = np.array([w["score"] for w in W if w["kind"] == "iso" and w["g"] % 3 == p and w["eb"] == eb])
            if len(s): print(f"     eb{eb} n {len(s)} frac>0 {np.mean(s>0):.2f} positive-score q25/50/75 {np.quantile(s[s>0],[0.25,0.5,0.75]).round(0) if (s>0).any() else '-'}  eff@own-q99 {np.mean(s>np.quantile(bg,0.99)):.2f} eff@q99(no edge) {np.mean(s>np.quantile(bg[bgt>=160],0.99)):.2f}")
