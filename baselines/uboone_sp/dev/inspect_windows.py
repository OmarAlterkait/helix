import json, sys, numpy as np, collections
d = sys.argv[1]
for r in sys.argv[2:]:
    W = json.load(open(f"{d}/windows_{r}.json"))
    bg = np.array([w["score"] for w in W if w["kind"] == "bg"]); thr = np.quantile(bg, 0.99)
    print(r, "bg: frac zero", np.mean(bg == 0), "q50/q90/q99", np.quantile(bg, [0.5, 0.9, 0.99]), "n", len(bg))
    for eb in range(3):
        s = np.array([w["score"] for w in W if w["kind"] == "iso" and w["eb"] == eb])
        g = np.array([w["g"] % 3 for w in W if w["kind"] == "iso" and w["eb"] == eb])
        print(f"  eb{eb} n {len(s)} frac zero {np.mean(s == 0):.3f} frac<=thr {np.mean(s <= thr):.3f}  zero by plane",
              [round(float(np.mean(s[g == p] == 0)), 3) for p in range(3)], "median score", np.median(s))
