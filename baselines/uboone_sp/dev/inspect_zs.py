"""For each isolated window: does the clean sensor carry any signal on the deposit's wires/ticks?
Split R0 window scores by that."""
import json, sys, numpy as np
sys.path.insert(0, "baselines/uboone_sp")
from sp.data import Event, registry
d = sys.argv[1]
W = json.load(open(f"{d}/windows_R0.json"))
reg = registry()
out = []
for ev in sorted(set(w["ev"] for w in W)):
    E = Event(ev); ws = [w for w in W if w["ev"] == ev]
    cache = {}
    for w, m in zip(ws, E.meta):
        if w["kind"] != "iso": continue
        g = w["g"]
        if g not in cache: cache[g] = (E.clean(g, reg), E.truth(g, (reg[g]["n_wires"], 4321)))
        C, T = cache[g]
        sl = (slice(w["w0"], w["w0"] + 32), slice(w["t0"], w["t0"] + 128))
        out.append((g % 3, w["eb"], w["score"], float(np.abs(C[sl]).max()), float(T[sl].sum()), float(T[sl].max())))
o = np.array(out)
for eb in range(3):
    for p in range(3):
        m = (o[:, 0] == p) & (o[:, 1] == eb)
        if not m.any(): continue
        nos = o[m, 3] == 0
        print(f"eb{eb} {'UVY'[p]} n {m.sum():3d}: no clean signal in window {nos.mean():.2f}; R0 score 0: {np.mean(o[m,2]==0):.2f}; "
              f"score 0 & no signal {np.mean((o[m,2]==0) & nos):.2f}; score 0 WITH signal {np.mean((o[m,2]==0) & ~nos):.2f}")
