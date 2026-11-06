"""Background windows at the readout start: score under the example's cyclic
deconvolution vs the same with 512 zero ticks padded at both ends, and the truth
charge at the END of the readout on the window's wires (cyclic wrap partner)."""
import sys, json, numpy as np
sys.path.insert(0, "baselines/uboone_sp")
from sp.data import Event, registry, noisy_planes
from sp.pipeline import Kernel, deconvolve_2d_dual, fine_cells
from sp.response import load_kernel
from sp.noise_filter import gate_clean
from helix.probe import resolution as R
reg = registry(); K = {p: Kernel(*load_kernel(p, "fit")) for p in "UVY"}
rows = []
for ev in range(260, 260 + int(sys.argv[1])):
    E = Event(ev)
    bgs = [m for m in E.meta if m["kind"] == "bg"]
    rng = np.random.default_rng(1000 + ev)
    for g in range(6):
        C = E.clean(g, reg); nw, nt = C.shape; pl = "UVY"[g % 3]
        L = np.asarray(reg[g]["wire_lengths"]); ped = reg[g]["pedestal"]
        inc, coh = noisy_planes(C, L, ped, rng)                    # same realisation as run_ladder
        mine = [m for m in bgs if m["g"] == g]
        if not mine:
            continue
        T = E.truth(g, C.shape)
        for lab, M, nl in (("R0", C, True), ("R2", gate_clean(coh), False)):
            for pad in (0, 512):
                _, cg = deconvolve_2d_dual(np.pad(M, ((0, 0), (pad, pad))), pl, K[pl], noiseless=nl)
                X = fine_cells(cg[:, pad:pad + nt])
                for m in mine:
                    w0, t0 = m["w0"], m["t0"]
                    s = X[w0 // 2:w0 // 2 + 16, t0 // 16:t0 // 16 + 8].max()
                    late = float(T[max(w0 - 16, 0):w0 + 48, nt - 200:].sum())
                    early = float(T[max(w0 - 16, 0):w0 + 48, :200].sum())
                    rows.append(dict(ev=ev, g=g, t0=t0, w0=w0, rung=lab, pad=pad, score=float(s), late=late, early=early))
json.dump(rows, open(sys.argv[2], "w"))
import collections
for lab in ("R0", "R2"):
    for pad in (0, 512):
        r = [x for x in rows if x["rung"] == lab and x["pad"] == pad]
        s = np.array([x["score"] for x in r]); t0 = np.array([x["t0"] for x in r]); late = np.array([x["late"] for x in r])
        e, m_, l = t0 < 160, (t0 >= 160) & (t0 < 4321 - 288), t0 >= 4321 - 288
        print(f"{lab} pad={pad:3d}: n {len(s)} | median score early {np.median(s[e]):.0f} mid {np.median(s[m_]):.0f} late {np.median(s[l]) if l.any() else float('nan'):.0f}"
              f" | q90 early {np.quantile(s[e],0.9):.0f} mid {np.quantile(s[m_],0.9):.0f} | early windows with charge in the last 200 ticks on their wires: "
              f"median score {np.median(s[e & (late>0)]) if (e&(late>0)).any() else float('nan'):.0f} (n {(e&(late>0)).sum()}) vs without {np.median(s[e & (late==0)]) if (e&(late==0)).any() else float('nan'):.0f} (n {(e&(late==0)).sum()})")
