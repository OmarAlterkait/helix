"""Why does the example's JAXTPC-derived kernel fit our sensor data worse than a
least-squares kernel? Test truncation, gain, time offset, and the residual each
fix leaves, on held-out test events (ev260-263), per plane."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from sp.data import Event, registry
from sp.response import load_kernel, forward, store_like_sensor

def on_grid(R, wo, tzb, T0=170, NT=420, W0=12):
    """Kernel on a common (time -T0..NT-T0, wire -W0..W0) grid."""
    G = np.zeros((NT, 2 * W0 + 1))
    for j, o in enumerate(wo):
        for k in range(R.shape[0]):
            t = k - tzb + T0
            if 0 <= t < NT and -W0 <= o <= W0:
                G[t, o + W0] += R[k, j]
    return G

def to_kernel(G, T0=170, W0=12):
    return G, np.arange(-W0, W0 + 1), T0

reg = registry()
for p, name in enumerate("UVY"):
    Re, woe, tze = load_kernel(name, "example"); Rf, wof, tzf = load_kernel(name, "fit")
    Ge, Gf = on_grid(Re, woe, tze), on_grid(Rf, wof, tzf)
    # support: energy of the fitted kernel outside the example's time / wire extent
    te = np.zeros(Ge.shape[0], bool); te[170 - tze:170 - tze + Re.shape[0]] = True
    we = np.zeros(Ge.shape[1], bool); we[woe + 12] = True
    e_out_t = (Gf[~te] ** 2).sum() / (Gf ** 2).sum(); e_out_w = (Gf[:, ~we] ** 2).sum() / (Gf ** 2).sum()
    print(f"\n== {name}: example {Re.shape} wires {woe.min()}..{woe.max()} tzb {tze}; fit {Rf.shape} wires {wof.min()}..{wof.max()} tzb {tzf}")
    print(f"  fitted-kernel energy outside the example's time extent {e_out_t:.4f}, outside its wires {e_out_w:.4f}")
    # best scale and time shift of example onto fit
    best = None
    for s in range(-6, 7):
        Gs = np.roll(Ge, s, axis=0)
        a = (Gs * Gf).sum() / (Gs ** 2).sum()
        r = np.linalg.norm(Gf - a * Gs) / np.linalg.norm(Gf)
        if best is None or r < best[2]:
            best = (s, a, r)
    print(f"  example -> fit: best time shift {best[0]:+d} ticks, gain {best[1]:.3f}, kernel-shape residual {best[2]:.3f}"
          f"  (no shift/gain: {np.linalg.norm(Gf - Ge) / np.linalg.norm(Gf):.3f})")
    for j in (-1, 0, 1):
        ce, cf = Ge[:, j + 12], Gf[:, j + 12]
        print(f"   wire {j:+d}: sum ex/fit {ce.sum():+.4f}/{cf.sum():+.4f}  peak ex/fit {np.abs(ce).max():.4f}/{np.abs(cf).max():.4f}"
              f"  argmax ex/fit {np.argmax(np.abs(ce)) - 170:+d}/{np.argmax(np.abs(cf)) - 170:+d}")
    # held-out forward-fold residuals with variants
    variants = {"example": (Re, woe, tze), "fit": (Rf, wof, tzf),
                "example shift+gain": to_kernel(best[1] * np.roll(Ge, best[0], axis=0)),
                "example + fit tails": to_kernel(np.where(te[:, None] & we[None, :], best[1] * np.roll(Ge, best[0], axis=0), Gf))}
    res = {k: [] for k in variants}
    for ev in range(260, 264):
        E = Event(ev)
        for g in (p, p + 3):
            C = E.clean(g, reg); H = E.truth(g, C.shape)
            if H.sum() <= 0:
                continue
            for k, (R, wo, tz) in variants.items():
                res[k].append(np.linalg.norm(C - store_like_sensor(forward(H, R, wo, tz))) / np.linalg.norm(C))
    print("  held-out residual |clean - stored(hits * K)| / |clean|: " + ", ".join(f"{k} {np.mean(v):.4f}" for k, v in res.items()))
