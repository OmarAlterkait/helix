"""Is the readout-start (t0 < 160) background excess an edge effect of the deconvolution?

For a few test events, per plane:
  (1) hits outside the readout window (tick < 0 or >= n_ticks) -- charge the
      truth drops but whose response can still reach the stored sensor;
  (2) stored clean signal in the first 160 ticks on wires with NO in-window truth
      in ticks [0, 300) -- sensor content no in-window charge explains;
  (3) the noise-free no-ROI deconvolution (R0nr) on charge-free wires, cell-sum
      max |value| in early (t < 160), middle and late (t > n-160) ticks, computed
      (a) as the example does (cyclic FFT over the readout) and (b) with the
      waveform zero-padded by 512 ticks at both ends first.
"""
import sys, numpy as np, h5py, hdf5plugin  # noqa
sys.path.insert(0, "baselines/uboone_sp")
from sp.data import Event, registry
from sp.pipeline import Kernel, deconvolve_2d_dual, fine_cells
from sp.response import load_kernel
from helix.probe.truth import decode_hits_plane

reg = registry()
K = {p: Kernel(*load_kernel(p, "fit")) for p in "UVY"}
agg = {}
for ev in range(260, 260 + int(sys.argv[1])):
    E = Event(ev)
    with h5py.File(E.hits, "r") as fh:
        evg = fh[f"event_{E.event:03d}"]
        for g in range(6):
            vi, pi = divmod(g, 3); pl = "UVY"[pi]
            s = decode_hits_plane(evg[f"volume_{vi}"][pl])
            C = E.clean(g, reg); nw, nt = C.shape
            T = E.truth(g, C.shape)
            lo, hi = s["tick"] < 0, s["tick"] >= nt
            a = agg.setdefault(pl, dict(q_neg=0.0, q_late=0.0, q_all=0.0, n_neg_planes=0, early_sig=[], cyc=[], pad=[]))
            a["q_neg"] += float(s["q"][lo].sum()); a["q_late"] += float(s["q"][hi].sum()); a["q_all"] += float(s["q"].sum())
            a["n_neg_planes"] += int(lo.any())
            free = T[:, :300].sum(1) == 0
            a["early_sig"].append(float(np.abs(C[free, :160]).sum() / max(free.sum(), 1)))
            quiet = T.sum(1) == 0                       # wires with no in-window charge at all
            for key, pad in (("cyc", 0), ("pad", 512)):
                M = np.pad(C, ((0, 0), (pad, pad)))
                _, cg = deconvolve_2d_dual(M, pl, K[pl], noiseless=True)
                cg = cg[:, pad:pad + nt]
                X = np.abs(np.pad(cg, ((0, nw % 2), (0, (-nt) % 16))).reshape((nw + nw % 2) // 2, 2, -1, 16).sum((1, 3)))
                qc = np.append(quiet, [True] * (nw % 2)).reshape(-1, 2).all(1)
                a[key].append([float(X[qc, :10].max(initial=0)), float(X[qc, 10:-10].max(initial=0)), float(X[qc, -10:].max(initial=0)),
                               float(np.median(X[qc, :10])), float(np.median(X[qc, 10:-10])), float(np.median(X[qc, -10:]))])
for pl, a in agg.items():
    c, p = np.array(a["cyc"]), np.array(a["pad"])
    print(f"{pl}: hits charge at tick<0: {a['q_neg'] / a['q_all']:.4f} of all ({a['n_neg_planes']} planes have some), tick>=n: {a['q_late'] / a['q_all']:.4f}; "
          f"mean |clean| ADC sum per charge-free wire in t<160: {np.mean(a['early_sig']):.1f}")
    print(f"    R0nr on charge-free wires, |cell| max early/mid/late (median over planes): cyclic {np.median(c[:, :3], 0).round(0)}, "
          f"padded {np.median(p[:, :3], 0).round(0)}; typical |cell| (median) early/mid/late: cyclic {np.median(c[:, 3:], 0).round(1)}, padded {np.median(p[:, 3:], 0).round(1)}")
