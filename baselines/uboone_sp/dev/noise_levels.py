"""What the MicroBooNE ROI thresholds are in electrons on our noise, and how big
the deconvolved noise is per tick and per fine cell (event 300's dumped planes)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sp.pipeline import PLANE_PARAMS, apply_lf_filter_to_wiener, fine_cells  # noqa: E402
from uboone_pipeline.roi import compute_rms  # noqa: E402

d = sys.argv[1]
for g in range(6):
    f = os.path.join(d, f"full_ev300_g{g}.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    pl = "UVY"[g % 3]
    p = PLANE_PARAMS[pl]
    T = z["truth"]
    quiet = T.sum(1) == 0                     # wires with no charge at all
    cw = z["R3_wiener"]
    out = [f"g{g} {pl}: {quiet.sum()} charge-free wires"]
    if pl == "Y":
        rms = np.median([compute_rms(cw[w]) for w in np.nonzero(quiet)[0][:300]])
        out.append(f"Wiener rms {rms:.0f} e-/tick -> ROI threshold {p['roi_th_factor']} x = {p['roi_th_factor'] * rms:.0f} e-/tick")
    else:
        for which in ("tight", "loose"):
            c = apply_lf_filter_to_wiener(cw, pl, which=which)
            rms = np.median([compute_rms(c[w]) for w in np.nonzero(quiet)[0][:300]])
            out.append(f"{which}-LF Wiener rms {rms:.0f} e-/tick -> threshold {p['roi_th_factor'] * rms:.0f}")
    cg = z["R3_gauss"]
    out.append(f"Gauss rms {np.median([compute_rms(cg[w]) for w in np.nonzero(quiet)[0][:300]]):.0f} e-/tick")
    X = fine_cells(cg * 1.0) * 0 + np.pad(cg, ((0, cg.shape[0] % 2), (0, (-cg.shape[1]) % 16))).reshape(
        (cg.shape[0] + cg.shape[0] % 2) // 2, 2, -1, 16).sum((1, 3))
    qc = np.repeat(quiet, 1)[: X.shape[0] * 2].reshape(-1, 2).all(1) if quiet.shape[0] % 2 == 0 else \
        np.append(quiet, True).reshape(-1, 2).all(1)
    out.append(f"unclipped cell sum (2w x 16t) on charge-free wires: std {X[qc].std():.0f} e-, q99 {np.quantile(X[qc], 0.99):.0f}")
    # a deposit's peak per-tick charge in truth, for scale: 0.1-0.2 MeV ~ 2-4k e- total
    print("; ".join(out), flush=True)
