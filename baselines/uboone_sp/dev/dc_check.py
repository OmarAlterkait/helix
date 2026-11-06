"""Why does Y lose ~8% of its charge from R0 to R1? Separate DC mode from noise."""
import sys, numpy as np
sys.path.insert(0, "baselines/uboone_sp")
from sp.data import Event, registry, noisy_planes
from sp.pipeline import Kernel, run_sp
from sp.response import load_kernel
reg = registry(); E = Event(300)
for g in (2, 0):
    pl = "UVY"[g % 3]
    C = E.clean(g, reg); T = E.truth(g, C.shape)
    K = Kernel(*load_kernel(pl, "fit"))
    rng = np.random.default_rng(1000 + 300)
    inc, _ = noisy_planes(C, np.asarray(reg[g]["wire_lengths"]), reg[g]["pedestal"], rng)
    for lab, M, nl in (("clean, noiseless=True", C, True), ("clean, noiseless=False", C, False), ("noisy, noiseless=False", inc, False)):
        r = run_sp(M, pl, K, truth=T, use_true_roi=True, noiseless=nl)
        m = r["roi_mask"]
        big = T.sum(1) > 20000
        print(f"{pl} {lab:24s}: ROI charge / hits {r['C_charge'][m].sum() / T.sum():.4f}; C_gauss in ROI {r['C_gauss'][m].sum()/T.sum():.4f}; "
              f"mean C_gauss outside ROI on big wires {r['C_gauss'][big][~m[big]].mean():.2f} e-/tick; mean ROI length {m.sum(1)[m.any(1)].mean():.0f}")
