"""C8: the pretrained FM's categorical value bins, in amplitude terms. Per band: bin edges in arcsinh(v/sigma) and,
for bins whose centre is bright (|v/sigma| > 27, arcsinh > 4), the bin width as an amplitude ratio exp(width) - 1."""
import sys
import numpy as np
sys.path.insert(0, "/global/u1/o/oalter/nu/helix")
from helix.model.artifact import load
art = load("/pscratch/sd/o/oalter/helix_work/probe/artifact_nz_pw16_s1")
sd = art.state_dict
for k in ("bin_edges", "bin_cent_asinh"):
    print(k, tuple(sd[k].shape) if k in sd else "missing")
E = np.asarray(sd["bin_edges"], np.float64)
E = E if E.ndim == 2 else E[None]
for b, e in enumerate(E):
    e = e[np.isfinite(e)]
    w = np.diff(e); c = 0.5 * (e[1:] + e[:-1])
    def rel(lo, hi):
        m = (np.abs(c) >= lo) & (np.abs(c) < hi)
        return (np.round(np.median(np.expm1(w[m])) * 100, 1), int(m.sum())) if m.any() else None
    print(f"band {b}: {len(e)} edges, range {e.min():.2f}..{e.max():.2f}; amplitude width (%, n bins) "
          f"|arcsinh| 0-2: {rel(0, 2)}  2-4: {rel(2, 4)}  4-6: {rel(4, 6)}  6+: {rel(6, 99)}")
