import sys, numpy as np
sys.path.insert(0, "baselines/uboone_sp")
from sp.data import Event, registry
from sp.pipeline import Kernel, run_sp, fine_cells
from sp.response import load_kernel
ev, g, w0, t0 = (int(x) for x in sys.argv[1:5])
reg = registry(); E = Event(ev); pl = "UVY"[g % 3]
C = E.clean(g, reg); T = E.truth(g, C.shape)
K = Kernel(*load_kernel(pl, "fit"))
r = run_sp(C, pl, K, truth=T, use_true_roi=True, noiseless=True)
sl = (slice(w0, w0 + 32), slice(t0, t0 + 128))
np.set_printoptions(linewidth=250, precision=0, suppress=True)
print("truth per wire:", T[sl].sum(1))
print("clean |max| per wire:", np.abs(C[sl]).max(1))
print("mask ticks per wire:", r["roi_mask"][sl].sum(1))
print("C_gauss sum per wire (all window ticks):", r["C_gauss"][sl].sum(1))
print("C_gauss sum per wire in mask:", (r["C_gauss"] * r["roi_mask"])[sl].sum(1))
print("C_charge sum per wire:", r["C_charge"][sl].sum(1))
X = fine_cells(r["C_charge"]); print("cell max in window", X[w0//2:w0//2+16, t0//16:t0//16+8].max())
w = int(np.argmax(T[sl].sum(1))) + w0
tt = np.nonzero(T[w])[0]; print("wire", w, "truth ticks", tt.min(), tt.max(), "truth", T[w, tt.min()-5:tt.max()+6])
a, b = tt.min() - 15, tt.max() + 16
print("clean  ", C[w, a:b]); print("C_gauss", r["C_gauss"][w, a:b]); print("mask   ", r["roi_mask"][w, a:b].astype(int)); print("C_charge", r["C_charge"][w, a:b])
