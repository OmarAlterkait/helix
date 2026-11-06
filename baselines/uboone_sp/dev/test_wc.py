import sys, time, json, numpy as np
sys.path.insert(0, "baselines/uboone_sp")
from sp.data import Event, registry, noisy_planes
from sp.noise_filter import gate_clean
from sp.pipeline import Kernel
from sp.response import load_kernel
from sp import wirecell_sp as WC
from roi_diag import window_flags
reg = registry(); ev = int(sys.argv[1]); E = Event(ev); rng = np.random.default_rng(1000 + ev)
for g in range(6):
    pl = "UVY"[g % 3]
    C = E.clean(g, reg); T = E.truth(g, C.shape)
    inc, coh = noisy_planes(C, np.asarray(reg[g]["wire_lengths"]), reg[g]["pedestal"], rng)
    if g not in (0, 1, 2): continue
    M = gate_clean(coh)
    t0 = time.time(); d = WC.decon_outputs(M, pl, Kernel(*load_kernel(pl, "fit"))); t1 = time.time()
    st = {}
    rois, rms, keep = WC.roi_chain(d, pl, stages=True); t2 = time.time()
    q, mask = WC.apply_roi(d["gauss"], rois, pl); t3 = time.time()
    bg = [(m["w0"], m["t0"]) for m in E.meta if m["kind"] == "bg" and m["g"] == g]
    iso = [(m["w0"], m["t0"]) for m in E.meta if m["kind"] == "iso" and m["g"] == g]
    print(f"{pl}: decon {t1-t0:.1f}s chain {t2-t1:.1f}s apply {t3-t2:.1f}s | rms median {np.median(rms):.0f} | ROI pixel frac {mask.mean():.4f} | charge/hits {q.sum()/T.sum():.3f} | truth in ROI {T[mask].sum()/T.sum():.3f}")
    for k, v in keep.items():
        mk = WC.rois_mask(v, C.shape)
        print(f"   {k:14s} nROI {sum(len(x) for x in v):6d} pix {mk.mean():.4f} bg-hit {np.mean(window_flags(mk, bg)) if bg else float('nan'):.3f} iso-hit {np.mean(window_flags(mk, iso)) if iso else float('nan'):.3f}")
