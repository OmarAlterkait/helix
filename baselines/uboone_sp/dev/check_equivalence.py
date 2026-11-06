"""Verify sp.pipeline reproduces the example's own code for a given kernel:
deconvolution outputs, LF filters, and the full run_pipeline (ROIs + charge),
on one real plane of our sim with the example's kernel, both DC modes."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sp.data import Event, registry  # noqa: E402
from sp.pipeline import Kernel, apply_lf_filter_to_wiener, deconvolve_2d_dual, run_sp  # noqa: E402
from sp.response import load_kernel  # noqa: E402
from uboone_pipeline import deconv as ref_deconv  # noqa: E402  (example)
from uboone_pipeline.pipeline import run_pipeline as ref_run  # noqa: E402

reg = registry()
E = Event(260)
for g in (3, 5):
    pl = "UVY"[g % 3]
    M = E.clean(g, reg)
    T = E.truth(g, M.shape)
    k = Kernel(*load_kernel(pl, "example"))
    for noiseless in (True, False):
        cw, cg = deconvolve_2d_dual(M, pl, k, noiseless=noiseless)
        rw, rg, _ = ref_deconv.deconvolve_2d_dual(M, pl, noiseless=noiseless)
        print(f"{pl} noiseless={noiseless}: max|dW| {np.abs(cw - rw).max():.3g} (scale {np.abs(rw).max():.3g}), "
              f"max|dG| {np.abs(cg - rg).max():.3g} (scale {np.abs(rg).max():.3g})")
        if pl == "U":
            for which in ("loose", "tight"):
                a = apply_lf_filter_to_wiener(cw, pl, which=which)
                b = ref_deconv.apply_lf_filter_to_wiener(rw, pl, which=which)
                print(f"   LF {which}: max diff {np.abs(a - b).max():.3g} (scale {np.abs(b).max():.3g})")
        for true_roi in (True, False):
            mine = run_sp(M, pl, k, truth=T, use_true_roi=true_roi, noiseless=noiseless)
            ref = ref_run(M, T, pl, use_true_roi=true_roi, noiseless=noiseless)
            same_rois = mine["rois"] == ref.rois
            print(f"   run (true_roi={true_roi}): identical ROIs {same_rois} "
                  f"(n wires {len(ref.rois)}), max|dC_charge| {np.abs(mine['C_charge'] - ref.C_charge).max():.3g}, "
                  f"mask agree {np.mean(mine['roi_mask'] == ref.roi_mask):.6f}")
