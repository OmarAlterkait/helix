"""Eval a ckpt under PLANE masking (hide a whole plane -> forces cross-plane triangulation) vs
RANDOM masking, reusing the trainer's exact make_mask + perband_mse. Compares grouped-serial vs full."""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import data as D; D.init_pipeline_cpu()
from train import perband_mse, TGT_VAR_BAND
from probe_3d_ridge import build
from probe_3d_mlp import build_serial

N_BAND = 4
def ve(pbd): return float(np.mean([(TGT_VAR_BAND[b] - pbd[f"b{b}"]) / TGT_VAR_BAND[b] for b in range(N_BAND)]))

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", default="ref")
ap.add_argument("--serial", type=int, default=0); ap.add_argument("--rope_split", type=int, default=0)
ap.add_argument("--eval_n", type=int, default=60)
a = ap.parse_args()

m = build_serial(a.ckpt, False, a.rope_split, 1024, 2048) if a.serial else build(a.ckpt, False)
cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/fm_cache_tpc")
files = sorted(glob.glob(os.path.join(cdir, "ev_*.npz")),
               key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p)))))
ev = files[:a.eval_n]                                  # matches trainer's held-out eval set (first eval_n of files[:val_n])
print(f"[{a.tag}] ckpt={os.path.basename(a.ckpt)} serial={a.serial} eval_n={len(ev)}", flush=True)
for mode, ratio, npl in [("random", 0.75, 1), ("plane", 0.75, 1)]:
    prm, pbd, bse, pvm, pvb = perband_mse(m, ev, mode, ratio, npl)
    print(f"  {mode:6s} n_planes={npl}: MASKED var_expl={ve(pbd)*100:5.1f}%  mse={prm:.3f}   "
          f"(visible var_expl={ve(pvb)*100:.1f}%)", flush=True)
