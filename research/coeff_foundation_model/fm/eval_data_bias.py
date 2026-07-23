"""Test the eval-bias hypothesis: the 20k floor eval is run_766-only, which is the
20k model's whole training distribution. Eval BOTH checkpoints (20k-trained vs
50k-trained) on each run separately. Prediction: 20k model strong on 766 but drops
on 767/768 (OOD); 50k model (trained on 766+767+768) generalizes across all.
"""
import sys, os, glob, numpy as np, torch
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from data import DEV, N_SLOT, N_BAND, N_PLANE
from model import FMModel
from train import perband_mse, TGT_VAR_BAND

D.init_pipeline_cpu()
CDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "fm_cache_tpc"))


def ve(pbd):
    return float(np.mean([(TGT_VAR_BAND[b] - pbd[f"b{b}"]) / TGT_VAR_BAND[b] for b in range(N_BAND)]))


def load(ck):
    c = torch.load(os.path.join(os.path.dirname(__file__), ck), map_location=DEV)
    m = FMModel(N_SLOT, N_BAND, N_PLANE, n_wirefeat=1, d=c["d"], blocks=c["blocks"],
                dec_blocks=c["dec_blocks"], cond="film").to(DEV)
    m.load_state_dict(c["model"]); m.eval()
    return m, c["step"]


def evalset(model, lo, n=40):
    fs = [f"{CDIR}/ev_{i:05d}.npz" for i in range(lo, lo + 2000) if os.path.exists(f"{CDIR}/ev_{i:05d}.npz")][:n]
    prm, pbd, bse, pvm, pvb = perband_mse(model, fs, "random", 0.75, 1)
    return ve(pbd) * 100, {b: (TGT_VAR_BAND[b] - pbd[f"b{b}"]) / TGT_VAR_BAND[b] * 100 for b in range(N_BAND)}


ranges = {"766 (train: both)": 0, "767 (train: only 50k)": 20000, "768 (train: only 50k)": 40000}
VB = ["A4", "D4", "D3", "D2"]
for name, ck in [("20k-trained", "ckpt_mae_ddp_long.pt"), ("50k-trained", "ckpt_mae_10L_50k.pt")]:
    model, step = load(ck)
    print(f"\n=== {name} ({ck}, step {step}) ===")
    for rn, lo in ranges.items():
        ov, pb = evalset(model, lo)
        print(f"  run_{rn:>22}: var_expl={ov:5.1f}%  | " + " ".join(f"{VB[b]}={pb[b]:.0f}" for b in range(N_BAND)))
