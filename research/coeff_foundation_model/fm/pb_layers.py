"""LAYER SWEEP: at which encoder block is the representation richest?

I've only ever probed the FINAL encoder layer (encode()). MAE reps are often best
MID-encoder (final layers specialize for the pretext). Slot-indexed linear probe
(ridge d->128, gather the pixel's slot = the OPTIMAL per-pixel readout, == how val_head
works) to clean coeff, per layer. If a mid layer >> the 89% final-layer baseline, every
probe number so far is a final-layer underestimate."""
import glob, torch, numpy as np
import data as D
D.init_pipeline_cpu()
from model import FMModel
dev = "cuda"; d = 768
LAYERS = [2, 4, 6, 8, 10, 12]
m = FMModel(128, 4, 6, d=d, blocks=12, dec_blocks=4, heads=4, cond="film", dec_mode="cross", nll=True).to(dev)
m.load_state_dict(torch.load("ckpt_sc_w_d768.pt", map_location=dev)["model"]); m.eval()
paths = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))
FIT = paths[20000:20040]; EVAL = paths[20040:20090]
print(f"held-out fit={len(FIT)} eval={len(EVAL)}; slot-indexed ridge probe per layer", flush=True)

XtX = {L: torch.zeros(d, d, device=dev) for L in LAYERS}; XtY = {L: torch.zeros(d, 128, device=dev) for L in LAYERS}
for p in FIT:
    B = D.get_cached(p, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        feats = m.encode_layers(B, set(LAYERS))
    Y = torch.zeros(feats[LAYERS[0]].shape[0], 128, device=dev); Y[B["cell"], B["slot"]] = B["target"].float()
    for L in LAYERS:
        f = feats[L].float(); XtX[L] += f.T @ f; XtY[L] += f.T @ Y
for L in LAYERS:
    W = torch.linalg.solve(XtX[L] + 1e-2 * torch.eye(d, device=dev), XtY[L])
    ssr = sst = 0.0
    for p in EVAL:
        B = D.get_cached(p, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            f = m.encode_layers(B, {L})[L].float()
        pred = (f @ W)[B["cell"], B["slot"]]; tg = B["target"].float()
        ssr += ((pred - tg) ** 2).sum().item(); sst += (tg ** 2).sum().item()
    print(f"  encoder block {L:2d}/12:  probe R2 = {(1 - ssr / sst) * 100:5.1f}%   (final-layer baseline 89.0)", flush=True)
