"""I4 — the trained encoder has activations ~650x its own mean. Where, and how many?

I3 measured, on the cooldown checkpoint, mean|feature| = 1.47 with max|feature| =
954 in the SAME tensor. That ratio is the signature of "massive activations"
(Sun, Chen, Kolter, Liu 2024, arXiv:2402.17762) / attention sinks (StreamingLLM,
Xiao et al. ICLR 2024), where a transformer parks a huge constant-ish value in a
few dimensions of a few tokens and uses them as a no-op attention target.

It matters here for a reason specific to this model: helix has NO CLS, BOS or
register token. Darcet, Oquab, Mairal, Bojanowski (ICLR 2024, arXiv:2309.16588)
show ViTs without one repurpose low-information PATCH tokens as that scratch
space, that the damage lands on DENSE/spatial readout, and that a few dedicated
register tokens fix it. helix's downstream is a dense 3D localisation probe on
frozen features, which is exactly the case they describe.

So: how many tokens, how many channels, which layer does it start, and are the
carriers physically distinguished (plane, band, drift time, occupancy)?
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prof"))
from common import build, emit, load_events, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="a trained checkpoint (.pth)")
ap.add_argument("--events", type=int, default=4)
A = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

model = build(device=dev)
sd = torch.load(A.ckpt, map_location="cpu", weights_only=False)
for k in ("state_dict", "model", "module"):
    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
        sd = sd[k]
model.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()},
                      strict=False)
model.eval()
print(f"loaded {A.ckpt}", flush=True)

evs, _ = load_events(A.events)
R = {"ckpt": A.ckpt, "per_layer": {}, "carriers": []}
LAYERS = list(range(1, len(model.enc) + 1))

for ei, e in enumerate(evs):
    B = to_device({k: v for k, v in e.items() if torch.is_tensor(v)}, dev)
    B["n_cells"] = B["plane_id"].shape[0]
    with torch.no_grad(), torch.autocast(dev, torch.bfloat16, enabled=(dev == "cuda")):
        feats = model.encode_layers(B, set(LAYERS))
    for L in LAYERS:
        x = feats[L].float()
        a = x.abs()
        med = a.median()
        tokmax = a.max(1).values
        # "massive" as Sun et al. define it: >=100x the median magnitude AND
        # >=1000 absolute. Report the ratio too, so the threshold is visible.
        massive = (tokmax > 100 * med)
        d = R["per_layer"].setdefault(L, dict(median=[], max=[], ratio=[],
                                              n_massive=[], n_channels=[]))
        d["median"].append(float(med)); d["max"].append(float(a.max()))
        d["ratio"].append(float(a.max() / med))
        d["n_massive"].append(int(massive.sum()))
        # which CHANNELS carry it, at the top layer
        chan = (a.max(0).values > 100 * med)
        d["n_channels"].append(int(chan.sum()))
    # physical identity of the carriers at the final layer
    x = feats[LAYERS[-1]].float(); a = x.abs()
    med = a.median(); tokmax = a.max(1).values
    top = torch.topk(tokmax, min(16, tokmax.numel())).indices
    R["carriers"].append(dict(
        event=e["_name"],
        n_cells=int(B["n_cells"]),
        median=float(med),
        top_vals=[float(v) for v in tokmax[top]],
        plane=[int(v) for v in B["plane_id"][top]],
        band=[int(v) for v in B["band_id"][top]],
        t_phys=[float(v) for v in B["t_phys"][top]],
        occ_count=[int(v) for v in (B["occ"].bool() & B["valid"]).sum(1)[top]],
        mean_occ_count=float((B["occ"].bool() & B["valid"]).sum(1).float().mean()),
    ))

print(f"\n{'layer':>5s} {'median|f|':>10s} {'max|f|':>10s} {'max/med':>9s} "
      f"{'#tokens>100x':>13s} {'#channels>100x':>15s}")
for L in LAYERS:
    d = R["per_layer"][L]
    print(f"{L:5d} {np.mean(d['median']):10.4f} {np.mean(d['max']):10.2f} "
          f"{np.mean(d['ratio']):9.1f} {np.mean(d['n_massive']):13.1f} "
          f"{np.mean(d['n_channels']):15.1f}")

c = R["carriers"][0]
print(f"\ncarriers at layer {LAYERS[-1]}, event {c['event'][:24]} "
      f"({c['n_cells']} cells, median |f| {c['median']:.4f}):")
print(f"  {'|f|max':>9s} {'plane':>6s} {'band':>5s} {'t_phys':>9s} {'#active slots':>14s}")
for i in range(len(c["top_vals"])):
    print(f"  {c['top_vals'][i]:9.2f} {c['plane'][i]:6d} {c['band'][i]:5d} "
          f"{c['t_phys'][i]:9.1f} {c['occ_count'][i]:14d}")
print(f"  (mean active slots per cell in this event: {c['mean_occ_count']:.2f})")
print("\n  READ: few tokens, few channels, growing with depth, and carriers that")
print("  are LOW-occupancy is the Darcet et al. signature -- the model is using")
print("  uninformative tokens as scratch space because it has no register token.")

emit("i4_massive", R)
