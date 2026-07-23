"""CONFIRMATION test: does the NLL encoder's representation advantage (84% linear
probe vs 53% for MSE) translate to the REAL downstream (deconv -> clean charge)?
Freeze each pretrained encoder, train a small head on FULL-context features to
predict charge, eval R^2 on encoder-HELD-OUT events. Same head/data for all."""
import torch, numpy as np, torch.nn as nn
import data as D
D.init_pipeline_cpu()
from model import FMModel
dev = "cuda"
# (ckpt, heads, tag): nll vs mask075 are same arch (d512) differing only in objective; vitb=d768 bonus
MODELS = [("ckpt_sc_w_d768.pt", 4, "NLL-d768 "), ("ckpt_mae_vitb.pt", 12, "MSE-d768 ")]
FIT = list(range(1000, 3000))          # encoder-TRAIN events (head is new -> fine); charge available
EVAL = list(range(0, 250))             # encoder-VAL events (held out from encoder); charge available


def build(sd, heads):
    d = sd["embed.weight"].shape[0]
    enc = len({k.split(".")[1] for k in sd if k.startswith("enc.")})
    decn = len({k.split(".")[1] for k in sd if k.startswith("dec.")})
    cross = any(k == "dec.0.kv.weight" for k in sd)
    nll = sd["val_head.weight"].shape[0] == 2 * 128
    m = FMModel(128, 4, 6, d=d, blocks=enc, dec_blocks=decn, heads=heads, cond="film",
                dec_mode="cross" if cross else "self", nll=nll).to(dev)
    m.load_state_dict(sd); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, d


def get(wi):
    return D.get_cached_charge(f"../artifacts/fm_cache_tpc/ev_{wi:05d}.npz",
                               f"../artifacts/fm_charge_tpc/ev_charge_{wi:05d}.npz", device=dev)


def probe(ckpt, heads, tag):
    m, d = build(torch.load(ckpt, map_location=dev)["model"], heads)
    head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 128)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), 1e-3)
    for step in range(1800):
        wi = FIT[step % len(FIT)]
        B = get(wi)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = m.encode(B).float()
        pred = head(feats)[B["cell"], B["slot"]]
        loss = ((pred - B["target_charge"].float()) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval(); ssr = 0.0; sst = 0.0
    with torch.no_grad():
        for wi in EVAL:
            B = get(wi)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = m.encode(B).float(); pred = head(feats)[B["cell"], B["slot"]].float()
            tg = B["target_charge"].float()
            ssr += ((pred - tg) ** 2).sum().item(); sst += ((tg - tg.mean()) ** 2).sum().item()
    r2 = (1 - ssr / sst) * 100
    print(f"{tag}: deconv->charge  R2 = {r2:.1f}%", flush=True)
    return r2


def main():
    print(f"frozen-encoder deconv probe | fit {len(FIT)} (enc-train) eval {len(EVAL)} (enc-VAL, held out)\n")
    for c, h, t in MODELS:
        probe(c, h, t)


if __name__ == "__main__":
    main()
