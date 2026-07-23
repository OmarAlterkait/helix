"""NONLINEAR fair-probe check (secondary to probe_3d_ridge). Same fair harness — PATCH
granularity, GEO-only null, per-(event,plane) Pearson-r metric, event train/eval split — but a
SMALL, strongly-regularized MLP with a FIXED step budget (no early-stop-on-oracle) + multi-seed.
Answers: is u NONLINEARLY decodable from the features (which ridge would miss)? If trained_r
stays ≈ geo/random here too, the weak-3D conclusion is robust to nonlinearity.
Usage: python probe_3d_mlp.py --ckpt ckpt_nll_base_fix_snap300000.pt --tag base_fix
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from pb_probe import load_event_data, dev, LAB
from pb_aw import fit_alongwire, u_target
from probe_3d_ridge import build, feats_of, event_patches, fisher_r


def build_serial(ckpt, randinit, rope_split, gp, gd, use_ema=False):
    """Build the grouped-serial encoder (matching how these ckpts were trained) instead of full attn."""
    from model_serial import SerialFMModel
    ck = torch.load(ckpt, map_location=dev)
    nslot = ck.get("n_slot", ck.get("pw", 16) * ck.get("pt", 8))
    heads = ck["d"] // ck["head_dim"] if ck.get("head_dim", 0) else ck.get("heads", 8)
    m = SerialFMModel(nslot, 4, 6, d=ck.get("d", 512), blocks=ck.get("blocks", 12), dec_blocks=ck.get("dec_blocks", 4),
                      heads=heads, cond=ck.get("cond", "film"), dec_mode=ck.get("dec_mode", "cross"),
                      nll=ck.get("nll", False), mup=ck.get("mup", True), d_base=ck.get("d_base", 128),
                      wire_rope=ck.get("wire_rope", True), rope_split=bool(rope_split), gp=gp, gd=gd).to(dev)
    if not randinit:
        sd = ck["ema"] if (use_ema and "ema" in ck) else ck["model"]
        if use_ema and "ema" not in ck: print("WARN: --use_ema but no ema in ckpt; using model", flush=True)
        m.load_state_dict(sd)
    m.eval(); return m


class SmallMLP(nn.Module):
    def __init__(self, din, h=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(h, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, x): return self.net(x).squeeze(-1)


def fit_mlp(Xtr, ytr, Xev, seed, steps=3000, bs=8192, wd=1e-2):
    torch.manual_seed(seed); np.random.seed(seed)
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xt = (Xtr - mu) / sd; Xe = (Xev - mu) / sd; ym = ytr.mean()
    net = SmallMLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=wd)
    n = Xt.shape[0]
    for s in range(steps):
        idx = torch.randint(0, n, (bs,), device=dev)
        pred = net(Xt[idx])
        loss = ((pred - (ytr[idx] - ym)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return (net(Xe) + ym).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", default="ref")
    ap.add_argument("--layer", type=int, default=12); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--events", default="30000-30079"); ap.add_argument("--out", default="probe_3d_mlp.jsonl")
    ap.add_argument("--serial", type=int, default=0); ap.add_argument("--rope_split", type=int, default=1)
    ap.add_argument("--use_ema", type=int, default=0)
    ap.add_argument("--gp", type=int, default=1024); ap.add_argument("--gd", type=int, default=2048)
    a = ap.parse_args()
    lo, hi = map(int, a.events.split("-"))
    evs = [e for e in range(lo, hi + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    print(f"[{a.tag}] loading {len(evs)} events, layer {a.layer}", flush=True)
    raw = [load_event_data(e) for e in evs]
    aw = fit_alongwire(raw)
    for r in raw: r["u"] = u_target(r, aw)
    if a.serial:
        mt = build_serial(a.ckpt, False, a.rope_split, a.gp, a.gd, use_ema=bool(a.use_ema)); mr = build_serial(a.ckpt, True, a.rope_split, a.gp, a.gd)
    else:
        mt = build(a.ckpt, randinit=False); mr = build(a.ckpt, randinit=True)
    packs = []
    for r in raw:
        ft = feats_of(mt, r["B"], a.layer); fr = feats_of(mr, r["B"], a.layer)
        packs.append(event_patches(r, ft, fr, a.layer))
    del mt, mr; torch.cuda.empty_cache()

    nev = len(packs); ntr = int(nev * 0.75)                       # 60 train / 20 eval event split
    ev = np.concatenate([np.full(len(p["y"]), i) for i, p in enumerate(packs)])
    plane = np.concatenate([p["plane"] for p in packs]); y = np.concatenate([p["y"] for p in packs])
    tr_mask = ev < ntr; ev_mask = ev >= ntr
    geo = np.concatenate([p["geo"] for p in packs])
    arms = {"trained": np.concatenate([p["Xtr"] for p in packs]),
            "random":  np.concatenate([p["Xrn"] for p in packs]),
            "raw":     np.concatenate([p["Xraw"] for p in packs])}
    yg = torch.tensor(y, dtype=torch.float32, device=dev)
    res = {"tag": a.tag, "layer": a.layer, "n_patch": int(len(y)), "probe": "mlp"}

    def run(design, name):
        Xtr = torch.tensor(design[tr_mask], dtype=torch.float32, device=dev)
        Xev = torch.tensor(design[ev_mask], dtype=torch.float32, device=dev)
        rs = []
        for sd in range(a.seeds):
            p = fit_mlp(Xtr, yg[tr_mask], Xev, sd)
            fr, _ = fisher_r(y[ev_mask], p, ev[ev_mask], plane[ev_mask]); rs.append(fr)
        return float(np.mean(rs)), float(np.std(rs))

    gr, gs = run(geo, "geo"); res["geo"] = {"fisher_r": round(gr, 4), "std": round(gs, 4)}
    print(f"  geo-only: fisher_r={gr:+.4f}±{gs:.4f}", flush=True)
    for name, Xf in arms.items():
        r_, s_ = run(np.concatenate([Xf, geo], 1), name)
        res[name] = {"fisher_r": round(r_, 4), "std": round(s_, 4), "d_over_geo_r": round(r_ - gr, 4)}
        print(f"  {name:8s}: fisher_r={r_:+.4f}±{s_:.4f}  Δr_over_geo={r_-gr:+.4f}", flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"WROTE {a.tag}", flush=True)


if __name__ == "__main__":
    main()
