"""LOAD-ONCE, PROBE-MANY 3D sweep. Fixes the Category-B pattern: instead of a fresh
pb_aw subprocess per (ckpt,layer) — each re-loading + re-assembling the same 80 events
from the shared FS (redundant + contention) — this loads the events ONCE, then for each
checkpoint does a single all-layers forward and fits the along-wire 3D probe per layer.

  python probe_sweep.py --ckpts nll_data:ckpt_nll_data_snap150000.pt:1  mse_long:...:0 ...
      --layers 4,8,12  --steps 1500  --out probe_sweep_results.jsonl
Each --ckpts entry is  tag:ckpt_path:mnll  (mnll=1 if the model has an NLL head)."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from pb_probe import load_event_data, readout, LAB, dev
from pb_aw import fit_alongwire, u_target, UProbe, evaluate
from model import FMModel


def build(ckpt):
    ck = torch.load(ckpt, map_location=dev)
    m = FMModel(128, 4, 6, d=ck["d"], blocks=ck["blocks"], dec_blocks=ck.get("dec_blocks", 4),
                heads=ck["heads"], cond=ck.get("cond", "film"), dec_mode=ck.get("dec_mode", "cross"),
                nll=ck.get("nll", False)).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["blocks"]


def fit_eval(raw_tr, raw_ev, steps, px):
    """pb_aw's along-wire probe fit+eval, arm='nll', on cached feats (evd['feats'] preset)."""
    fdim = raw_tr[0]["fdim"]
    smp = torch.cat([e["feats"][:4096] for e in raw_tr[:10]]); mu = smp.mean(0).to(dev); sd = (smp.std(0) + 1e-6).to(dev)
    miss = nn.Parameter(torch.zeros(4, fdim, device=dev))
    probe = UProbe(fdim * 4, 4 + 5 + 6 + 2).to(dev)
    opt = torch.optim.AdamW(list(probe.parameters()) + [miss], 1e-3, weight_decay=1e-5)
    for step in range(1, steps + 1):
        evd = raw_tr[np.random.randint(len(raw_tr))]
        sel = torch.tensor(np.random.randint(0, evd["n"], min(px, evd["n"])))
        Xf, Xg = readout(evd, miss, "nll", sel, mu, sd)
        pred = probe(Xf, Xg)
        dom = evd["lb"]["dom"][sel].to(dev)
        loss = ((pred - evd["u"][sel].to(dev)) ** 2 * dom).sum() / max(dom.sum(), 1)
        opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(probe, raw_ev, miss, "nll", mu, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)         # tag:path:mnll
    ap.add_argument("--layers", default="4,8,12")
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--px", type=int, default=8192)
    ap.add_argument("--train", default="30000-30059"); ap.add_argument("--eval", default="30060-30079")
    ap.add_argument("--out", default="probe_sweep_results.jsonl")
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    lo, hi = map(int, a.train.split("-")); el, eh = map(int, a.eval.split("-"))
    tr = [e for e in range(lo, hi + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    ev = [e for e in range(el, eh + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]

    # ---- STAGE 0: load + assemble the events ONCE (the expensive, previously-redundant part) ----
    print(f"loading {len(tr)}+{len(ev)} events once (assemble on GPU, reused for every ckpt/layer)...", flush=True)
    raw_tr = [load_event_data(e) for e in tr]
    raw_ev = [load_event_data(e) for e in ev]
    aw = fit_alongwire(raw_tr)
    for r in raw_tr + raw_ev:
        r["u"] = u_target(r, aw)
    want = [int(x) for x in a.layers.split(",")]
    outf = open(a.out, "a")

    for spec in a.ckpts:
        tag, ckpt, mnll = spec.split(":")
        model, blocks = build(ckpt)
        layers = sorted(set([L for L in want if L <= blocks] + [blocks]))    # valid layers + the final
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # ONE all-layers forward per event
            ft = [model.encode_layers(r["B"], set(layers)) for r in raw_tr]
            fe = [model.encode_layers(r["B"], set(layers)) for r in raw_ev]
        for L in layers:
            for r, f in zip(raw_tr, ft):
                r["feats"] = f[L].float(); r["fdim"] = f[L].shape[1]
            for r, f in zip(raw_ev, fe):
                r["feats"] = f[L].float(); r["fdim"] = f[L].shape[1]
            res = fit_eval(raw_tr, raw_ev, a.steps, a.px)
            rec = dict(tag=tag, layer=L, blocks=blocks, within_plane=res["within_plane"],
                       v0Y=res.get("v0Y"), v1Y=res.get("v1Y"), pooled=res["overall_pooled_DILUTED"])
            print(f"  {tag:14s} L{L:>2}: within_plane={res['within_plane']:+.3f}", flush=True)
            outf.write(json.dumps(rec) + "\n"); outf.flush()
        del model, ft, fe; torch.cuda.empty_cache()
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
