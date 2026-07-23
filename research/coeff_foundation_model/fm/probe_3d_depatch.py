"""DE-PATCHIFYING per-wire head (the 'in-practice' readout). Standard dense-prediction pattern:
frozen PATCHED encoder -> per-pixel decode. For each active pixel we take its 4-band PATCH features
+ its WITHIN-PATCH slot position (phase = wire-in-patch, tick-in-patch/band) + plane, and predict a
PER-WIRE value (u = cross-plane along-wire 3D, or fine drift-time t). No absolute wire/t is given —
so this purely tests whether the patch features, once de-patchified to per-wire, carry 3D.
Fair harness: nonlinear MLP head, per-(event,plane) Pearson-r, geo-only null, event train/eval split.
Compares base_fix (canonical coord) vs nll_long (survivor-max) to test the readout-granularity thesis.
Usage: python probe_3d_depatch.py --ckpt ckpt_nll_base_fix_snap300000.pt --tag base_fix
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from pb_probe import load_event_data, dev, LAB
from pb_aw import fit_alongwire, u_target
from probe_3d_ridge import build, feats_of, fisher_r
from probe_3d_mlp import SmallMLP, fit_mlp


def hit_design(evd, feats_tr, feats_rn, cap=15000, seed=0):
    """Per-PIXEL de-patchified inputs (DOM pixels only, subsampled to `cap`/event to bound host RAM):
    4-band patch feats (missing->0) + phase(5) + plane(6). NO absolute wire/t."""
    dom = np.where(evd["lb"]["dom"].numpy().astype(bool))[0]
    if len(dom) > cap:
        dom = np.sort(np.random.RandomState(seed).choice(dom, cap, replace=False))
    seln = torch.tensor(dom, dtype=torch.long)
    def gather(feats):
        fd = feats.shape[1]
        X = torch.zeros((len(dom), 4 * fd), device=feats.device)
        for b in range(4):
            idxs, hit = evd["cols"][b]
            idxs = idxs[seln].to(feats.device); hit = hit[seln].to(feats.device)
            fb = feats[idxs].clone(); fb[~hit] = 0.0
            X[:, b * fd:(b + 1) * fd] = fb
        return X.cpu().numpy().astype(np.float32)
    Xtr = gather(feats_tr); Xrn = gather(feats_rn); Xraw = gather(evd["B"]["inp"].float())
    phase = evd["phase"].numpy().astype(np.float32)[dom]                  # (k,5) within-patch slot pos
    gidp = evd["lb"]["gidp"].numpy().astype(int)[dom]
    geo = np.concatenate([phase, np.eye(6, dtype=np.float32)[gidp]], 1)   # NO absolute wire/t
    return dict(Xtr=Xtr, Xrn=Xrn, Xraw=Xraw, geo=geo, plane=gidp,
                u=evd["u"].numpy().astype(np.float32)[dom],
                t=(evd["t"].astype(np.float32) / 4321.0)[dom],            # per-wire drift time (norm)
                dom=np.ones(len(dom), bool))                             # all kept pixels are dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", default="ref")
    ap.add_argument("--layer", type=int, default=12); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--events", default="30000-30079"); ap.add_argument("--out", default="probe_3d_depatch.jsonl")
    a = ap.parse_args()
    lo, hi = map(int, a.events.split("-"))
    evs = [e for e in range(lo, hi + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    print(f"[{a.tag}] loading {len(evs)} events, layer {a.layer}", flush=True)
    raw = [load_event_data(e) for e in evs]
    aw = fit_alongwire(raw)
    for r in raw: r["u"] = u_target(r, aw)
    mt = build(a.ckpt, randinit=False); mr = build(a.ckpt, randinit=True)
    packs = [hit_design(r, feats_of(mt, r["B"], a.layer), feats_of(mr, r["B"], a.layer)) for r in raw]
    del mt, mr; torch.cuda.empty_cache()

    nev = len(packs); ntr = int(nev * 0.75)
    ev = np.concatenate([np.full(len(p["u"]), i) for i, p in enumerate(packs)])
    plane = np.concatenate([p["plane"] for p in packs])
    dom = np.concatenate([p["dom"] for p in packs])
    geo = np.concatenate([p["geo"] for p in packs])
    arms = {"trained": np.concatenate([p["Xtr"] for p in packs]),
            "random":  np.concatenate([p["Xrn"] for p in packs]),
            "raw":     np.concatenate([p["Xraw"] for p in packs])}
    tr = (ev < ntr) & dom; te = (ev >= ntr) & dom                        # DOM pixels only (scored population)

    res = {"tag": a.tag, "layer": a.layer, "probe": "depatch_perwire", "n_dom": int(dom.sum())}
    for tgt_name, y in [("u", np.concatenate([p["u"] for p in packs])),
                        ("t", np.concatenate([p["t"] for p in packs]))]:
        yg = torch.tensor(y, dtype=torch.float32, device=dev)
        def run(design):
            Xtr = torch.tensor(design[tr], dtype=torch.float32, device=dev)
            Xev = torch.tensor(design[te], dtype=torch.float32, device=dev)
            rs = [fisher_r(y[te], fit_mlp(Xtr, yg[tr], Xev, s), ev[te], plane[te])[0] for s in range(a.seeds)]
            return float(np.mean(rs)), float(np.std(rs))
        gr, gs = run(geo)
        res[tgt_name] = {"geo": round(gr, 4)}
        print(f"[{a.tag}] target={tgt_name}: geo={gr:+.4f}±{gs:.4f}", flush=True)
        for name, Xf in arms.items():
            r_, s_ = run(np.concatenate([Xf, geo], 1))
            res[tgt_name][name] = round(r_, 4); res[tgt_name][name + "_d"] = round(r_ - gr, 4)
            print(f"    {name:8s}: r={r_:+.4f}±{s_:.4f}  Δ_over_geo={r_-gr:+.4f}", flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"WROTE {a.tag}", flush=True)


if __name__ == "__main__":
    main()
