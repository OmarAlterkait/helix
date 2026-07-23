"""SELECTIVE cross-plane triangulating head. The mean-pool 'cross' arm (probe_3d_triangulate)
crippled itself: RoPE-encoded wire features do NOT average (averaging rotations destroys position),
so mean-pooling the slab blurred the partner's wire -> only recovered u to 0.13 while the mean true-WIRE
scalar reached 0.70. Fix: SELECT the single drift-time-nearest partner in each other plane and read ITS
feature un-averaged. Arms:
  solo    = target feats only
  sel     = target + drift-time-NEAREST partner features (q1,q2)   -> THE test (selection vs pooling)
  selwire = geo + nearest partner TRUE wire                         -> selective geometric ceiling
Match uses physical drift tick (observable), never true wire in the feature arms (no u leak).
If base_fix 'sel' climbs from 0.13 toward the ceiling, a SELECTIVE (attention) cross-plane head recovers
3D from the frozen encoder -> build the two-tier epipolar attention, keep the encoder simple.
Usage: python probe_3d_select.py --ckpt ckpt_nll_base_fix_snap300000.pt --tag base_fix
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from pb_probe import load_event_data, dev, LAB
from pb_aw import fit_alongwire, u_target
from probe_3d_ridge import build, feats_of, fisher_r
from probe_3d_mlp import fit_mlp


def event_select(evd, feats, cap=6000, seed=0):
    n = evd["n"]; fd = feats.shape[1]
    fpix = torch.zeros((n, 4 * fd), device=feats.device)
    for b in range(4):
        idxs, hit = evd["cols"][b]; idxs = idxs.to(feats.device); hit = hit.to(feats.device)
        fb = feats[idxs].clone(); fb[~hit] = 0.0
        fpix[:, b * fd:(b + 1) * fd] = fb
    fpix_cpu = fpix.cpu().numpy(); del fpix; torch.cuda.empty_cache()     # FULL 4-band partner (no band-pool)
    FD = 4 * fd
    gidp = evd["lb"]["gidp"].numpy().astype(int); vol = gidp // 3; plane = gidp % 3
    t = evd["t"].astype(np.float64); wire = evd["wire"].astype(np.float32)
    vp = {}                                                              # (vol,plane) -> (sorted t, pixel idx)
    for v in range(2):
        for p in range(3):
            s = np.where((vol == v) & (plane == p))[0]
            if len(s): vp[(v, p)] = (t[s][np.argsort(t[s])], s[np.argsort(t[s])])
    dom = np.where(evd["lb"]["dom"].numpy().astype(bool))[0]
    if len(dom) > cap: dom = np.sort(np.random.RandomState(seed).choice(dom, cap, replace=False))
    k = len(dom)
    selfeat = np.zeros((k, 2 * FD), np.float32); selwire = np.zeros((k, 2), np.float32)
    for oi, off in enumerate([1, 2]):
        for i, ti in enumerate(dom):
            key = (vol[ti], (plane[ti] + off) % 3)
            if key not in vp: continue
            tt, order = vp[key]
            j = int(np.searchsorted(tt, t[ti]))
            cands = [c for c in (j - 1, j) if 0 <= c < len(tt)]
            if not cands: continue
            pi = order[min(cands, key=lambda c: abs(tt[c] - t[ti]))]     # drift-time-NEAREST partner
            selfeat[i, oi * FD:(oi + 1) * FD] = fpix_cpu[pi]; selwire[i, oi] = wire[pi] / 2000.0
    geo = np.concatenate([evd["phase"].numpy().astype(np.float32)[dom], np.eye(6, dtype=np.float32)[gidp[dom]]], 1)
    return dict(target=fpix_cpu[dom], selfeat=selfeat, selwire=selwire, geo=geo,
                u=evd["u"].numpy().astype(np.float32)[dom], plane=gidp[dom])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", default="ref")
    ap.add_argument("--layer", type=int, default=12); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--events", default="30000-30079"); ap.add_argument("--out", default="probe_3d_sel.jsonl")
    a = ap.parse_args()
    lo, hi = map(int, a.events.split("-"))
    evs = [e for e in range(lo, hi + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    print(f"[{a.tag}] loading {len(evs)} events", flush=True)
    raw = [load_event_data(e) for e in evs]
    aw = fit_alongwire(raw)
    for r in raw: r["u"] = u_target(r, aw)
    mt = build(a.ckpt, randinit=False)
    packs = [event_select(r, feats_of(mt, r["B"], a.layer), cap=a.cap) for r in raw]
    del mt; torch.cuda.empty_cache()

    nev = len(packs); ntr = int(nev * 0.75)
    ev = np.concatenate([np.full(len(p["u"]), i) for i, p in enumerate(packs)])
    plane = np.concatenate([p["plane"] for p in packs]); y = np.concatenate([p["u"] for p in packs])
    geo = np.concatenate([p["geo"] for p in packs]); tgt = np.concatenate([p["target"] for p in packs])
    sf = np.concatenate([p["selfeat"] for p in packs]); sw = np.concatenate([p["selwire"] for p in packs])
    tr = ev < ntr; te = ev >= ntr
    yg = torch.tensor(y, dtype=torch.float32, device=dev)
    designs = {"solo": np.concatenate([tgt, geo], 1),
               "sel": np.concatenate([tgt, sf, geo], 1),
               "selwire": np.concatenate([sw, geo], 1)}
    res = {"tag": a.tag, "probe": "select", "n": int(len(y)), "cap": a.cap}
    print(f"[{a.tag}] {len(y)} targets", flush=True)
    for name, X in designs.items():
        Xtr = torch.tensor(X[tr], dtype=torch.float32, device=dev); Xev = torch.tensor(X[te], dtype=torch.float32, device=dev)
        rs = [fisher_r(y[te], fit_mlp(Xtr, yg[tr], Xev, s), ev[te], plane[te])[0] for s in range(a.seeds)]
        del Xtr, Xev; torch.cuda.empty_cache()
        res[name] = round(float(np.mean(rs)), 4); res[name + "_sd"] = round(float(np.std(rs)), 4)
        print(f"  {name:8s}: u_r = {np.mean(rs):+.4f} ± {np.std(rs):.4f}", flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"WROTE {a.tag}", flush=True)


if __name__ == "__main__":
    main()
