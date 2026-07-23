"""CROSS-PLANE TRIANGULATING head. Tests: is base_fix's weak per-wire 3D (u) because the info is
GONE, or because it's UN-COMBINED across planes? For each target dom pixel (plane p, volume v, drift
time t) we gather the DRIFT-TIME-MATCHED context from the OTHER two planes (same v, |Δt|<=bin) — the
epipolar constraint — as the mean of base_fix's OWN features there, and regress u from
[target 4-band feats | matched plane-q1 pooled feats | matched plane-q2 pooled feats | within-patch pos | plane].
Matching uses the physical drift tick (an observable), NOT true wire (no u leak). Arms:
  solo  = target feats only (no cross)         -> reproduces the per-plane de-patch result (~0.09)
  cross = target + drift-matched cross-plane    -> THE test (does explicit matching recover u?)
  xwire = geo + matched-plane mean TRUE wire    -> geometric matching+triangulation CEILING reference
If base_fix 'cross' jumps to nll_long / toward xwire, the coordinate only limited MATCHING, not the
representation -> triangulate at the head, keep the encoder simple. Fair metric: per-(event,plane) Pearson-r.
Usage: python probe_3d_triangulate.py --ckpt ckpt_nll_base_fix_snap300000.pt --tag base_fix
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from pb_probe import load_event_data, dev, LAB
from pb_aw import fit_alongwire, u_target
from probe_3d_ridge import build, feats_of, fisher_r
from probe_3d_mlp import fit_mlp


def event_tri(evd, feats, cap=6000, tbin=8, seed=0):
    n = evd["n"]; fd = feats.shape[1]
    fpix = torch.zeros((n, 4 * fd), device=feats.device)                 # per-pixel 4-band feature
    for b in range(4):
        idxs, hit = evd["cols"][b]; idxs = idxs.to(feats.device); hit = hit.to(feats.device)
        fb = feats[idxs].clone(); fb[~hit] = 0.0
        fpix[:, b * fd:(b + 1) * fd] = fb
    pooled = fpix.reshape(n, 4, fd).mean(1)                              # band-pooled, for cross context (fd)
    gidp = evd["lb"]["gidp"].numpy().astype(int); vol = gidp // 3; plane = gidp % 3
    t = evd["t"].astype(np.int64); tb = t // tbin; wire = evd["wire"].astype(np.float32)
    ntb = int(tb.max()) + 2
    key = ((vol * 3 + plane).astype(np.int64) * ntb + tb)               # (vol,plane,tbin) cell
    ukey, inv = np.unique(key, return_inverse=True); ng = len(ukey)
    invt = torch.tensor(inv, device=feats.device)
    sump = torch.zeros((ng, fd), device=feats.device).index_add_(0, invt, pooled)
    cnt = torch.zeros(ng, device=feats.device).index_add_(0, invt, torch.ones(n, device=feats.device))
    meanp = (sump / cnt[:, None].clamp(min=1)).cpu().numpy()            # matched-slab mean feature
    sumw = np.zeros(ng); np.add.at(sumw, inv, wire); meanw = sumw / np.maximum(np.bincount(inv, minlength=ng), 1)
    krow = {int(k): i for i, k in enumerate(ukey)}
    fpix_cpu = fpix.cpu().numpy(); del fpix, pooled; torch.cuda.empty_cache()

    dom = np.where(evd["lb"]["dom"].numpy().astype(bool))[0]
    if len(dom) > cap: dom = np.sort(np.random.RandomState(seed).choice(dom, cap, replace=False))
    k = len(dom)
    cross = np.zeros((k, 2 * fd), np.float32); xwire = np.zeros((k, 2), np.float32)
    for oi, off in enumerate([1, 2]):
        qkey = ((vol[dom] * 3 + (plane[dom] + off) % 3).astype(np.int64) * ntb + tb[dom])
        for i, kk in enumerate(qkey):
            r = krow.get(int(kk), -1)
            if r >= 0:
                cross[i, oi * fd:(oi + 1) * fd] = meanp[r]; xwire[i, oi] = meanw[r] / 2000.0
    geo = np.concatenate([evd["phase"].numpy().astype(np.float32)[dom], np.eye(6, dtype=np.float32)[gidp[dom]]], 1)
    return dict(target=fpix_cpu[dom], cross=cross, xwire=xwire, geo=geo,
                u=evd["u"].numpy().astype(np.float32)[dom], plane=gidp[dom])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", default="ref")
    ap.add_argument("--layer", type=int, default=12); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--events", default="30000-30079"); ap.add_argument("--out", default="probe_3d_tri.jsonl")
    a = ap.parse_args()
    lo, hi = map(int, a.events.split("-"))
    evs = [e for e in range(lo, hi + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    print(f"[{a.tag}] loading {len(evs)} events, layer {a.layer}", flush=True)
    raw = [load_event_data(e) for e in evs]
    aw = fit_alongwire(raw)
    for r in raw: r["u"] = u_target(r, aw)
    mt = build(a.ckpt, randinit=False)
    packs = [event_tri(r, feats_of(mt, r["B"], a.layer), cap=a.cap) for r in raw]
    del mt; torch.cuda.empty_cache()

    nev = len(packs); ntr = int(nev * 0.75)
    ev = np.concatenate([np.full(len(p["u"]), i) for i, p in enumerate(packs)])
    plane = np.concatenate([p["plane"] for p in packs]); y = np.concatenate([p["u"] for p in packs])
    geo = np.concatenate([p["geo"] for p in packs]); tgt = np.concatenate([p["target"] for p in packs])
    crs = np.concatenate([p["cross"] for p in packs]); xw = np.concatenate([p["xwire"] for p in packs])
    tr = ev < ntr; te = ev >= ntr
    yg = torch.tensor(y, dtype=torch.float32, device=dev)
    designs = {"solo":  np.concatenate([tgt, geo], 1),
               "cross": np.concatenate([tgt, crs, geo], 1),
               "xwire": np.concatenate([xw, geo], 1)}                    # geometric matched-wire ceiling
    res = {"tag": a.tag, "layer": a.layer, "probe": "triangulate", "n": int(len(y)), "cap": a.cap}
    print(f"[{a.tag}] {len(y)} target pixels", flush=True)
    for name, X in designs.items():
        Xtr = torch.tensor(X[tr], dtype=torch.float32, device=dev); Xev = torch.tensor(X[te], dtype=torch.float32, device=dev)
        rs = [fisher_r(y[te], fit_mlp(Xtr, yg[tr], Xev, s), ev[te], plane[te])[0] for s in range(a.seeds)]
        del Xtr, Xev; torch.cuda.empty_cache()
        res[name] = round(float(np.mean(rs)), 4); res[name + "_sd"] = round(float(np.std(rs)), 4)
        print(f"  {name:6s}: u_r = {np.mean(rs):+.4f} ± {np.std(rs):.4f}", flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"WROTE {a.tag}", flush=True)


if __name__ == "__main__":
    main()
