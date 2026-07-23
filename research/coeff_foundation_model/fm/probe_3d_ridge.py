"""FAIR 3D probe (per 3-Fable-agent review). Fixes the metric artifacts that made the
per-coefficient MLP probe uninterpretable (scores below the oracle-zeroed floor measured
probe MEMORIZATION, not u-content). Prescription:
  - CLOSED-FORM RIDGE (linear), deterministic, no early-stopping/seed variance.
  - PATCH granularity: one row per (event, 4-band patch-tuple); label = dom-charge-weighted
    mean u; feature = the tuple's 4 band-patch vectors (missing band -> 0).
  - GEO-ONLY null (plane/wire/t/pres) = the physics floor (one plane can't see u).
  - Metric = per-(event,plane) PEARSON r (Fisher-z mean): bounded, offset/scale invariant,
    isolates within-plane cross-plane-triangulation signal. Plus paired per-event ΔMSE(feat-geo).
  - 8-fold EVENT-grouped CV over all 80 events; standardize + fit alongwire on train folds.
Arms: trained | random(same-arch, correct mup) | raw(input coeffs) | geo(position only).
Usage: python probe_3d_ridge.py --ckpt ckpt_nll_base_fix_snap300000.pt --tag base_fix
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from pb_probe import load_event_data, dev, LAB
from pb_aw import fit_alongwire, u_target
from model import FMModel


def build(ckpt, randinit):
    ck = torch.load(ckpt, map_location=dev)
    nslot = ck.get("n_slot", ck.get("pw", 16) * ck.get("pt", 8))
    heads = ck["d"] // ck["head_dim"] if ck.get("head_dim", 0) else ck.get("heads", 8)
    m = FMModel(nslot, 4, 6, d=ck.get("d", 512), blocks=ck.get("blocks", 12), dec_blocks=ck.get("dec_blocks", 4),
                heads=heads, cond=ck.get("cond", "film"), dec_mode=ck.get("dec_mode", "cross"),
                nll=ck.get("nll", False), mup=ck.get("mup", True), d_base=ck.get("d_base", 128),
                wire_rope=ck.get("wire_rope", True)).to(dev)     # mup read from ckpt -> clean random floor
    if not randinit:
        m.load_state_dict(ck["model"])
    m.eval(); return m


@torch.no_grad()
def feats_of(model, B, layer):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.encode_layers(B, {layer})[layer].float()


def event_patches(evd, feats_tr, feats_rn, layer):
    """-> dict of per-(dom)patch: Xtr, Xrn, Xraw (4-band gathered, missing=0), geo, y, plane."""
    n = evd["n"]
    uk_cols = np.full((n, 4), -1, np.int64)
    for b in range(4):
        idxs, hit = evd["cols"][b]
        uk_cols[:, b] = np.where(hit.numpy().astype(bool), idxs.numpy().astype(np.int64), -1)
    uk, grp = np.unique(uk_cols, axis=0, return_inverse=True)     # unique feature-identity = patch
    ng = len(uk)
    dom = evd["lb"]["dom"].numpy().astype(bool)
    u = evd["u"].numpy(); q = evd["lb"]["qtot"].numpy()
    wq = q * dom
    wsum = np.zeros(ng); usum = np.zeros(ng); cnt = np.zeros(ng)
    np.add.at(wsum, grp, wq); np.add.at(usum, grp, wq * u); np.add.at(cnt, grp, dom.astype(float))
    has = cnt > 0                                                 # keep only patches with >=1 dom hit
    y = np.zeros(ng); y[has] = usum[has] / np.maximum(wsum[has], 1e-12)
    gidp = evd["lb"]["gidp"].numpy().astype(int)
    wire = evd["wire"].astype(float); t = evd["t"].astype(float)
    sw = np.zeros(ng); st = np.zeros(ng); gc = np.zeros(ng)
    np.add.at(sw, grp, wire); np.add.at(st, grp, t); np.add.at(gc, grp, 1.0)
    wmean = sw / np.maximum(gc, 1); tmean = st / np.maximum(gc, 1)
    rep = np.full(ng, -1, int)
    for i in range(n):
        if rep[grp[i]] < 0: rep[grp[i]] = i
    plane = gidp[rep]

    def gather(feats):
        fd = feats.shape[1]
        X = torch.zeros((ng, 4 * fd), device=feats.device)
        for b in range(4):
            cid = uk[:, b]; m = cid >= 0
            if m.any():
                X[torch.tensor(np.where(m)[0], device=feats.device), b * fd:(b + 1) * fd] = \
                    feats[torch.tensor(cid[m], device=feats.device)]
        return X.cpu().numpy().astype(np.float32)

    Xtr = gather(feats_tr); Xrn = gather(feats_rn); Xraw = gather(evd["B"]["inp"].float())
    pres = (uk >= 0).astype(np.float32)
    ponehot = np.eye(6, dtype=np.float32)[plane]
    geo = np.concatenate([ponehot, (wmean / 2000.0)[:, None], (tmean / 4321.0)[:, None], pres], 1)
    s = has
    return dict(Xtr=Xtr[s], Xrn=Xrn[s], Xraw=Xraw[s], geo=geo[s], y=y[s], plane=plane[s])


def ridge_oof(X, y, ev, folds, lams):
    """8-fold event-grouped ridge via CPU-streamed normal equations (memory-frugal: only D×D
    Gram matrices, never the full standardized design). Standardize + center from train-fold
    moments analytically. Return OOF preds at the best global lambda + that lambda."""
    N, D = X.shape
    def acc(): return dict(S2=np.zeros((D, D)), S1=np.zeros(D), Sxy=np.zeros(D), Sy=0.0, n=0)
    nf = int(folds.max()) + 1
    total = acc(); fa = [acc() for _ in range(nf)]
    for e in np.unique(ev):                                       # per-event Gram, add to its fold + total
        s = ev == e; Xe = X[s].astype(np.float64); ye = y[s].astype(np.float64); f = int(folds[s][0])
        S2 = Xe.T @ Xe; S1 = Xe.sum(0); Sxy = Xe.T @ ye; Sy = float(ye.sum()); n = len(ye)
        for A in (total, fa[f]):
            A["S2"] += S2; A["S1"] += S1; A["Sxy"] += Sxy; A["Sy"] += Sy; A["n"] += n
    best = None
    for lam in lams:
        oof = np.zeros_like(y, dtype=np.float64)
        for f in range(nf):
            tr = {k: total[k] - fa[f][k] for k in total}
            n = tr["n"]; mu = tr["S1"] / n; ym = tr["Sy"] / n
            sd = np.sqrt(np.maximum(tr["S2"].diagonal() / n - mu ** 2, 1e-12)); Di = 1.0 / sd
            A = (Di[:, None] * (tr["S2"] - np.outer(tr["S1"], tr["S1"]) / n) * Di[None, :]) + lam * np.eye(D)
            b = Di * (tr["Sxy"] - ym * tr["S1"])                  # standardized+centered normal eqs (exact)
            w = np.linalg.solve(A, b)
            s = folds == f; Xv = X[s].astype(np.float64)
            oof[s] = ((Xv - mu) * Di) @ w + ym
        mse = float(np.mean((oof - y) ** 2))
        if best is None or mse < best[0]:
            best = (mse, lam, oof.copy())
    return best[2], best[1]


def fisher_r(y, p, ev, pl):
    rs = []
    for e in np.unique(ev):
        for g in range(6):
            s = (ev == e) & (pl == g)
            if s.sum() >= 100 and y[s].std() > 1e-6 and p[s].std() > 1e-6:
                rs.append(np.corrcoef(y[s], p[s])[0, 1])
    rs = np.array(rs); z = np.arctanh(np.clip(rs, -0.999, 0.999))
    return float(np.tanh(z.mean())), rs


def per_event_mse(y, p, ev):
    return {int(e): float(np.mean((y[ev == e] - p[ev == e]) ** 2)) for e in np.unique(ev)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", default="ref")
    ap.add_argument("--layer", type=int, default=12); ap.add_argument("--nfold", type=int, default=8)
    ap.add_argument("--events", default="30000-30079"); ap.add_argument("--out", default="probe_3d_ridge.jsonl")
    a = ap.parse_args()
    lo, hi = map(int, a.events.split("-"))
    evs = [e for e in range(lo, hi + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    print(f"[{a.tag}] loading {len(evs)} events, layer {a.layer}", flush=True)
    raw = [load_event_data(e) for e in evs]
    aw = fit_alongwire(raw)                                       # plane geometry (stable; Agent2 verified)
    for r in raw: r["u"] = u_target(r, aw)
    mt = build(a.ckpt, randinit=False); mr = build(a.ckpt, randinit=True)
    packs = []
    for r in raw:
        ft = feats_of(mt, r["B"], a.layer); fr = feats_of(mr, r["B"], a.layer)
        packs.append(event_patches(r, ft, fr, a.layer))
    del mt, mr; torch.cuda.empty_cache()

    ev = np.concatenate([np.full(len(p["y"]), i) for i, p in enumerate(packs)])
    plane = np.concatenate([p["plane"] for p in packs]); y = np.concatenate([p["y"] for p in packs])
    folds = (ev // (int(np.ceil((ev.max() + 1) / a.nfold)))).astype(int)     # contiguous event blocks
    geo = np.concatenate([p["geo"] for p in packs])
    arms = {"trained": np.concatenate([p["Xtr"] for p in packs]),
            "random":  np.concatenate([p["Xrn"] for p in packs]),
            "raw":     np.concatenate([p["Xraw"] for p in packs])}
    lams = [1e1, 1e2, 1e3, 1e4, 1e5]
    print(f"[{a.tag}] {len(y)} dom-patches, {folds.max()+1} folds, geo_dim={geo.shape[1]}", flush=True)

    # geo-only null
    gp, glam = ridge_oof(geo, y, ev, folds, lams)
    gr, _ = fisher_r(y, gp, ev, plane); gmse = per_event_mse(y, gp, ev)
    res = {"tag": a.tag, "layer": a.layer, "n_patch": int(len(y)),
           "geo": {"fisher_r": round(gr, 4), "lam": glam}}
    print(f"  geo-only:  fisher_r={gr:+.4f}", flush=True)
    for name, Xf in arms.items():
        X = np.concatenate([Xf, geo], 1)                         # feature + geo covariates
        p, lam = ridge_oof(X, y, ev, folds, lams)
        fr, rs = fisher_r(y, p, ev, plane); fmse = per_event_mse(y, p, ev)
        dmse = np.array([gmse[e] - fmse[e] for e in fmse])       # >0 = feature beats geo
        # Wilcoxon signed-rank (paired), no scipy: use sign + normal approx on ranks
        d = dmse[dmse != 0]; Rp = None
        if len(d):
            rank = np.argsort(np.argsort(np.abs(d))) + 1
            Wp = rank[d > 0].sum(); Wm = rank[d < 0].sum(); nR = len(d)
            mu = nR * (nR + 1) / 4; sig = np.sqrt(nR * (nR + 1) * (2 * nR + 1) / 24)
            Rp = round(float((min(Wp, Wm) - mu) / (sig + 1e-9)), 2)
        res[name] = {"fisher_r": round(fr, 4), "lam": lam, "d_over_geo_r": round(fr - gr, 4),
                     "median_dMSE": round(float(np.median(dmse)), 5),
                     "frac_events_beat_geo": round(float((dmse > 0).mean()), 3), "wilcoxon_z": Rp}
        print(f"  {name:8s}: fisher_r={fr:+.4f}  Δr_over_geo={fr-gr:+.4f}  "
              f"median_ΔMSE={np.median(dmse):+.5f}  beat_geo={float((dmse>0).mean()):.2f}", flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"WROTE {a.tag} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
