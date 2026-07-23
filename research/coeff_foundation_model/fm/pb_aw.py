"""CORRECTED 3D probe: the ALONG-WIRE coordinate (the cross-plane DOF).

The earlier B1x/y/z metric was broken: x=0.99 was just 'which volume' (bimodal),
and fixed-axis y/z pooled across planes mixed wire-PINNED (trivial) with along-wire
(hard) sub-populations plane-by-plane -> diluted, flat, uninterpretable.

Here the target is u = along-wire coordinate = position PERP to the wire-pitch
direction, in the (y,z) wire plane. A single plane cannot determine u (the wire only
fixes the pitch coordinate); recovering it REQUIRES cross-plane triangulation. Per
(vol,plane) we fit wire ~ a*y+b*z to get pitch dir (a,b); along-wire dir = (-b,a).
The head is RETRAINED on u (the old head optimized coarse x,y,z MSE). Arms:
  nll/random = FM/init features ; raw = input coeffs ; geo = gidp+wire+t (has t so it
  captures any drift-leakage into u -> FM's gain OVER geo = the clean cross-plane signal).
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from pb_probe import load_event, readout, dev, LAB, FMModel


def fit_alongwire(evds):
    """Per gidp: unit along-wire vector (uy, uz) in the (y,z) plane."""
    Y = {g: [[], []] for g in range(6)}
    for e in evds:
        gp = e["lb"]["gidp"].numpy().astype(int); b1 = e["lb"]["b1"].numpy(); w = e["wire"].astype(float)
        for g in range(6):
            s = gp == g
            if s.any():
                Y[g][0].append(np.column_stack([b1[s, 1], b1[s, 2], np.ones(s.sum())])); Y[g][1].append(w[s])
    aw = {}
    for g in range(6):
        if not Y[g][0]:
            continue
        A = np.concatenate(Y[g][0]); w = np.concatenate(Y[g][1])
        c, *_ = np.linalg.lstsq(A, w, rcond=None)                # wire = a*y + b*z + c0
        a, b = c[0], c[1]; n = np.hypot(a, b)
        aw[g] = np.array([-b / n, a / n])                        # along-wire unit (perp to pitch)
    return aw


def u_target(evd, aw):
    gp = evd["lb"]["gidp"].numpy().astype(int); b1 = evd["lb"]["b1"].numpy()
    u = np.zeros(len(gp))
    for g, v in aw.items():
        s = gp == g
        u[s] = b1[s, 1] * v[0] + b1[s, 2] * v[1]                 # mm
    return torch.tensor(u / 1000.0, dtype=torch.float32)          # -> m


class UProbe(nn.Module):
    def __init__(self, in_feat, in_geo, h=512):
        super().__init__()
        self.fb = nn.Sequential(nn.Linear(in_feat, h), nn.GELU()) if in_feat > 0 else None
        self.gb = nn.Sequential(nn.Linear(in_geo, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear((h if in_feat > 0 else 0) + 64, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, xf, xg):
        z = self.gb(xg) if self.fb is None else torch.cat([self.fb(xf), self.gb(xg)], 1)
        return self.fuse(z).squeeze(-1)


def r2(y, p, w=None):
    if w is None:
        return 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    yb = (w * y).sum() / w.sum()
    return 1 - (w * (y - p) ** 2).sum() / max((w * (y - yb) ** 2).sum(), 1e-9)


@torch.no_grad()
def evaluate(probe, evds, miss, arm, mu, sd, chunk=65536):
    U = []; P = []; G = []; F = []; Q = []
    for evd in evds:
        outs = []
        for s0 in range(0, evd["n"], chunk):
            sel = torch.arange(s0, min(s0 + chunk, evd["n"]))
            outs.append(probe(*readout(evd, miss, arm, sel, mu, sd)).cpu())
        P.append(torch.cat(outs).numpy()); U.append(evd["u"].numpy())
        G.append(evd["lb"]["gidp"].numpy()); F.append(evd["lb"]["F"].numpy()); Q.append(evd["lb"]["qtot"].numpy())
        dom = evd["lb"]["dom"].numpy().astype(bool)
    U = np.concatenate(U); P = np.concatenate(P); G = np.concatenate(G); F = np.concatenate(F); Q = np.concatenate(Q)
    dom = np.concatenate([e["lb"]["dom"].numpy().astype(bool) for e in evds])
    pm = np.zeros(6)                                             # per-(vol,plane) mean of true u (dom)
    for g in range(6):
        sg = dom & (G == g)
        if sg.any(): pm[g] = U[sg].mean()
    Gi = G.astype(int)
    def r2wp(sel):                                               # denom = WITHIN-plane variance
        num = ((U[sel] - P[sel]) ** 2).sum()
        den = ((U[sel] - pm[Gi[sel]]) ** 2).sum()
        return round(float(1 - num / max(den, 1e-9)), 3)
    out = {"overall_pooled_DILUTED": round(float(r2(U[dom], P[dom])), 3),
           "within_plane": r2wp(dom)}
    for g, nm in [(0, "v0U"), (1, "v0V"), (2, "v0Y"), (3, "v1U"), (4, "v1V"), (5, "v1Y")]:
        sg = dom & (G == g); out[nm] = r2wp(sg) if sg.sum() > 500 else None
    for nm, lo, hi in [("F.5-.6", .5, .6), ("F.8-.95", .8, .95), ("F.95-1", .95, 1.01)]:
        sg = dom & (F >= lo) & (F < hi); out[nm] = r2wp(sg) if sg.sum() > 500 else None
    for nm, lo, hi in [("q250-500", 250, 500), ("q1k-3k", 1000, 3000), ("q3k+", 3000, 1e12)]:
        sg = dom & (Q >= lo) & (Q < hi); out[nm] = r2wp(sg) if sg.sum() > 500 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["nll", "random", "raw", "geo"], required=True)
    ap.add_argument("--ckpt", default="ckpt_sc_w_d768.pt"); ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--dec", default="cross")
    ap.add_argument("--mnll", type=int, default=1); ap.add_argument("--tag", default="")
    ap.add_argument("--cwloss", action="store_true"); ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--train", default="30000-30059"); ap.add_argument("--eval", default="30060-30079")
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--px", type=int, default=8192)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    import pb_probe as PP
    if a.layer > 0: PP.PROBE_LAYER = a.layer   # probe an intermediate encoder layer
    model = None
    if a.arm in ("nll", "random"):
        model = FMModel(128, 4, 6, d=a.d, blocks=a.blocks, dec_blocks=4, heads=a.heads, cond="film",
                        dec_mode=a.dec, nll=bool(a.mnll)).to(dev)
        if a.arm == "nll":
            model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
        model.eval()
    lo, hi = map(int, a.train.split("-")); el, eh = map(int, a.eval.split("-"))
    tr = [e for e in range(lo, hi + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    ev = [e for e in range(el, eh + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    evds_tr = [load_event(e, a.arm, model) for e in tr]
    evds_ev = [load_event(e, a.arm, model) for e in ev]
    aw = fit_alongwire(evds_tr)
    print(f"arm={a.arm} tag={a.tag} train={len(tr)} eval={len(ev)} alongwire={ {g:v.round(2).tolist() for g,v in aw.items()} }", flush=True)
    for e in evds_tr + evds_ev:
        e["u"] = u_target(e, aw)
    fdim = evds_tr[0]["fdim"]
    mu = sd = None
    if fdim > 0:
        smp = torch.cat([e["feats"][:4096] for e in evds_tr[:10]]); mu = smp.mean(0).to(dev); sd = (smp.std(0) + 1e-6).to(dev)
    in_feat = fdim * 4 if a.arm != "geo" else 0
    miss = nn.Parameter(torch.zeros(4, fdim, device=dev)) if a.arm in ("nll", "random") else torch.zeros(4, 1)
    probe = UProbe(in_feat, 4 + 5 + 6 + 2).to(dev)
    params = list(probe.parameters()) + ([miss] if isinstance(miss, nn.Parameter) else [])
    opt = torch.optim.AdamW(params, 1e-3, weight_decay=1e-5)
    for step in range(1, a.steps + 1):
        evd = evds_tr[np.random.randint(len(evds_tr))]
        sel = torch.tensor(np.random.randint(0, evd["n"], min(a.px, evd["n"])))
        Xf, Xg = readout(evd, miss, a.arm, sel, mu, sd)
        pred = probe(Xf, Xg)
        dom = evd["lb"]["dom"][sel].to(dev)
        w = dom * (evd["lb"]["qtot"][sel].to(dev) if a.cwloss else 1.0)
        loss = ((pred - evd["u"][sel].to(dev)) ** 2 * w).sum() / max(w.sum(), 1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"  step {step} loss {float(loss):.4f}", flush=True)
    print("AWFINAL " + json.dumps(dict(arm=a.arm, tag=a.tag, **evaluate(probe, evds_ev, miss, a.arm, mu, sd))), flush=True)


if __name__ == "__main__":
    main()
