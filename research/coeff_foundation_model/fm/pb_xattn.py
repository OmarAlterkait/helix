"""Event-level CROSS-ATTENTION readout for the along-wire (cross-plane) coordinate u.

The per-token readout (pb_aw) only sees the pixel's OWN plane's 4 tokens -> it under-
credits if the FM triangulated but stored u NON-LOCALLY (in the V/Y tokens / cross-plane
relations). Here a query anchored at the pixel (its gidp,wire,t geometry) attends over
ALL event tokens (all planes) and predicts u. If FM u-R2 >> the per-token +0.09 AND >>
random-init tokens (same geometry, untrained content), the FM DID triangulate.

Controls: random = attention over random-init FM tokens (geometry is in tokens either
way, so this isolates LEARNED content); raw = attention over input-coeff tokens.
Metric = WITHIN-(vol,plane) R2 on u (pooled u is dominated by 'which plane')."""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from pb_probe import load_event, dev, LAB, FMModel, CACHE
from pb_aw import fit_alongwire, u_target, r2


def qgeo(evd, sel):
    gp = evd["lb"]["gidp"][sel].long()
    g1 = torch.nn.functional.one_hot(gp, 6).float()
    w = torch.tensor(evd["wire"][sel.numpy()] / 2000.0, dtype=torch.float32)[:, None]
    t = torch.tensor(evd["t"][sel.numpy()] / 4321.0, dtype=torch.float32)[:, None]
    ph = evd["phase"][sel]
    return torch.cat([g1, w, t, ph], 1).to(dev)                  # (Nq, 6+1+1+5)


class XProbe(nn.Module):
    def __init__(self, d, in_geo, nh=8):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(in_geo, d), nn.GELU(), nn.Linear(d, d))
        self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.pos = nn.Linear(in_geo, d)                          # additive key-position (geometry of each cell)
        self.nh = nh; self.hd = d // nh
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, qg, H, kg):
        Q = self.q(qg); K = self.k(H) + self.pos(kg); V = self.v(H)
        Nq, Nc = Q.shape[0], H.shape[0]
        Q = Q.view(Nq, self.nh, self.hd).transpose(0, 1)
        K = K.view(Nc, self.nh, self.hd).transpose(0, 1)
        V = V.view(Nc, self.nh, self.hd).transpose(0, 1)
        A = torch.softmax(Q @ K.transpose(-1, -2) / self.hd ** 0.5, -1)
        ctx = (A @ V).transpose(0, 1).reshape(Nq, -1)
        return self.out(ctx).squeeze(-1)


def cell_geo(evd):
    """Per-CELL geometry (gidp,wire,t,phase) for key-position — built from the cache keys."""
    d = np.load(os.path.join(CACHE, f"ev_{evd['ev']:05d}.npz")) if "ev" in evd else None
    # cells are (gid, band, wb, tb): recover gidp=gid, wire=wb*16, tau=tb*8
    from pb_probe import cache_keys_sorted
    keys = cache_keys_sorted(os.path.join(CACHE, f"ev_{evd['evn']:05d}.npz"))
    gid = (keys >> 40) & 0x3F; band = (keys >> 36) & 0xF
    wb = (keys >> 18) & 0x3FFFF; tb = keys & 0x3FFFF
    g1 = np.eye(6, dtype=np.float32)[gid]
    w = (wb * 16 / 2000.0).astype(np.float32)[:, None]
    t = (tb * 8 * (1 << np.array([4, 4, 3, 2])[band]) / 4321.0).astype(np.float32)[:, None]
    ph = np.zeros((len(keys), 5), dtype=np.float32)              # cells have no sub-slot phase; pad
    return torch.tensor(np.concatenate([g1, w, t, ph], 1), device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["nll", "random", "raw"], required=True)
    ap.add_argument("--ckpt", default="ckpt_sc_w_d768.pt"); ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--tag", default="")
    ap.add_argument("--train", default="30000-30059"); ap.add_argument("--eval", default="30060-30079")
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--px", type=int, default=2048)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    model = FMModel(128, 4, 6, d=a.d, blocks=12, dec_blocks=4, heads=a.heads, cond="film",
                    dec_mode="cross", nll=True).to(dev)
    if a.arm == "nll":
        model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
    model.eval()
    lo, hi = map(int, a.train.split("-")); el, eh = map(int, a.eval.split("-"))
    tr = [e for e in range(lo, hi + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    ev = [e for e in range(el, eh + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    arm_feat = "nll" if a.arm in ("nll", "random") else "raw"
    evds_tr = [load_event(e, arm_feat, model) for e in tr]
    evds_ev = [load_event(e, arm_feat, model) for e in ev]
    for e, n in zip(evds_tr, tr): e["evn"] = n
    for e, n in zip(evds_ev, ev): e["evn"] = n
    aw = fit_alongwire(evds_tr)
    for e in evds_tr + evds_ev:
        e["u"] = u_target(e, aw); e["kg"] = cell_geo(e); e["H"] = e["feats"].cpu(); e["feats"] = None
    pmt = np.zeros(6); cnt = np.zeros(6)                         # per-plane mean of u (train) -> demean the TARGET
    for e in evds_tr:
        g = e["lb"]["gidp"].numpy().astype(int); u = e["u"].numpy(); dm = e["lb"]["dom"].numpy().astype(bool)
        for gi in range(6):
            s = dm & (g == gi); pmt[gi] += u[s].sum(); cnt[gi] += s.sum()
    pmt = pmt / np.maximum(cnt, 1); pmt_t = torch.tensor(pmt, dtype=torch.float32)
    print(f"train on WITHIN-PLANE-DEMEANED u (plane means {pmt.round(0).tolist()})", flush=True)
    fdim = evds_tr[0]["fdim"]
    print(f"arm={a.arm} tag={a.tag} train={len(tr)} val={len(ev)} d_attn={fdim}", flush=True)
    probe = XProbe(fdim, 6 + 1 + 1 + 5).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), 5e-4, weight_decay=1e-4)   # more WD vs overfit
    Gv = np.concatenate([e["lb"]["gidp"].numpy().astype(int) for e in evds_ev])
    Uv = np.concatenate([e["u"].numpy() for e in evds_ev]); domv = np.concatenate([e["lb"]["dom"].numpy().astype(bool) for e in evds_ev])
    pmv = np.array([Uv[domv & (Gv == g)].mean() if (domv & (Gv == g)).any() else 0 for g in range(6)])

    @torch.no_grad()
    def val_r2():                                                # within-plane R2 on VAL (per-event H moved to GPU)
        P = []
        for evd in evds_ev:
            H = evd["H"].to(dev); o = []
            for s0 in range(0, evd["n"], 3072):
                sel = torch.arange(s0, min(s0 + 3072, evd["n"]))
                o.append(probe(qgeo(evd, sel), H, evd["kg"]).cpu())
            P.append(torch.cat(o).numpy()); del H
        P = np.concatenate(P) + pmv[Gv]
        r = 1 - ((Uv[domv] - P[domv]) ** 2).sum() / max(((Uv[domv] - pmv[Gv[domv]]) ** 2).sum(), 1e-9)
        return float(r), P

    best = -9.0; bestP = None
    for step in range(1, a.steps + 1):
        evd = evds_tr[np.random.randint(len(evds_tr))]
        sel = torch.tensor(np.random.randint(0, evd["n"], min(a.px, evd["n"])))
        pred = probe(qgeo(evd, sel), evd["H"].to(dev), evd["kg"])
        dom = evd["lb"]["dom"][sel].to(dev)
        tgt = (evd["u"][sel] - pmt_t[evd["lb"]["gidp"][sel].long()]).to(dev)   # demeaned
        loss = ((pred - tgt) ** 2 * dom).sum() / max(dom.sum(), 1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            vr, P = val_r2()
            if vr > best: best, bestP = vr, P
            print(f"  step {step} loss {float(loss):.4f}  val_R2 {vr:.4f} (best {best:.4f})", flush=True)
    P = bestP; G = Gv; U = Uv; dom = domv; pm = pmv                # early-stopped (best-val) predictions
    F = np.concatenate([e["lb"]["F"].numpy() for e in evds_ev]); Q = np.concatenate([e["lb"]["qtot"].numpy() for e in evds_ev])
    r2wp = lambda s: round(float(1 - ((U[s] - P[s]) ** 2).sum() / max(((U[s] - pm[G[s]]) ** 2).sum(), 1e-9)), 3)
    out = {"within_plane_BEST": round(float(best), 3)}
    for g, nm in [(0, "v0U"), (1, "v0V"), (2, "v0Y"), (3, "v1U"), (4, "v1V"), (5, "v1Y")]:
        s = dom & (G == g); out[nm] = r2wp(s) if s.sum() > 500 else None
    for nm, lo2, hi2 in [("F.5-.6", .5, .6), ("F.95-1", .95, 1.01)]:
        s = dom & (F >= lo2) & (F < hi2); out[nm] = r2wp(s) if s.sum() > 500 else None
    for nm, lo2, hi2 in [("q250-500", 250, 500), ("q3k+", 3000, 1e12)]:
        s = dom & (Q >= lo2) & (Q < hi2); out[nm] = r2wp(s) if s.sum() > 500 else None
    print("XATTN " + json.dumps(dict(arm=a.arm, tag=a.tag, **out)), flush=True)


if __name__ == "__main__":
    main()
