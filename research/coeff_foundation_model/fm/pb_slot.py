"""OPTIMAL per-pixel readout: SLOT-INDEXED head (val_head-style) at the best layer.

My 4-band-concat readout shares the cell feature across all 128 slots (only additive
phase separates pixels). The model's OWN per-pixel mechanism is val_head: Linear(d->128),
one weight row per slot. Here the along-wire probe uses the same: per band, project the
pixel's cell feature to 128 slots and GATHER the pixel's slot = (wire%16)*8+(tau%8).
This is the maximally-expressive per-pixel readout. Run at --layer 6 (the 3D peak).
Baselines: 4-band-concat @L6 = 0.20, @L12 = 0.09; geo ~ -0.05."""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
import pb_probe as PP
from pb_probe import load_event, readout, dev, LAB, FMModel
from pb_aw import fit_alongwire, u_target, r2
from pb_labels import DEC, DELTA_T, LENS_T, TOFF, PLANES


def slots_for(evd):
    wire = evd["wire"]; t = evd["t"]; gidp = evd["lb"]["gidp"].numpy().astype(int)
    out = []
    for b in range(4):
        toff = np.array([TOFF[PLANES[g % 3]] for g in gidp])
        tau = np.clip(np.round((t + toff) / DEC[b] - DELTA_T[b]).astype(int), 0, LENS_T[b] - 1)
        out.append(torch.tensor((wire % 16) * 8 + (tau % 8), dtype=torch.long))   # slot in [0,128)
    return out


class UProbeSlot(nn.Module):
    def __init__(self, d, in_geo, h=256):
        super().__init__()
        self.sh = nn.ModuleList([nn.Linear(d, 128) for _ in range(4)])            # per-band val_head-style
        self.gb = nn.Sequential(nn.Linear(in_geo, 64), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(4 + 64, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, feats, cols, slots, xg, sel, mu, sd):
        us = []
        for b in range(4):
            idxs, hit = cols[b]
            fb = feats[idxs[sel].to(dev)]
            if mu is not None:
                fb = (fb - mu) / sd
            proj = self.sh[b](fb)                                                 # (n,128)
            ub = proj.gather(1, slots[b][sel].to(dev)[:, None]).squeeze(1) * hit[sel].to(dev).float()
            us.append(ub)
        z = torch.cat([torch.stack(us, 1), self.gb(xg)], 1)
        return self.fuse(z).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=6); ap.add_argument("--tag", default="slot")
    ap.add_argument("--train", default="30000-30059"); ap.add_argument("--eval", default="30060-30079")
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--px", type=int, default=8192)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    if a.layer > 0: PP.PROBE_LAYER = a.layer
    model = FMModel(128, 4, 6, d=768, blocks=12, dec_blocks=4, heads=4, cond="film", dec_mode="cross", nll=True).to(dev)
    model.load_state_dict(torch.load("ckpt_sc_w_d768.pt", map_location=dev)["model"]); model.eval()
    lo, hi = map(int, a.train.split("-")); el, eh = map(int, a.eval.split("-"))
    tr = [e for e in range(lo, hi + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    ev = [e for e in range(el, eh + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    evds_tr = [load_event(e, "nll", model) for e in tr]; evds_ev = [load_event(e, "nll", model) for e in ev]
    aw = fit_alongwire(evds_tr)
    for e in evds_tr + evds_ev:
        e["u"] = u_target(e, aw); e["slots"] = slots_for(e)
    print(f"layer={a.layer} tag={a.tag} train={len(tr)} eval={len(ev)}", flush=True)
    fdim = evds_tr[0]["fdim"]
    smp = torch.cat([e["feats"][:4096] for e in evds_tr[:10]]); mu = smp.mean(0).to(dev); sd = (smp.std(0) + 1e-6).to(dev)
    in_geo = 4 + 5 + 6 + 2
    probe = UProbeSlot(fdim, in_geo).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), 1e-3, weight_decay=1e-5)
    # per-plane demean (train) — optimize the along-wire residual, not the coarse 'which plane'
    pmt = np.zeros(6); cnt = np.zeros(6)
    for e in evds_tr:
        g = e["lb"]["gidp"].numpy().astype(int); u = e["u"].numpy(); dm = e["lb"]["dom"].numpy().astype(bool)
        for gi in range(6):
            s = dm & (g == gi); pmt[gi] += u[s].sum(); cnt[gi] += s.sum()
    pmt = pmt / np.maximum(cnt, 1); pmt_t = torch.tensor(pmt, dtype=torch.float32)
    for step in range(1, a.steps + 1):
        evd = evds_tr[np.random.randint(len(evds_tr))]
        sel = torch.tensor(np.random.randint(0, evd["n"], min(a.px, evd["n"])))
        _, Xg = readout(evd, torch.zeros(4, fdim, device=dev), "nll", sel, mu, sd)
        pred = probe(evd["feats"], evd["cols"], evd["slots"], Xg, sel, mu, sd)
        dom = evd["lb"]["dom"][sel].to(dev)
        tgt = (evd["u"][sel] - pmt_t[evd["lb"]["gidp"][sel].long()]).to(dev)
        loss = ((pred - tgt) ** 2 * dom).sum() / max(dom.sum(), 1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0: print(f"  step {step} loss {float(loss):.4f}", flush=True)
    # eval within-plane R2
    U = []; P = []; G = []; F = []; Q = []; DM = []
    with torch.no_grad():
        for evd in evds_ev:
            o = []
            for s0 in range(0, evd["n"], 32768):
                sel = torch.arange(s0, min(s0 + 32768, evd["n"]))
                _, Xg = readout(evd, torch.zeros(4, fdim, device=dev), "nll", sel, mu, sd)
                o.append(probe(evd["feats"], evd["cols"], evd["slots"], Xg, sel, mu, sd).cpu())
            P.append((torch.cat(o).numpy())); U.append(evd["u"].numpy()); G.append(evd["lb"]["gidp"].numpy().astype(int))
            F.append(evd["lb"]["F"].numpy()); Q.append(evd["lb"]["qtot"].numpy()); DM.append(evd["lb"]["dom"].numpy().astype(bool))
    U = np.concatenate(U); G = np.concatenate(G); P = np.concatenate(P) + pmt[G]
    F = np.concatenate(F); Q = np.concatenate(Q); dom = np.concatenate(DM)
    r2wp = lambda s: round(float(1 - ((U[s] - P[s]) ** 2).sum() / max(((U[s] - pmt[G[s]]) ** 2).sum(), 1e-9)), 3)
    out = {"within_plane": r2wp(dom)}
    for g, nm in [(0, "v0U"), (1, "v0V"), (2, "v0Y"), (3, "v1U"), (4, "v1V"), (5, "v1Y")]:
        s = dom & (G == g); out[nm] = r2wp(s) if s.sum() > 500 else None
    for nm, lo2, hi2 in [("F.5-.6", .5, .6), ("F.95-1", .95, 1.01)]:
        s = dom & (F >= lo2) & (F < hi2); out[nm] = r2wp(s) if s.sum() > 500 else None
    for nm, lo2, hi2 in [("q250-500", 250, 500), ("q3k+", 3000, 1e12)]:
        s = dom & (Q >= lo2) & (Q < hi2); out[nm] = r2wp(s) if s.sum() > 500 else None
    print("SLOT " + json.dumps(dict(layer=a.layer, tag=a.tag, **out)), flush=True)


if __name__ == "__main__":
    main()
