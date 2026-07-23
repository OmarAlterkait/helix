"""Stage 2: group-level probe runs on frozen encoder features.

Per-pixel readout = concat of the 4 band-token features covering the pixel
(missing band -> learned embedding) + presence bits + slot phases -> shared MLP
trunk -> heads:  D (logD regression + D>5mm binary), F, B1 (x,y,z mm), theta/phi.
Zero-shot X = cross-plane same-gid retrieval on charge-weighted group embeddings
(cosine, whitened), candidates within +-5 ticks, same volume.

Arms (--arm): nll   = sc_w_d768 (heads=4! train.py default — the heads=12 loads were a bug)
              random= random-init FMModel d768 (head-capacity control)
              raw   = the 4 tokens' raw input coeffs instead of features (no-FM baseline)
              geo   = position-only (gid/wire/t/phase) — geometry floor for B1
"""
import sys, os, glob, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
import data as D
D.init_pipeline_cpu()
from model import FMModel
from pb_labels import LENS_T, DEC, DELTA_T, TOFF, PLANES, pix_token_keys

dev = "cuda"
CACHE = "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/artifacts/fm_cache_tpc"
PROBE_LAYER = None   # None=final encode(); int k=features after block k (layer sweep)
LAB = "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/artifacts/probe_labels"


def cache_keys_sorted(path):
    d = np.load(path)
    band = d["band"].astype(np.int64); gid = d["gid"].astype(np.int64)
    wire = d["wire"].astype(np.int64); idx = d["idx"].astype(np.int64)
    tau = idx % LENS_T[band]
    key = (gid << 40) | (band << 36) | ((wire // 16) << 18) | (tau // 8)
    return np.unique(key)                                        # == vit_tpc cell order (np.unique)


def load_event_data(ev, device=dev):
    """Model-INDEPENDENT part of an event: token batch B + labels + geometry + per-pixel cols.
    Split out of load_event so a sweep can load events ONCE and forward many models/layers
    against the cached tokens (kills the redundant re-load + parallel FS contention)."""
    lab = np.load(os.path.join(LAB, f"pl_{ev:05d}.npz"))
    cpath = os.path.join(CACHE, f"ev_{ev:05d}.npz")
    keys = cache_keys_sorted(cpath)
    B = D.get_cached(cpath, device=device)
    assert B["n_cells"] == len(keys), f"cell-order mismatch ev{ev}: {B['n_cells']} vs {len(keys)}"
    gidp = lab["gidp"].astype(np.int64); wire = lab["wire"].astype(np.int64); t = lab["t"].astype(np.int64)
    n = len(gidp)
    phase = [torch.tensor((wire % 16) / 16.0, dtype=torch.float32)]
    cols = []
    pres = torch.tensor(lab["pres"].astype(np.float32))
    for b in range(4):
        toff = np.array([TOFF[PLANES[g % 3]] for g in gidp])
        tau = np.clip(np.round((t + toff) / DEC[b] - DELTA_T[b]).astype(np.int64), 0, LENS_T[b] - 1)
        kb = (gidp << 40) | (np.int64(b) << 36) | ((wire // 16) << 18) | (tau // 8)
        pos = np.searchsorted(keys, kb)
        hit = (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == kb)
        idxs = torch.tensor(np.where(hit, pos, 0), dtype=torch.long)
        cols.append((idxs, torch.tensor(hit)))
        phase.append(torch.tensor((tau % 8) / 8.0, dtype=torch.float32))
    lb = {k: torch.tensor(np.asarray(lab[k], dtype=np.float32)) for k in ("qtot", "F", "D", "theta", "phi")}
    lb["b1"] = torch.tensor(lab["b1"].astype(np.float32)); lb["gidp"] = torch.tensor(gidp)
    lb["dom"] = torch.tensor(lab["dom"].astype(bool))
    grp = {k: np.asarray(lab[f"grp_{k}"]) for k in ("vol", "plane", "gid", "tmean")}
    out = dict(B=B, feats=None, fdim=0, cols=cols, pres=pres, phase=torch.stack(phase, 1),
               lb=lb, grp=grp, n=n, wire=wire, t=t)
    if "lgid" in lab:
        out["lgid"] = lab["lgid"].astype(np.int64)
    return out


def set_feats(evd, arm, model, layers=None):
    """Attach model features to a raw evd (from load_event_data). layers=None -> PROBE_LAYER
    (or final); a set -> returns per-layer feats via encode_layers (one forward, many layers)."""
    B = evd["B"]
    if arm in ("nll", "random"):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if layers is not None:
                return model.encode_layers(B, set(layers))       # {layer: feats} — caller sets per layer
            feats = (model.encode(B) if PROBE_LAYER is None else model.encode_layers(B, {PROBE_LAYER})[PROBE_LAYER]).float()
        evd["feats"] = feats; evd["fdim"] = feats.shape[1]
    elif arm == "raw":
        evd["feats"] = B["inp"].float(); evd["fdim"] = evd["feats"].shape[1]
    else:
        evd["feats"] = None; evd["fdim"] = 0
    return evd


def load_event(ev, arm, model):
    """-> per-pixel readout X (n, dim) on GPU + label dict. (Back-compat: load + forward.)"""
    return set_feats(load_event_data(ev), arm, model)


def readout(evd, miss, arm, sel=None, mu=None, sd=None):
    """Per-pixel inputs -> (feat_part or None, geo_part). Features standardized by (mu, sd)."""
    n = evd["n"]; sel = torch.arange(n) if sel is None else sel
    Xf = None
    if arm in ("nll", "random", "raw"):
        parts = []
        f = evd["feats"]
        for b in range(4):
            idxs, hit = evd["cols"][b]
            fb = f[idxs[sel].to(dev)].clone()
            if mu is not None:
                fb = (fb - mu) / sd
            fb[~hit[sel].to(dev)] = miss[b] if arm != "raw" else 0.0
            parts.append(fb)
        Xf = torch.cat(parts, 1)
    gp = torch.nn.functional.one_hot(evd["lb"]["gidp"][sel].long(), 6).float().to(dev)
    Xg = torch.cat([evd["pres"][sel].to(dev), evd["phase"][sel].to(dev), gp,
                    torch.tensor(evd["wire"][sel.numpy()] / 2000.0, dtype=torch.float32, device=dev)[:, None],
                    torch.tensor(evd["t"][sel.numpy()] / 4321.0, dtype=torch.float32, device=dev)[:, None]], 1)
    return Xf, Xg


class Probe(nn.Module):
    """Two-branch: geometry scalars get their OWN branch (bug#6 fix — a shared trunk
    drowns 17 geo dims in 3072 feature dims; geo arm beat feature arms on B1y/z)."""

    def __init__(self, in_feat, in_geo, h=512):
        super().__init__()
        self.fb = nn.Sequential(nn.Linear(in_feat, h), nn.GELU()) if in_feat > 0 else None
        self.gb = nn.Sequential(nn.Linear(in_geo, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear((h if in_feat > 0 else 0) + 64, h), nn.GELU())
        self.hD = nn.Linear(h, 2); self.hF = nn.Linear(h, 1)
        self.hB = nn.Linear(h, 3); self.hA = nn.Linear(h, 3)     # angle = unit VECTOR (bug#7 fix)

    def forward(self, xf, xg):
        z = self.gb(xg) if self.fb is None else torch.cat([self.fb(xf), self.gb(xg)], 1)
        z = self.fuse(z)
        return self.hD(z), self.hF(z).squeeze(-1), self.hB(z), self.hA(z)


def losses(out, lb, sel, cw=False):
    oD, oF, oB, oA = out
    logD = torch.log(lb["D"][sel].to(dev) + 0.1)
    d5 = (lb["D"][sel].to(dev) > 5.0).float()
    dom = lb["dom"][sel].to(dev)
    lD = ((oD[:, 0] - logD) ** 2).mean() + nn.functional.binary_cross_entropy_with_logits(oD[:, 1], d5)
    lF = ((oF - lb["F"][sel].to(dev)) ** 2).mean()
    b1 = lb["b1"][sel].to(dev) / 1000.0                          # mm -> m scale
    wB = dom * (lb["qtot"][sel].to(dev) if cw else 1.0)          # charge-weight B1 -> place bright pixels well
    lB = (((oB - b1) ** 2).sum(1) * wB).sum() / max(wB.sum(), 1)
    th = lb["theta"][sel].to(dev); ph = lb["phi"][sel].to(dev)
    u = torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], 1)
    ua = oA / (oA.norm(dim=1, keepdim=True) + 1e-6)
    lA = ((1 - (ua * u).sum(1).abs()) * dom).sum() / max(dom.sum(), 1)   # sign-invariant (phi circular + antipodal)
    return lD + lF + lB + 0.5 * lA


@torch.no_grad()
def evaluate(model_p, evds, miss, arm, mu=None, sd=None, chunk=65536):
    agg = {k: [] for k in ("logD", "pD", "d5", "pd5", "F", "pF", "b1", "pb1", "ang", "pang", "dom")}
    for evd in evds:
        outs = []
        for s0 in range(0, evd["n"], chunk):                     # chunked: full-event readout would OOM
            sel = torch.arange(s0, min(s0 + chunk, evd["n"]))
            outs.append(model_p(*readout(evd, miss, arm, sel, mu, sd)))
        oD, oF, oB, oA = (torch.cat([o[i] for o in outs]) for i in range(4))
        agg["logD"].append(torch.log(evd["lb"]["D"] + 0.1).numpy()); agg["pD"].append(oD[:, 0].cpu().numpy())
        agg["d5"].append((evd["lb"]["D"] > 5.0).numpy()); agg["pd5"].append(torch.sigmoid(oD[:, 1]).cpu().numpy())
        agg["F"].append(evd["lb"]["F"].numpy()); agg["pF"].append(oF.cpu().numpy())
        agg["b1"].append(evd["lb"]["b1"].numpy() / 1000.0); agg["pb1"].append(oB.cpu().numpy())
        th = evd["lb"]["theta"].numpy(); ph = evd["lb"]["phi"].numpy()
        u = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
        agg["ang"].append(u)
        ua = oA.cpu().numpy(); agg["pang"].append(ua / (np.linalg.norm(ua, axis=1, keepdims=True) + 1e-6))
        agg["dom"].append(evd["lb"]["dom"].numpy())
    c = {k: np.concatenate(v) for k, v in agg.items()}
    r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    dm = c["dom"].astype(bool)
    # AUC for D>5
    from numpy import argsort
    y, s = c["d5"].astype(int), c["pd5"]
    o = argsort(s); r = np.empty_like(o, float); r[o] = np.arange(len(s))
    auc = (r[y == 1].mean() - (y.sum() - 1) / 2) / max((y == 0).sum(), 1) if 0 < y.sum() < len(y) else float("nan")
    res = dict(logD_R2=r2(c["logD"], c["pD"]), D5_AUC=auc, F_R2=r2(c["F"], c["pF"]),
               B1x_R2=r2(c["b1"][dm, 0], c["pb1"][dm, 0]), B1y_R2=r2(c["b1"][dm, 1], c["pb1"][dm, 1]),
               B1z_R2=r2(c["b1"][dm, 2], c["pb1"][dm, 2]),
               ang_med_deg=float(np.degrees(np.median(np.arccos(np.clip(
                   np.abs((c["ang"][dm] * c["pang"][dm]).sum(1)), 0, 1))))))
    return {k: round(float(v), 4) for k, v in res.items()}


@torch.no_grad()
def strat_eval(model_p, evds, miss, arm, mu, sd, chunk=65536):
    """B1y/z R^2 stratified by time-slice ambiguity m = mean #groups within +-5 ticks
    in the OTHER two planes (same volume). Separates information-limit (high at m~1,
    falls with m) from representation-limit (low even at m~1)."""
    rows = []
    for evd in evds:
        outs = []
        for s0 in range(0, evd["n"], chunk):
            sel = torch.arange(s0, min(s0 + chunk, evd["n"]))
            outs.append(model_p(*readout(evd, miss, arm, sel, mu, sd))[2].cpu())
        pb = torch.cat(outs).numpy()
        dom = evd["lb"]["dom"].numpy().astype(bool)
        b1 = evd["lb"]["b1"].numpy() / 1000.0
        gidp = evd["lb"]["gidp"].numpy().astype(int); t = evd["t"]
        g = evd["grp"]
        tm = {}
        for v in (0, 1):
            for p in range(3):
                tm[(v, p)] = np.sort(g["tmean"][(g["vol"] == v) & (g["plane"] == p)])
        m = np.zeros(len(gidp))
        for v in (0, 1):
            for p in range(3):
                selp = (gidp == v * 3 + p)
                if not selp.any():
                    continue
                cnt = np.zeros(selp.sum())
                for po in range(3):
                    if po == p:
                        continue
                    ts = tm[(v, po)]
                    cnt += (np.searchsorted(ts, t[selp] + 5) - np.searchsorted(ts, t[selp] - 5))
                m[selp] = cnt / 2.0
        rows.append(np.column_stack([m[dom], b1[dom, 1], b1[dom, 2], pb[dom, 1], pb[dom, 2]]))
    R = np.concatenate(rows)
    r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    out = {}
    for name, lo, hi in [("m<=2", 0, 2), ("m3-10", 3, 10), ("m11-30", 11, 30), ("m31-100", 31, 100), ("m>100", 101, 10 ** 9)]:
        sel = (R[:, 0] >= lo) & (R[:, 0] <= hi)
        if sel.sum() < 500:
            out[name] = dict(n=int(sel.sum())); continue
        out[name] = dict(n=int(sel.sum()), y=round(float(r2(R[sel, 1], R[sel, 3])), 3),
                         z=round(float(r2(R[sel, 2], R[sel, 4])), 3))
    return out


@torch.no_grad()
def fq_eval(model_p, evds, miss, arm, mu, sd, chunk=65536):
    """B1y/z R^2 UNWEIGHTED vs CHARGE-WEIGHTED, and stratified by F (top-1 fraction =
    per-pixel contributor cleanliness) and qtot (brightness). Tests whether the low
    uniform R^2 is label-noise averaging over faint/ambiguous pixels (info IS there
    for clean/bright pixels) vs a genuine per-token interface limit."""
    yz = []; pyz = []; F = []; Q = []
    for evd in evds:
        outs = []
        for s0 in range(0, evd["n"], chunk):
            sel = torch.arange(s0, min(s0 + chunk, evd["n"]))
            outs.append(model_p(*readout(evd, miss, arm, sel, mu, sd))[2].cpu())
        pb = torch.cat(outs).numpy()
        dom = evd["lb"]["dom"].numpy().astype(bool)
        b1 = evd["lb"]["b1"].numpy() / 1000.0
        yz.append(b1[dom][:, 1:3]); pyz.append(pb[dom][:, 1:3])
        F.append(evd["lb"]["F"].numpy()[dom]); Q.append(evd["lb"]["qtot"].numpy()[dom])
    yz = np.concatenate(yz); pyz = np.concatenate(pyz); F = np.concatenate(F); Q = np.concatenate(Q)

    def r2(y, p, w=None):
        if w is None:
            return 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
        yb = (w * y).sum() / w.sum()
        return 1 - (w * (y - p) ** 2).sum() / max((w * (y - yb) ** 2).sum(), 1e-9)

    out = {"overall": dict(n=len(F),
                           y_uw=round(float(r2(yz[:, 0], pyz[:, 0])), 3), z_uw=round(float(r2(yz[:, 1], pyz[:, 1])), 3),
                           y_cw=round(float(r2(yz[:, 0], pyz[:, 0], Q)), 3), z_cw=round(float(r2(yz[:, 1], pyz[:, 1], Q)), 3))}
    for nm, lo, hi in [("F.5-.6", .5, .6), ("F.6-.8", .6, .8), ("F.8-.95", .8, .95), ("F.95-1", .95, 1.01)]:
        s = (F >= lo) & (F < hi)
        out[nm] = dict(n=int(s.sum())) if s.sum() < 500 else dict(
            n=int(s.sum()), y=round(float(r2(yz[s, 0], pyz[s, 0])), 3), z=round(float(r2(yz[s, 1], pyz[s, 1])), 3))
    for nm, lo, hi in [("q250-500", 250, 500), ("q500-1k", 500, 1000), ("q1k-3k", 1000, 3000), ("q3k+", 3000, 1e12)]:
        s = (Q >= lo) & (Q < hi)
        out[nm] = dict(n=int(s.sum())) if s.sum() < 500 else dict(
            n=int(s.sum()), y=round(float(r2(yz[s, 0], pyz[s, 0])), 3), z=round(float(r2(yz[s, 1], pyz[s, 1])), 3))
    return out


@torch.no_grad()
def xprobe(evds, miss, arm, mu=None, sd=None, chunk=65536):
    """Zero-shot cross-plane same-gid retrieval on whitened cosine group embeddings.
    Group emb = (qtot*F)-weighted mean of member pixels' FEATURE readout (via lgid).
    Query: each U group; candidates: V groups, same volume, |tmean diff|<=5 ticks."""
    stats = dict(r1=0, r5=0, mrr=0.0, n=0, ncand=[]); emu = esd = None
    for evd in evds:
        if evd["fdim"] == 0 or "lgid" not in evd:
            return None
        fpart = evd["fdim"] * 4
        embs = {}
        gidp = evd["lb"]["gidp"].numpy(); lgid = evd["lgid"]
        w = (evd["lb"]["qtot"] * evd["lb"]["F"]).numpy()
        rows = {}
        for s0 in range(0, evd["n"], chunk):
            sel = torch.arange(s0, min(s0 + chunk, evd["n"]))
            X = readout(evd, miss, arm, sel, mu, sd)[0].cpu().numpy()
            for j, i in enumerate(sel.numpy()):
                key = (int(gidp[i]) // 3, int(gidp[i]) % 3, int(lgid[i]))    # (vol, plane, gid)
                a = rows.setdefault(key, [np.zeros(fpart), 0.0])
                a[0] += w[i] * X[j]; a[1] += w[i]
        for k, (v, ww) in rows.items():
            if ww > 0:
                embs[k] = v / ww
        if emu is None:                                           # emb-whiten stats (NOT the feature mu/sd!)
            allv = np.stack(list(embs.values())); emu = allv.mean(0); esd = allv.std(0) + 1e-6
        gt = {k: v for k, v in zip(zip(evd["grp"]["vol"], evd["grp"]["plane"], evd["grp"]["gid"]),
                                   evd["grp"]["tmean"])}
        for v in (0, 1):
            qs = [(g, t) for (vv, pp, g), t in gt.items() if vv == v and pp == 0 and (v, 0, g) in embs]
            cands = [(g, t) for (vv, pp, g), t in gt.items() if vv == v and pp == 1 and (v, 1, g) in embs]
            if not cands:
                continue
            cg = np.array([g for g, _ in cands]); ct = np.array([t for _, t in cands])
            C = np.stack([(embs[(v, 1, g)] - emu) / esd for g, _ in cands])
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
            for g, t in qs:
                m = np.abs(ct - t) <= 5.0
                if m.sum() < 2 or g not in cg[m]:
                    continue
                q = (embs[(v, 0, g)] - emu) / esd; q /= np.linalg.norm(q) + 1e-9
                sc = C[m] @ q
                order = np.argsort(-sc); ranked = cg[m][order]
                r = int(np.where(ranked == g)[0][0]) + 1
                stats["r1"] += r == 1; stats["r5"] += r <= 5; stats["mrr"] += 1.0 / r
                stats["n"] += 1; stats["ncand"].append(int(m.sum()))
    n = max(stats["n"], 1)
    return dict(X_R1=round(stats["r1"] / n, 4), X_R5=round(stats["r5"] / n, 4),
                X_MRR=round(stats["mrr"] / n, 4), X_n=stats["n"],
                X_medcand=int(np.median(stats["ncand"])) if stats["ncand"] else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["nll", "random", "raw", "geo"], required=True)
    ap.add_argument("--ckpt", default="ckpt_sc_w_d768.pt"); ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--dec", default="cross")
    ap.add_argument("--mnll", type=int, default=1); ap.add_argument("--tag", default="")
    ap.add_argument("--nox", action="store_true"); ap.add_argument("--strat", action="store_true")
    ap.add_argument("--fq", action="store_true"); ap.add_argument("--cwloss", action="store_true")
    ap.add_argument("--train", default="30000-30059"); ap.add_argument("--eval", default="30060-30079")
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--px", type=int, default=8192)
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(0)
    global PROBE_LAYER
    if a.layer > 0: PROBE_LAYER = a.layer   # probe an intermediate encoder block
    model = None
    if a.arm in ("nll", "random"):
        model = FMModel(128, 4, 6, d=a.d, blocks=a.blocks, dec_blocks=4, heads=a.heads, cond="film",
                        dec_mode=a.dec, nll=bool(a.mnll)).to(dev)
        if a.arm == "nll":
            model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
        model.eval()
    lo, hi = map(int, a.train.split("-")); el, eh = map(int, a.eval.split("-"))
    tr_ev = [e for e in range(lo, hi + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    ev_ev = [e for e in range(el, eh + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    print(f"arm={a.arm} tag={a.tag} ckpt={a.ckpt} d={a.d} heads={a.heads} train={len(tr_ev)} eval={len(ev_ev)}", flush=True)
    evds_tr = [load_event(e, a.arm, model) for e in tr_ev]
    evds_ev = [load_event(e, a.arm, model) for e in ev_ev]
    fdim = evds_tr[0]["fdim"]
    mu = sd = None
    if fdim > 0:                                                  # per-dim feature standardization (bug#6 fix)
        smp = torch.cat([e["feats"][:4096] for e in evds_tr[:10]])
        mu = smp.mean(0).to(dev); sd = (smp.std(0) + 1e-6).to(dev)
    in_feat = fdim * 4 if a.arm != "geo" else 0
    in_geo = 4 + 5 + 6 + 2
    miss = nn.Parameter(torch.zeros(4, fdim, device=dev)) if a.arm in ("nll", "random") else torch.zeros(4, 1)
    probe = Probe(in_feat, in_geo).to(dev)
    params = list(probe.parameters()) + ([miss] if isinstance(miss, nn.Parameter) else [])
    opt = torch.optim.AdamW(params, 1e-3, weight_decay=1e-5)
    for step in range(1, a.steps + 1):
        evd = evds_tr[np.random.randint(len(evds_tr))]
        sel = torch.tensor(np.random.randint(0, evd["n"], min(a.px, evd["n"])))
        Xf, Xg = readout(evd, miss, a.arm, sel, mu, sd)
        loss = losses(probe(Xf, Xg), evd["lb"], sel, cw=a.cwloss)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  step {step}: loss {float(loss):.4f}", flush=True)
            print(json.dumps(dict(arm=a.arm, tag=a.tag, step=step, **evaluate(probe, evds_ev, miss, a.arm, mu, sd))), flush=True)
    print("FINAL " + json.dumps(dict(arm=a.arm, tag=a.tag, **evaluate(probe, evds_ev, miss, a.arm, mu, sd))), flush=True)
    if a.strat:
        print("STRAT " + json.dumps(dict(arm=a.arm, tag=a.tag, **strat_eval(probe, evds_ev, miss, a.arm, mu, sd))), flush=True)
    if a.fq:
        print("FQ " + json.dumps(dict(arm=a.arm, tag=a.tag, cw=a.cwloss, **fq_eval(probe, evds_ev, miss, a.arm, mu, sd))), flush=True)
    if a.arm in ("nll", "random", "raw") and not a.nox:
        xr = xprobe(evds_ev, miss, a.arm, None, None)
        print("XPROBE " + json.dumps(dict(arm=a.arm, tag=a.tag, **(xr or {}))), flush=True)


if __name__ == "__main__":
    main()
