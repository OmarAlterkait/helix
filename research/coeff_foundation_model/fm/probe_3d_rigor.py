"""DISCRIMINATING 3D experiment (per Fable review). Fixes the protocol holes:
  - probe-val split + EARLY STOPPING (select on val per-event within-plane R^2) -> kills the
    overfit-driven negative floor and the "harder-to-memorize ranks higher" confound.
  - score against TWO baselines: pooled per-plane mean AND per-(event,plane) mean. The latter
    removes coarse event-context structure, so a positive R^2 there IS per-pixel triangulation.
  - RANDOM-INIT control (same arch, unloaded) + multi-seed -> measures the noise floor and the
    winner's-curse. Winner must beat random-init AND stay positive under per-event residual.
Runs at ONE (ckpt, layer): the winning config. Usage:
  python probe_3d_rigor.py --ckpt ckpt_mae_nll_s0_snap150000.pt --layer 12 --seeds 8"""
import sys, os, json, argparse, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from pb_probe import load_event_data, readout, LAB, dev
from pb_aw import fit_alongwire, u_target, UProbe
from model import FMModel


def build(ckpt, randinit, blocks=0, heads=0, key="model"):
    ck = torch.load(ckpt, map_location=dev)
    nslot = ck.get("n_slot", ck.get("pw", 16) * ck.get("pt", 8))       # per-ckpt patch geometry (p32x8 -> 256)
    m = FMModel(nslot, 4, 6, d=ck.get("d", 512), blocks=blocks or ck.get("blocks", 12), dec_blocks=ck.get("dec_blocks", 4),
                heads=heads or ck.get("heads", 4), cond=ck.get("cond", "film"), dec_mode=ck.get("dec_mode", "cross"),
                nll=ck.get("nll", False), wire_rope=ck.get("wire_rope", True)).to(dev)
    if not randinit:
        m.load_state_dict(ck[key])                                     # key='teacher' for JEPA (EMA), 'model' for MAE
    m.eval(); return m


def set_tokenizer_from_ckpt(ckpt):
    """Set vit_tpc/data patch globals to the ckpt's pw/pt BEFORE loading events, so eval tokens
    match the model's n_slot (p32x8 uses pw=32 -> n_slot=256; default 16x8 -> 128)."""
    ck = torch.load(ckpt, map_location="cpu")
    import vit_tpc as _vtp, data as _D
    pw, pt = ck.get("pw", 16), ck.get("pt", 8)
    _vtp.PW, _vtp.PT, _vtp.N_SLOT = pw, pt, pw * pt
    _D.N_SLOT = pw * pt
    print(f"tokenizer set from ckpt: pw={pw} pt={pt} n_slot={pw*pt} wire_rope={ck.get('wire_rope', True)}", flush=True)


def patchify_u(evd):
    """Granularity-match the label to the feature: hits that share the SAME 4-band patch-tuple
    get identical features, so replace each hit's sharp per-coeff u with the dom-weighted MEAN u
    over its group. Removes irreducible within-patch u-variance (the source of the strong-negative
    R^2) — the probe is then asked to predict only what the patch feature can possibly resolve."""
    n = evd["n"]
    keyparts = []
    for b in range(4):
        idxs, hit = evd["cols"][b]
        k = torch.where(hit, idxs, torch.full_like(idxs, -1)).numpy().astype(np.int64)
        keyparts.append(k)
    keymat = np.stack(keyparts, 1)                                  # (n,4): the feature-identity of each hit
    _, grp = np.unique(keymat, axis=0, return_inverse=True)
    u = evd["u"].numpy(); dom = evd["lb"]["dom"].numpy().astype(bool)
    ng = int(grp.max()) + 1
    ssum = np.zeros(ng); scnt = np.zeros(ng)
    np.add.at(ssum, grp[dom], u[dom]); np.add.at(scnt, grp[dom], 1.0)
    gmean = ssum / np.maximum(scnt, 1.0)
    up = gmean[grp].copy()
    nodom = scnt[grp] == 0                                          # groups with no dom hit: leave as-is (dom-masked anyway)
    up[nodom] = u[nodom]
    grpsz = scnt[scnt > 0]
    print(f"  patchify: {n} hits -> {int((scnt>0).sum())} dom-patches (mean {grpsz.mean():.1f} hits/patch)", flush=True)
    return torch.tensor(up, dtype=torch.float32)


def set_feats(raws, model, layer):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for r in raws:
            f = model.encode_layers(r["B"], {layer})[layer].float()
            r["feats"] = f; r["fdim"] = f.shape[1]


@torch.no_grad()
def collect(raws, probe, miss, mu, sd):
    U = []; P = []; G = []; E = []; DM = []
    for ei, evd in enumerate(raws):
        sel = torch.arange(evd["n"])
        Xf, Xg = readout(evd, miss, "nll", sel, mu, sd)
        P.append(probe(Xf, Xg).cpu().numpy()); U.append(evd["u"].numpy())
        G.append(evd["lb"]["gidp"].numpy().astype(int)); E.append(np.full(evd["n"], ei))
        DM.append(evd["lb"]["dom"].numpy().astype(bool))
    return (np.concatenate(U), np.concatenate(P), np.concatenate(G),
            np.concatenate(E), np.concatenate(DM))


def r2_two(U, P, G, E, dom):
    """within-plane R^2 vs (a) pooled per-plane mean, (b) per-(event,plane) mean."""
    base_pl = np.zeros_like(U); base_ep = np.zeros_like(U)
    for g in range(6):
        sg = dom & (G == g)
        if sg.any(): base_pl[G == g] = U[sg].mean()
        for e in np.unique(E):
            s = dom & (G == g) & (E == e)
            if s.any(): base_ep[(G == g) & (E == e)] = U[s].mean()
    num = ((U[dom] - P[dom]) ** 2).sum()
    r_pl = 1 - num / max(((U[dom] - base_pl[dom]) ** 2).sum(), 1e-9)
    r_ep = 1 - num / max(((U[dom] - base_ep[dom]) ** 2).sum(), 1e-9)
    return float(r_pl), float(r_ep)


def fit_es(fit_raws, val_raws, seed, steps=6000, ev=400, px=8192, patience=6):
    torch.manual_seed(seed); np.random.seed(seed)
    fdim = fit_raws[0]["fdim"]
    smp = torch.cat([e["feats"][:4096] for e in fit_raws[:10]]); mu = smp.mean(0).to(dev); sd = (smp.std(0) + 1e-6).to(dev)
    miss = nn.Parameter(torch.zeros(4, fdim, device=dev))
    probe = UProbe(fdim * 4, 4 + 5 + 6 + 2).to(dev)
    opt = torch.optim.AdamW(list(probe.parameters()) + [miss], 1e-3, weight_decay=1e-5)
    best = -1e9; best_sd = None; bad = 0
    for step in range(1, steps + 1):
        evd = fit_raws[np.random.randint(len(fit_raws))]
        sel = torch.tensor(np.random.randint(0, evd["n"], min(px, evd["n"])))
        Xf, Xg = readout(evd, miss, "nll", sel, mu, sd)
        pred = probe(Xf, Xg); dom = evd["lb"]["dom"][sel].to(dev)
        loss = ((pred - evd["u"][sel].to(dev)) ** 2 * dom).sum() / max(dom.sum(), 1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % ev == 0:                                              # EARLY STOP on val per-event R^2
            probe.eval()
            _, vep = r2_two(*collect(val_raws, probe, miss, mu, sd))
            probe.train()
            if vep > best:
                best = vep; best_sd = (copy.deepcopy(probe.state_dict()), miss.detach().clone()); bad = 0
            else:
                bad += 1
                if bad >= patience: break
    probe.load_state_dict(best_sd[0]); miss.data = best_sd[1]
    return probe, miss, mu, sd, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--layers", default="")                             # comma list -> scan (overrides --layer)
    ap.add_argument("--patch", type=int, default=0)                      # 1 = granularity-match label to patch (mean u/patch)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--arms", default="trained,random"); ap.add_argument("--tag", default="ref")
    ap.add_argument("--blocks", type=int, default=0); ap.add_argument("--heads", type=int, default=0)
    ap.add_argument("--key", default="model")                          # 'teacher' for JEPA EMA
    ap.add_argument("--train", default="30000-30059"); ap.add_argument("--eval", default="30060-30079")
    ap.add_argument("--out", default="probe_3d_rigor.jsonl")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else [a.layer]
    set_tokenizer_from_ckpt(a.ckpt)                                     # patch geometry BEFORE loading events
    lo, hi = map(int, a.train.split("-")); el, eh = map(int, a.eval.split("-"))
    tr = [e for e in range(lo, hi + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    ev = [e for e in range(el, eh + 1) if os.path.exists(os.path.join(LAB, f"pl_{e:05d}.npz"))]
    print(f"loading {len(tr)}+{len(ev)} events once...", flush=True)
    raw = [load_event_data(e) for e in tr]; raw_ev = [load_event_data(e) for e in ev]
    aw = fit_alongwire(raw)
    for r in raw + raw_ev: r["u"] = u_target(r, aw)
    if a.patch:                                                        # granularity-match: mean u per patch-group
        for r in raw + raw_ev: r["u"] = patchify_u(r)
    fit_raws, val_raws = raw[:50], raw[50:]                            # probe-val split (10 held-out train events)
    outf = open(a.out, "a")
    for arm in a.arms.split(","):
        model = build(a.ckpt, randinit=(arm == "random"), blocks=a.blocks, heads=a.heads, key=a.key)
        for layer in layers:
            for rr in (fit_raws, val_raws, raw_ev): set_feats(rr, model, layer)
            pes = []
            for seed in range(a.seeds):
                probe, miss, mu, sd, valbest = fit_es(fit_raws, val_raws, seed)
                r_pl, r_ep = r2_two(*collect(raw_ev, probe, miss, mu, sd))
                rec = dict(tag=a.tag, arm=arm, seed=seed, layer=layer, val_ep=round(valbest, 3),
                           eval_pooled=round(r_pl, 3), eval_per_event=round(r_ep, 3))
                print(f"  {a.tag} {arm:8s} L{layer} seed{seed}: pooled={r_pl:+.3f}  per_event={r_ep:+.3f}", flush=True)
                outf.write(json.dumps(rec) + "\n"); outf.flush(); pes.append(r_ep)
            print(f"SUMMARY {a.tag} {arm} L{layer}: per_event {np.mean(pes):+.3f}±{np.std(pes):.3f}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
