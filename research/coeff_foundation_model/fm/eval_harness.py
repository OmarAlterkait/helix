"""Standardized FM eval harness — all on TRULY held-out events (>=20000, never in
the 20k training set). For each checkpoint:
  (1) mask-generalization curve: var-explained + large-coeff shrinkage at a COMMON
      mask sweep {0.15,0.5,0.75,0.9} (decoupled from training mask), seeded.
  (2) frozen-feature LINEAR PROBE: ridge from the FULL-context encoder features to
      the clean coeff -> R^2 = representation quality (the metric that matters).
  (3) NLL calibration (if the head is heteroscedastic).
Fixes the earlier bug where probes sampled events 19000+ (which are in train_b)."""
import glob, torch, numpy as np
import data as D
D.init_pipeline_cpu()
from model import FMModel
dev = "cuda"; BN = ["A4", "D4", "D3", "D2"]; NB = 4
MASKS = [0.15, 0.5, 0.75, 0.9]
# (ckpt, heads): train.py runs default heads=4; DDP vitb used 12
CKPTS = [("ckpt_sc_w_d768.pt", 4, True), ("ckpt_mae_vitb.pt", 12, False)]   # (ckpt, heads, mup)
FIT = None; EVAL = None                                          # held-out event lists, set in main


def build(sd, heads, mup=False):
    d = sd["embed.weight"].shape[0]
    enc = len({k.split(".")[1] for k in sd if k.startswith("enc.")})
    dec = len({k.split(".")[1] for k in sd if k.startswith("dec.")})
    cross = any(k == "dec.0.kv.weight" for k in sd)
    nll = sd["val_head.weight"].shape[0] == 2 * 128
    m = FMModel(128, NB, 6, d=d, blocks=enc, dec_blocks=dec, heads=heads, cond="film",
                dec_mode="cross" if cross else "self", nll=nll, mup=mup, d_base=128).to(dev)   # mup: readout 1/m — WRONG head outputs without it
    m.load_state_dict(sd); m.eval()
    return m, dict(d=d, enc=enc, dec=dec, cross=cross, nll=nll)


def mask_curve(m):
    out = {}
    for mr in MASKS:
        bse = np.zeros(NB); bsv = np.zeros(NB); at = np.zeros(NB); ap = np.zeros(NB)
        gen = torch.Generator(device=dev).manual_seed(11)
        for p in EVAL:
            B = D.get_cached(p, device=dev); n = B["inp"].shape[0]
            mask = torch.rand(n, generator=gen, device=dev) < mr
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                mu = m(B, mask)[1]
            cell, slot = B["cell"], B["slot"]; sel = mask[cell]
            pr = mu[cell, slot].float()[sel].cpu().numpy(); tg = B["target"].float()[sel].cpu().numpy()
            bd = B["band_id"][cell][sel].cpu().numpy()
            for b in range(NB):
                mb = bd == b; big = mb & (np.abs(tg) >= 2.0)
                bse[b] += ((pr[mb] - tg[mb]) ** 2).sum(); bsv[b] += (tg[mb] ** 2).sum()
                at[b] += np.abs(tg[big]).sum(); ap[b] += np.abs(pr[big]).sum()
        ve = np.mean([(1 - bse[b] / bsv[b]) * 100 for b in range(NB)])
        sh = np.mean([ap[b] / max(at[b], 1e-9) for b in range(NB)])
        out[mr] = (ve, sh)
    return out


def linear_probe(m):
    """Ridge from FULL-context encoder features -> dense clean coeff; R^2 on active slots."""
    d = m.d; XtX = torch.zeros(d, d, device=dev); XtY = torch.zeros(d, 128, device=dev)
    for p in FIT:
        B = D.get_cached(p, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = m.encode(B).float()                         # (n_cells, d), full context, no mask
        Y = torch.zeros(feats.shape[0], 128, device=dev)
        Y[B["cell"], B["slot"]] = B["target"].float()
        XtX += feats.T @ feats; XtY += feats.T @ Y
    W = torch.linalg.solve(XtX + 1e-2 * torch.eye(d, device=dev), XtY)
    ss_res = 0.0; ss_tot = 0.0
    for p in EVAL:
        B = D.get_cached(p, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = m.encode(B).float()
        pred = (feats @ W)[B["cell"], B["slot"]]
        tg = B["target"].float()
        ss_res += ((pred - tg) ** 2).sum().item(); ss_tot += (tg ** 2).sum().item()
    return (1 - ss_res / ss_tot) * 100


def nll_calib(m):
    mbins = [0.0, 1.0, 2.0, 4.0, 100.0]; nb = len(mbins) - 1
    ap = np.zeros(nb); at = np.zeros(nb); s2 = np.zeros(nb); ae = np.zeros(nb); cnt = np.zeros(nb)
    gen = torch.Generator(device=dev).manual_seed(11)
    for p in EVAL:
        B = D.get_cached(p, device=dev); n = B["inp"].shape[0]
        mask = torch.rand(n, generator=gen, device=dev) < 0.75
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = m(B, mask)
        cell, slot = B["cell"], B["slot"]; sel = mask[cell]
        pr = mu[cell, slot].float()[sel].cpu().numpy(); tg = B["target"].float()[sel].cpu().numpy()
        sig2 = torch.exp(lv[cell, slot].float().clamp(-8, 8))[sel].cpu().numpy()
        amag = np.abs(tg); err = (pr - tg) ** 2
        for j in range(nb):
            m2 = (amag >= mbins[j]) & (amag < mbins[j + 1])
            ap[j] += np.abs(pr[m2]).sum(); at[j] += np.abs(tg[m2]).sum()
            s2[j] += sig2[m2].sum(); ae[j] += err[m2].sum(); cnt[j] += m2.sum()
    print("   NLL calib @0.75:  |tgt|-bin  pred/tgt  sig^2  err  sig^2/err")
    for j in range(nb):
        if cnt[j] == 0: continue
        print(f"      [{mbins[j]:.0f},{mbins[j+1]:<4.0f}) {ap[j]/max(at[j],1e-9):7.2f} {s2[j]/cnt[j]:7.2f} "
              f"{ae[j]/cnt[j]:6.2f} {s2[j]/max(ae[j],1e-9):8.2f}")


def main():
    global FIT, EVAL
    paths = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))
    FIT = paths[20000:20040]; EVAL = paths[20040:20090]          # BOTH held out from the 20k training
    print(f"HELD-OUT: fit={len(FIT)} eval={len(EVAL)} events (all >=20000)\n")
    print(f"{'checkpoint':22s} {'cfg':22s} {'probeR2':>7} | var-expl @ mask (shrink):  " + "  ".join(f"{m}" for m in MASKS))
    for f, heads, mup in CKPTS:
        ck = torch.load(f, map_location=dev); m, cfg = build(ck["model"], heads, mup)
        mc = mask_curve(m); pr2 = linear_probe(m)
        cfgs = f"d{cfg['d']}e{cfg['enc']}{'X' if cfg['cross'] else 'S'}{'+nll' if cfg['nll'] else ''}"
        print(f"{f.replace('ckpt_','').replace('.pt',''):22s} {cfgs:22s} {pr2:6.1f}% | " +
              "  ".join(f"{mc[mm][0]:.0f}%({mc[mm][1]:.2f})" for mm in MASKS))
        if cfg["nll"]:
            nll_calib(m)


if __name__ == "__main__":
    main()
