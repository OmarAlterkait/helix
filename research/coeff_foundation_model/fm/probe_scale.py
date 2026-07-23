"""Scale-shrinkage probe: does the MSE mean-regression (pred/tgt < 1 on large coeffs)
move toward 1.0 as the model scales? Loops the existing checkpoints, auto-infers
config from the state_dict, reports per-band var-explained + large-coeff shrinkage."""
import glob, torch, numpy as np
import data as D
D.init_pipeline_cpu()
from model import FMModel
dev = "cuda"
BN = ["A4", "D4", "D3", "D2"]; NB = 4
# (ckpt, heads): train.py runs use the FMModel default heads=4; the DDP vitb used 12
CKPTS = [("ckpt_fmmae_shallow.pt", 4), ("ckpt_fmmae_deep.pt", 4), ("ckpt_mae_vitb.pt", 12)]


def build_from(sd, heads):
    d = sd["embed.weight"].shape[0]
    enc = len({k.split(".")[1] for k in sd if k.startswith("enc.")})
    dec = len({k.split(".")[1] for k in sd if k.startswith("dec.")})
    dec_cross = any("dec.0.kv.weight" == k for k in sd)          # CrossBlock has .kv; Block has .qkv
    nll = sd["val_head.weight"].shape[0] == 2 * 128
    m = FMModel(128, NB, 6, d=d, blocks=enc, dec_blocks=dec, heads=heads,
                cond="film", dec_mode="cross" if dec_cross else "self", nll=nll).to(dev)
    m.load_state_dict(sd); m.eval()
    return m, dict(d=d, enc=enc, dec=dec, cross=dec_cross, nll=nll)


def main():
    paths = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[19000:19050]
    print(f"{'ckpt':22s} {'cfg':28s} {'overall':>7} | per-band VE% | large-coeff pred/tgt (>=2)")
    for f, heads in CKPTS:
        ck = torch.load(f, map_location=dev)
        model, cfg = build_from(ck["model"], heads)
        bse = np.zeros(NB); bsv = np.zeros(NB)
        at = np.zeros(NB); ap = np.zeros(NB); lc = np.zeros(NB)         # large-coeff |tgt|, |pred|, count
        gen = torch.Generator(device=dev).manual_seed(0)
        for p in paths:
            B = D.get_cached(p, device=dev); n = B["inp"].shape[0]
            mask = torch.rand(n, generator=gen, device=dev) < 0.75
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(B, mask); mu = out[1]
            cell, slot = B["cell"], B["slot"]; sel = mask[cell]
            pr = mu[cell, slot].float()[sel].cpu().numpy(); tg = B["target"].float()[sel].cpu().numpy()
            bd = B["band_id"][cell][sel].cpu().numpy()
            for b in range(NB):
                mb = bd == b
                bse[b] += ((pr[mb] - tg[mb]) ** 2).sum(); bsv[b] += (tg[mb] ** 2).sum()
                big = mb & (np.abs(tg) >= 2.0)
                at[b] += np.abs(tg[big]).sum(); ap[b] += np.abs(pr[big]).sum(); lc[b] += big.sum()
        ve = [(1 - bse[b] / bsv[b]) * 100 for b in range(NB)]
        overall = np.mean(ve)
        shrink = [ap[b] / max(at[b], 1e-9) for b in range(NB)]
        cfgs = f"d{cfg['d']} e{cfg['enc']} dec{cfg['dec']}{'X' if cfg['cross'] else 'S'}{'+nll' if cfg['nll'] else ''}"
        print(f"{f:22s} {cfgs:28s} {overall:6.1f}% | " +
              " ".join(f"{BN[b]}={ve[b]:.0f}" for b in range(NB)) + " | " +
              " ".join(f"{BN[b]}={shrink[b]:.2f}" for b in range(NB)))


if __name__ == "__main__":
    main()
