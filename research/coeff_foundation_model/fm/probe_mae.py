"""Probe a successful MAE (mae_vitb, 74%) — decompose the MASKED-prediction loss by
band AND by target-coefficient magnitude, to find WHERE it fails:
- is the residual on SMALL (noise-like) coeffs (=> irreducible floor), or
- is it UNDER-predicting LARGE (signal) coeffs (=> regression-to-mean / blurring)?
Also: occupancy prediction quality on masked tokens (does it know WHERE coeffs are?)."""
import glob, torch, numpy as np
import data as D
D.init_pipeline_cpu()
from model import FMModel
dev = "cuda"
BN = ["A4", "D4", "D3", "D2"]; NB = 4
mbins = [0.0, 0.5, 1.0, 2.0, 4.0, 100.0]; nm = len(mbins) - 1


def main():
    ck = torch.load("ckpt_mae_vitb.pt", map_location=dev)
    model = FMModel(128, NB, 6, d=768, blocks=12, dec_blocks=4, heads=12, cond="film", dec_mode="self").to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    paths = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))
    test = paths[19000:19060]                                   # held-out slice (training used earlier events)

    se = np.zeros((NB, nm)); st2 = np.zeros((NB, nm)); at = np.zeros((NB, nm)); ap = np.zeros((NB, nm)); cnt = np.zeros((NB, nm))
    bse = np.zeros(NB); bsv = np.zeros(NB)
    occ_tp = occ_fp = occ_fn = occ_tn = 0
    gen = torch.Generator(device=dev).manual_seed(0)
    for p in test:
        B = D.get_cached(p, device=dev); n = B["inp"].shape[0]
        mask = torch.rand(n, generator=gen, device=dev) < 0.75
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, mask)
        cell, slot = B["cell"], B["slot"]
        sel = mask[cell]
        pr = mu[cell, slot].float()[sel].cpu().numpy()
        tg = B["target"].float()[sel].cpu().numpy()
        bd = B["band_id"][cell][sel].cpu().numpy()
        err = (pr - tg) ** 2; amag = np.abs(tg)
        for b in range(NB):
            mb = bd == b
            bse[b] += err[mb].sum(); bsv[b] += (tg[mb] ** 2).sum()
            for j in range(nm):
                m2 = mb & (amag >= mbins[j]) & (amag < mbins[j + 1])
                se[b, j] += err[m2].sum(); st2[b, j] += (tg[m2] ** 2).sum()
                at[b, j] += np.abs(tg[m2]).sum(); ap[b, j] += np.abs(pr[m2]).sum(); cnt[b, j] += m2.sum()
        # occupancy on MASKED valid slots: does the model predict WHERE coeffs are?
        mvalid = mask[:, None] & B["valid"]
        pocc = (torch.sigmoid(occ)[mvalid] > 0.5).cpu().numpy()
        tocc = B["occ"][mvalid].bool().cpu().numpy()
        occ_tp += (pocc & tocc).sum(); occ_fp += (pocc & ~tocc).sum()
        occ_fn += (~pocc & tocc).sum(); occ_tn += (~pocc & ~tocc).sum()

    print(f"overall var_expl (sanity vs run 74.3%): {np.mean([1 - bse[b]/bsv[b] for b in range(NB)])*100:.1f}%")
    print(f"per-band var_expl: " + "  ".join(f"{BN[b]}={((1-bse[b]/bsv[b])*100):.1f}%" for b in range(NB)))
    tot = se.sum()
    print("\nband  |tgt|-bin      rows%   mean|tgt|  mean|pred|  pred/tgt   recon%(1-se/E[t^2])  %of-total-err")
    for b in range(NB):
        for j in range(nm):
            if cnt[b, j] == 0: continue
            mt = at[b, j] / cnt[b, j]; mp = ap[b, j] / cnt[b, j]
            recon = (1 - se[b, j] / max(st2[b, j], 1e-9)) * 100
            print(f"{BN[b]:4s}  [{mbins[j]:.1f},{mbins[j+1]:<4.0f}) {cnt[b,j]/cnt.sum()*100:6.1f}%  "
                  f"{mt:8.2f}  {mp:9.2f}  {mp/max(mt,1e-9):8.2f}   {recon:14.1f}   {se[b,j]/tot*100:10.1f}%")
    tot_occ = occ_tp + occ_fp + occ_fn + occ_tn
    prec = occ_tp / max(occ_tp + occ_fp, 1); rec = occ_tp / max(occ_tp + occ_fn, 1)
    print(f"\nMASKED occupancy prediction: acc={(occ_tp+occ_tn)/tot_occ*100:.1f}%  "
          f"precision={prec*100:.1f}%  recall={rec*100:.1f}%  (base rate active={ (occ_tp+occ_fn)/tot_occ*100:.1f}%)")


if __name__ == "__main__":
    main()
