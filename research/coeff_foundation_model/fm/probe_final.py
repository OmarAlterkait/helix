"""Completes the MSE-bottleneck proof:
 A) shrinkage vs mask — does pred/tgt -> 1 as mask drops (context removes hedging)?
 B) NLL calibration  — is predicted sigma^2 LARGE exactly where pred/tgt shrinks,
    and calibrated (sigma^2 ~ actual squared error)?  => shrinkage is uncertainty."""
import glob, torch, numpy as np
import data as D
D.init_pipeline_cpu()
from model import FMModel
dev = "cuda"; BN = ["A4", "D4", "D3", "D2"]; NB = 4
mbins = [0.0, 0.5, 1.0, 2.0, 4.0, 100.0]; nm = len(mbins) - 1


def load(f, nll=False):
    m = FMModel(128, NB, 6, d=512, blocks=12, dec_blocks=4, heads=4, cond="film", dec_mode="cross", nll=nll).to(dev)
    m.load_state_dict(torch.load(f, map_location=dev)["model"]); m.eval(); return m


def main():
    paths = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[19000:19050]
    gen = torch.Generator(device=dev).manual_seed(0)

    print("=== A) shrinkage vs mask (each model eval'd at its TRAINING mask) ===")
    print(f"{'model':20s} {'mask':>5} {'overall':>7} | large-coeff pred/tgt: A4 D4 D3 D2")
    for f, mr in [("ckpt_test_mask015.pt", 0.15), ("ckpt_test_mask055.pt", 0.55),
                  ("ckpt_test_mask075.pt", 0.75), ("ckpt_test_mask090.pt", 0.90)]:
        m = load(f)
        bse = np.zeros(NB); bsv = np.zeros(NB); at = np.zeros(NB); ap = np.zeros(NB)
        for p in paths:
            B = D.get_cached(p, device=dev); n = B["inp"].shape[0]
            mask = torch.rand(n, generator=gen, device=dev) < mr
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                mu = m(B, mask)[1]
            cell, slot = B["cell"], B["slot"]; sel = mask[cell]
            pr = mu[cell, slot].float()[sel].cpu().numpy(); tg = B["target"].float()[sel].cpu().numpy()
            bd = B["band_id"][cell][sel].cpu().numpy()
            for b in range(NB):
                mb = bd == b; big = mb & (np.abs(tg) >= 2.0)
                bse[b] += ((pr[mb]-tg[mb])**2).sum(); bsv[b] += (tg[mb]**2).sum()
                at[b] += np.abs(tg[big]).sum(); ap[b] += np.abs(pr[big]).sum()
        ve = np.mean([(1-bse[b]/bsv[b])*100 for b in range(NB)])
        sh = [ap[b]/max(at[b], 1e-9) for b in range(NB)]
        print(f"{f.replace('ckpt_test_','').replace('.pt',''):20s} {mr:5.2f} {ve:6.1f}% | " + " ".join(f"{sh[b]:.2f}" for b in range(NB)))

    print("\n=== B) NLL calibration (mask 0.75): per |tgt|-bin -> pred/tgt, mean sigma^2, actual err ===")
    m = load("ckpt_test_nll.pt", nll=True)
    at = np.zeros(nm); ap = np.zeros(nm); s2 = np.zeros(nm); ae = np.zeros(nm); cnt = np.zeros(nm)
    for p in paths:
        B = D.get_cached(p, device=dev); n = B["inp"].shape[0]
        mask = torch.rand(n, generator=gen, device=dev) < 0.75
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = m(B, mask)
        cell, slot = B["cell"], B["slot"]; sel = mask[cell]
        pr = mu[cell, slot].float()[sel].cpu().numpy(); tg = B["target"].float()[sel].cpu().numpy()
        sig2 = torch.exp(lv[cell, slot].float().clamp(-8, 8))[sel].cpu().numpy()
        err = (pr-tg)**2; amag = np.abs(tg)
        for j in range(nm):
            m2 = (amag >= mbins[j]) & (amag < mbins[j+1])
            at[j] += np.abs(tg[m2]).sum(); ap[j] += np.abs(pr[m2]).sum()
            s2[j] += sig2[m2].sum(); ae[j] += err[m2].sum(); cnt[j] += m2.sum()
    print(f"{'|tgt|-bin':12s} {'rows%':>6} {'pred/tgt':>9} {'mean_sig^2':>11} {'actual_err':>11} {'calib(s2/err)':>13}")
    for j in range(nm):
        if cnt[j] == 0: continue
        pt = ap[j]/max(at[j], 1e-9); ms2 = s2[j]/cnt[j]; mae = ae[j]/cnt[j]
        print(f"[{mbins[j]:.1f},{mbins[j+1]:<4.0f}) {cnt[j]/cnt.sum()*100:5.1f}% {pt:9.2f} {ms2:11.2f} {mae:11.2f} {ms2/max(mae,1e-9):13.2f}")


if __name__ == "__main__":
    main()
