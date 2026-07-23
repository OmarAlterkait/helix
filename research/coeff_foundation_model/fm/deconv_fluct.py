"""Fluctuation-sensitive evaluation of a trained deconv model (CPU).

The deconv MSE/NLL head predicts the conditional MEAN of the charge coeff -> it
regresses fluctuations toward the mean (var(pred) < var(true)). This script
quantifies that shortfall, the thing R^2 cannot see:
  - var(pred)/var(true) per band  (1.0 = fluctuations preserved; <1 = blurred)
  - CRPS of the predictive distribution (NLL ckpts) vs a degenerate point forecast
  - R^2 for reference
Run on several checkpoints (different training levels) to see how fluctuation
capture tracks training.  IMPORTANT: deconv ckpts were trained with the OLD
base=10000 RoPE -> instantiate with the matching wavelength band.

Run:  python deconv_fluct.py --ckpt ckpt_dc_d768_ref.pt --events 24
"""
import sys, os, math, argparse, numpy as np, torch
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from data import N_SLOT, N_BAND
from model import FMModel

VB = ["A4", "D4", "D3", "D2"]
OLD_LAM = (2 * math.pi, 47120.0)        # reproduces the base=10000 RoPE the ckpts trained with


def band_scales(items):
    sq = np.zeros(N_BAND); n = np.zeros(N_BAND)
    for it in items:
        B = D.get_cached_charge(*it, device="cpu")
        c = B["target_charge"].numpy(); bd = B["band_id"][B["cell"]].numpy()
        np.add.at(sq, bd, c ** 2); np.add.at(n, bd, 1)
    return np.sqrt(sq / np.maximum(n, 1)) + 1e-6


def crps_gauss(mu, sig, y):                 # closed-form Gaussian CRPS (torch)
    z = (y - mu) / sig
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return sig * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--events", type=int, default=24)
    ap.add_argument("--first", type=int, default=0, help="start index (use a held-out slice)")
    args = ap.parse_args()
    D.init_pipeline_cpu()
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_cache_tpc"))
    qdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_charge_tpc"))
    ck = torch.load(os.path.join(here, args.ckpt), map_location="cpu")
    d = ck.get("d", 512); blocks = ck.get("blocks", 10); dec = ck.get("dec_blocks", 4); nll = ck.get("nll", False)
    model = FMModel(N_SLOT, N_BAND, 6, n_wirefeat=1, d=d, blocks=blocks, dec_blocks=dec,
                    nll=nll, cond="film", lam_t=OLD_LAM, lam_w=OLD_LAM)
    model.load_state_dict(ck["model"]); model.eval()

    have = [i for i in range(args.first, args.first + 4000)
            if os.path.exists(f"{qdir}/ev_charge_{i:05d}.npz") and os.path.exists(f"{cdir}/ev_{i:05d}.npz")]
    paths = [(f"{cdir}/ev_{i:05d}.npz", f"{qdir}/ev_charge_{i:05d}.npz") for i in have[:args.events]]
    S = torch.tensor(band_scales(paths[:min(20, len(paths))]), dtype=torch.float32)

    P = [[] for _ in range(N_BAND)]; T = [[] for _ in range(N_BAND)]; SG = [[] for _ in range(N_BAND)]
    with torch.no_grad():
        for it in paths:
            B = D.get_cached_charge(*it, device="cpu")
            m = torch.zeros(int(B["n_cells"]), dtype=torch.bool)
            _, mu, lv = model(B, m)
            bd = B["band_id"][B["cell"]].numpy()
            tgt = torch.asinh(B["target_charge"] / S[B["band_id"][B["cell"]]])
            pr = mu[B["cell"], B["slot"]].float()
            for b in range(N_BAND):
                mm = bd == b; P[b].append(pr[mm]); T[b].append(tgt[mm])
                if nll:
                    sg = torch.exp(0.5 * lv[B["cell"], B["slot"]].float()[mm].clamp(-8, 8)); SG[b].append(sg)
    print(f"\n{args.ckpt}: d={d} blocks={blocks} nll={nll}  ({len(paths)} events, step={ck.get('step','?')})")
    hdr = f"{'band':>5} {'R2':>6} {'var(pred)/var(true)':>19}"
    if nll: hdr += f" {'CRPS':>7} {'CRPS_pt':>8} {'cov1':>5}"
    print(hdr)
    ov = {}
    for b in range(N_BAND):
        p = torch.cat(P[b]); t = torch.cat(T[b])
        r2 = 1 - ((p - t) ** 2).mean() / (t.var() + 1e-9); vr = p.var() / (t.var() + 1e-9)
        line = f"{VB[b]:>5} {r2*100:>5.1f} {vr:>19.3f}"
        if nll:
            sg = torch.cat(SG[b])
            crps = crps_gauss(p, sg, t).mean()
            crps_pt = crps_gauss(p, torch.full_like(sg, 1e-3), t).mean()    # ~point-forecast (= MAE)
            cov1 = ((t - p).abs() < sg).float().mean()
            line += f" {crps:>7.3f} {crps_pt:>8.3f} {cov1*100:>4.0f}%"
        print(line)
    print(f"  -> var-ratio mean: {np.mean([float(torch.cat(P[b]).var()/(torch.cat(T[b]).var()+1e-9)) for b in range(N_BAND)]):.3f}"
          "  (1.0=fluctuations preserved, <1=mean blurs them)")


if __name__ == "__main__":
    main()
