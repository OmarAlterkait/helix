"""Gradient noise scale B_noise = tr(Sigma)/|G|^2 (McCandlish et al.) with a LARGE pool + bootstrap CI,
and the full E[|G_B|^2] curve out to B=256 by directly forming batches (with-replacement iid draws,
modeling training that samples from ~80k events). Fit E[|G_B|^2]=|G|^2 + tr(Sigma)/B is a line in 1/B.

Point estimate from pool sufficient stats (Gram matrix), CI by bootstrapping the M events.
"""
import sys, os, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import data as D; D.init_pipeline_cpu()
from train import make_mask
from model import losses_fused
from probe_3d_mlp import build_serial
dev = "cuda"


def flat_grad(model):
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None])


def per_event_grad(model, B, mask):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        occ, mu, lv = model(B, mask)
        bce, val = losses_fused(occ, mu, lv, B, mask, vis_w=0.0, noisy=False)
    (bce + val).backward()
    return flat_grad(model).float()


def bnoise_from_gram(diag, total, idx, M):
    """Given precomputed per-event |g_i|^2 (diag) and Gram, compute B_noise for a subset `idx`."""
    m1 = diag[idx].mean()
    sub = total[np.ix_(idx, idx)]
    m2 = sub.sum() / (len(idx) ** 2)
    tr = (len(idx) / (len(idx) - 1)) * (m1 - m2)
    Gsq = m1 - tr
    return Gsq, tr, (tr / Gsq if Gsq > 0 else np.inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", default="ref")
    ap.add_argument("--M", type=int, default=384)
    ap.add_argument("--Bs", default="1,2,4,8,16,32,48,64,96,128,192,256")
    ap.add_argument("--nboot", type=int, default=500); ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--mask", type=float, default=0.75); ap.add_argument("--mask_mode", default="random")
    ap.add_argument("--lo", type=int, default=1000); ap.add_argument("--hi", type=int, default=81000)
    ap.add_argument("--hold", default="30000-30299"); ap.add_argument("--out", default="bnoise.jsonl")
    a = ap.parse_args()
    Bs = [int(x) for x in a.Bs.split(",")]

    model = build_serial(a.ckpt, randinit=False, rope_split=0, gp=1024, gd=2048)
    for p in model.parameters(): p.requires_grad_(True)
    cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/fm_cache_tpc")
    hlo, hhi = map(int, a.hold.split("-"))
    eid = lambda p: int("".join(filter(str.isdigit, os.path.basename(p))))
    pool = [p for p in glob.glob(os.path.join(cdir, "ev_*.npz"))
            if a.lo <= eid(p) < a.hi and not (hlo <= eid(p) <= hhi)]
    rng = np.random.default_rng(0)
    files = list(rng.choice(sorted(pool), size=a.M, replace=False))
    print(f"[{a.tag}] pool={len(pool)}, M={a.M}, mask={a.mask_mode}-{a.mask}, Bs={Bs}", flush=True)

    grads = []
    for i, f in enumerate(files):
        try:
            B = D.get_cached(f, device="cpu")
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {type(e).__name__}", flush=True); continue
        B = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in B.items()}
        m = make_mask(B, a.mask_mode, a.mask, 1)
        grads.append(per_event_grad(model, B, m).cpu().numpy())
        if (i + 1) % 32 == 0: print(f"  {i+1}/{a.M}", flush=True)
    G = np.stack(grads); M = G.shape[0]; del grads
    print(f"  computing Gram ({M}x{M}) ...", flush=True)
    Gram = G @ G.T                                              # (M,M)
    diag = np.diag(Gram).copy(); dim = G.shape[1]; del G

    # point estimate
    Gsq, tr, Bn = bnoise_from_gram(diag, Gram, np.arange(M), M)
    # bootstrap CI over events
    boots = []
    for _ in range(a.nboot):
        idx = rng.integers(0, M, size=M)
        boots.append(bnoise_from_gram(diag, Gram, idx, M)[2])
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, med, hi = np.percentile(boots, [16, 50, 84])

    # direct curve: with-replacement batches, |mean|^2 via Gram
    def EGB_direct(Bsz, K):
        vals = np.empty(K)
        for k in range(K):
            idx = rng.integers(0, M, size=Bsz)
            vals[k] = Gram[np.ix_(idx, idx)].sum() / (Bsz * Bsz)
        return vals.mean(), vals.std() / np.sqrt(K)

    print(f"\n[{a.tag}] dim={dim} M={M}")
    print(f"  |G|^2={Gsq:.5g}  tr(Sigma)={tr:.5g}")
    print(f"  B_noise = {Bn:.2f}   bootstrap 68% CI [{lo:.1f}, {hi:.1f}] (median {med:.1f})\n")
    print(f"  {'B':>4} {'E|G_B|^2 direct':>18} {'closed':>10} {'noise/signal':>13} {'steps-factor':>12}")
    curve = {}
    for Bsz in Bs:
        d, se = EGB_direct(Bsz, a.K); c = Gsq + tr / Bsz
        sf = 1 + Bn / Bsz                                       # S(B)/S_min : steps-to-target vs infinite batch
        print(f"  {Bsz:>4} {d:>13.5g}+-{se:.2g} {c:>10.5g} {c/Gsq:>13.3f} {sf:>12.2f}")
        curve[str(Bsz)] = {"direct": float(d), "se": float(se), "closed": float(c)}
    res = {"tag": a.tag, "M": int(M), "Gsq": float(Gsq), "trSigma": float(tr), "Bnoise": float(Bn),
           "ci68": [float(lo), float(hi)], "boot_med": float(med), "curve": curve}
    with open(a.out, "a") as fo: fo.write(json.dumps(res) + "\n")
    print(f"\nWROTE {a.tag}", flush=True)


if __name__ == "__main__":
    main()
