"""Fully-supervised deconvolution: train the model end-to-end to map noisy wire coeffs
-> true CHARGE coeffs (all tokens visible, no masking). This is the UPPER BOUND on how
well the architecture can deconvolve -> tells us the per-band ceiling.

If detail bands (D3/D2) stay ~0 even here, deconvolution of detail is information-limited
(response destroyed it); if they jump vs the frozen probe, the rep was the bottleneck.

Run:  python deconv_train.py --events 300 --steps 15000 --init scratch
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, math
import data as D
from data import DEV, N_SLOT, N_BAND
from model import FMModel

VAR_BANDS = ["A4", "D4", "D3", "D2"]


def move(B, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in B.items()}


def load_b(item):
    return D.get_cached_charge(*item) if isinstance(item, tuple) else item


@torch.no_grad()
def band_scales(batches):
    sq = np.zeros(N_BAND); n = np.zeros(N_BAND)
    for it in batches:
        B = load_b(it)
        c = B["target_charge"].cpu().numpy(); bd = B["band_id"][B["cell"]].cpu().numpy()
        np.add.at(sq, bd, c**2); np.add.at(n, bd, 1)
    return np.sqrt(sq / np.maximum(n, 1)) + 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=300)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--dec_blocks", type=int, default=4)
    ap.add_argument("--init", default="scratch", choices=["scratch", "mae"])
    ap.add_argument("--mae_ckpt", default="ckpt_dual44M.pt")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--ckpt", default="", help="checkpoint path (auto from --init if empty)")
    ap.add_argument("--resume", action="store_true", help="resume from --ckpt if present")
    ap.add_argument("--nll", action="store_true", help="Gaussian NLL head (mu+logvar) instead of MSE")
    ap.add_argument("--clean_input", action="store_true", help="ORACLE: feed CLEAN wire coeffs (no noise)")
    ap.add_argument("--ffn_mult", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    D.init_pipeline_cpu()
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_cache_tpc"))
    qdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_charge_tpc"))

    model = FMModel(N_SLOT, N_BAND, 6, n_wirefeat=1, d=args.d, blocks=args.blocks,
                    dec_blocks=args.dec_blocks, nll=args.nll, ffn_mult=args.ffn_mult).to(DEV)
    if args.init == "mae":
        sd = torch.load(os.path.join(here, args.mae_ckpt), map_location=DEV)["model"]
        msd = model.state_dict(); keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(keep, strict=False)            # skip val_head (nll changes its shape)
        print(f"MAE init: loaded {len(keep)}/{len(sd)} tensors (skipped {[k for k in sd if k not in keep]})", flush=True)
    npar = sum(p.numel() for p in model.parameters())
    print(f"deconv d={args.d} init={args.init} nll={args.nll} clean_input={args.clean_input} "
          f"ffn={args.ffn_mult} params={npar/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ckpt_path = args.ckpt or os.path.join(here, f"ckpt_deconv_{args.init}.pt")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):           # resume overrides the MAE init
        ck = torch.load(ckpt_path, map_location=DEV)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"RESUMED deconv from {ckpt_path} at step {start_step}", flush=True)

    def lr_at(s):
        if args.warmup and s < args.warmup:
            return args.lr * s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    have = [i for i in range(args.events)
            if os.path.exists(f"{qdir}/ev_charge_{i:05d}.npz") and os.path.exists(f"{cdir}/ev_{i:05d}.npz")]
    paths = [(f"{cdir}/ev_{i:05d}.npz", f"{qdir}/ev_charge_{i:05d}.npz") for i in have]
    lazy = len(paths) > 500                                  # RAM-safe for big sets
    allb = paths if lazy else [move(D.get_cached_charge(*p), "cpu") for p in paths]
    print(f"{'lazy' if lazy else 'staged'} loading {len(allb)} events", flush=True)
    nte = max(1, int(len(allb) * args.test_frac)); test_b, train_b = allb[:nte], allb[nte:]
    S = torch.tensor(band_scales(train_b[:200]), dtype=torch.float32, device=DEV)
    print(f"train={len(train_b)} test={len(test_b)} | S={S.cpu().numpy().round(1)}", flush=True)

    def tgt(B):
        return torch.asinh(B["target_charge"] / S[B["band_id"][B["cell"]]])

    def run(B):
        if args.clean_input:
            B = {**B, "inp": B["inp_clean"]}                  # ORACLE: clean wire coeffs as input
        m = torch.zeros(int(B["n_cells"]), dtype=torch.bool, device=DEV)   # all visible
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, m)
        return mu, lv

    def loss_fn(mu, lv, B):
        t = tgt(B); p = mu[B["cell"], B["slot"]].float()
        if args.nll:
            l = lv[B["cell"], B["slot"]].float().clamp(-8, 8)
            return 0.5 * ((p - t) ** 2 * torch.exp(-l) + l).mean()      # Gaussian NLL (no const)
        return ((p - t) ** 2).mean()

    @torch.no_grad()
    def eval_metrics(batches):
        """per band: R2 of mean, + (nll) Gaussian NLL & calibration coverage (|z|<1,<2)."""
        model.eval(); bd_, e_, t_, z_, nll_ = [], [], [], [], []
        for it in batches:
            B = move(load_b(it), DEV); mu, lv = run(B)
            p = mu[B["cell"], B["slot"]].float(); t = tgt(B)
            bd_.append(B["band_id"][B["cell"]].cpu().numpy())
            e_.append(((p - t) ** 2).cpu().numpy()); t_.append(t.cpu().numpy())
            if args.nll:
                l = lv[B["cell"], B["slot"]].float().clamp(-8, 8); s = torch.exp(0.5 * l)
                z_.append(((p - t) / s).cpu().numpy())
                nll_.append((0.5 * ((p - t) ** 2 * torch.exp(-l) + l + math.log(2 * math.pi))).cpu().numpy())
        model.train()
        bd = np.concatenate(bd_); e = np.concatenate(e_); tt = np.concatenate(t_)
        out = {}
        for b in range(N_BAND):
            mm = bd == b; o = {"r2": 1 - e[mm].mean() / (tt[mm].var() + 1e-9)}
            if args.nll:
                z = np.concatenate(z_)[mm]
                o["nll"] = float(np.concatenate(nll_)[mm].mean())
                o["cov1"] = float((np.abs(z) < 1).mean()); o["cov2"] = float((np.abs(z) < 2).mean())
            out[b] = o
        return out

    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        B = move(load_b(train_b[step % len(train_b)]), DEV)
        mu, lv = run(B); loss = loss_fn(mu, lv, B)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0 or step == start_step + 1:
            print(f"  step {step}: loss {float(loss):.4f} lr={lr_at(step):.1e} "
                  f"{(time.time()-t0)/(step-start_step)*1000:.0f} ms/step", flush=True)
        if step % args.eval_every == 0:
            M = eval_metrics(test_b[:100]); ov = np.mean([M[b]["r2"] for b in range(N_BAND)])
            line = f"   [eval {step}] R2 ov={ov*100:.1f}% | " + " ".join(f"{VAR_BANDS[b]}={M[b]['r2']*100:.1f}" for b in range(N_BAND))
            if args.nll:
                line += (f" || NLL={np.mean([M[b]['nll'] for b in range(N_BAND)]):.3f}"
                         " cov1[" + "/".join(f"{M[b]['cov1']*100:.0f}" for b in range(N_BAND)) + "]"
                         " cov2[" + "/".join(f"{M[b]['cov2']*100:.0f}" for b in range(N_BAND)) + "]")
            print(line, flush=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, d=args.d,
                            blocks=args.blocks, dec_blocks=args.dec_blocks, nll=args.nll), ckpt_path)
    print("done")


if __name__ == "__main__":
    main()
