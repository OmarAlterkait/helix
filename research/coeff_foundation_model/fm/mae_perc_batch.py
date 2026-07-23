"""Batched Perceiver MAE pretraining. Mask `mask_ratio` of tokens, encode the
visible ones, reconstruct the MASKED tokens' coeffs (on their occupied slots).
--head value  -> Linear + MSE recon (clean architecture test vs FMModel MAE)
--head flow   -> flow-matching head (distributional; position-only decode means a
                 genuinely stochastic conditional, unlike the deconv decode).
Same SLURM/config/resume machinery as deconv_perc_batch (atomic full-state
checkpoint, --resume, self-terminate). Metric: var-explained on masked coeffs
per band (+ var-ratio for the flow head).
"""
import sys, os, math, time, argparse, random, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from data import DEV, N_SLOT, N_BAND
from perceiver_mae import PerceiverMAE, make_mask
from flow_head import FlowHead, flow_loss, flow_sample
VB = ["A4", "D4", "D3", "D2"]
TGT_VAR_BAND = {0: 14.00, 1: 13.83, 2: 11.51, 3: 6.11}    # clean-target variance per band (matches FMModel train.py)
NOISY_VAR_BAND = {0: 15.06, 1: 14.23, 2: 11.99, 3: 7.17}  # noisy-target variance (self-supervised; no clean truth)


def move(B, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in B.items()}


def _collate(x):
    return x


class Cached(torch.utils.data.Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return D.get_cached(self.items[i], device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=20000); ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=24); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--head", choices=["value", "flow"], default="value")
    ap.add_argument("--flow_steps", type=int, default=4); ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=4000); ap.add_argument("--ckpt", default="ckpt_mae.pt")
    ap.add_argument("--ckpt_every", type=int, default=1000, help="frequent cheap checkpoint (preemption safety, decoupled from eval)")
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--noisy_target", action="store_true", help="self-supervised: predict NOISY input coeff (no clean truth)")
    ap.add_argument("--parallel_decode", action="store_true", help="decode: query cout(latents) & cout2(visible) with the CLEAN address, fuse (vs sequential)")
    ap.add_argument("--rope_cout2", action="store_true", help="axial RoPE (relative position) in the visible-token cross-attention cout2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config", default=None)
    _pre, _ = ap.parse_known_args()
    if _pre.config:
        import yaml
        with open(_pre.config) as _f:
            _cfg = yaml.safe_load(_f) or {}
        for _k in ("script", "hours"): _cfg.pop(_k, None)
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in _cfg.items()})
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0); D.init_pipeline_cpu()
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_cache_tpc"))

    model = PerceiverMAE(N_SLOT, N_BAND, 6, d=args.d, M=args.M, depth=args.depth, heads=args.heads,
                         parallel_decode=args.parallel_decode, rope_cout2=args.rope_cout2).to(DEV)
    head = FlowHead(args.d, N_SLOT).to(DEV) if args.head == "flow" else nn.Linear(args.d, N_SLOT).to(DEV)
    npar = (sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())) / 1e6
    print(f"Perceiver-MAE head={args.head} d={args.d} M={args.M} depth={args.depth} batch={args.batch} mask={args.mask_ratio} params={npar:.1f}M", flush=True)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)

    def lr_at(s):
        if s < args.warmup: return args.lr * s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    have = [i for i in range(args.events) if os.path.exists(f"{cdir}/ev_{i:05d}.npz")]
    paths = [f"{cdir}/ev_{i:05d}.npz" for i in have]
    nte = max(args.batch, int(len(paths) * 0.1)); test_b, train_b = paths[:nte], paths[nte:]
    print(f"train={len(train_b)} test={len(test_b)} head={args.head}", flush=True)
    from torch.utils.data import DataLoader
    loader = DataLoader(Cached(train_b), batch_size=args.batch, shuffle=True, collate_fn=_collate,
                        num_workers=args.workers, persistent_workers=True, prefetch_factor=4,
                        worker_init_fn=D._worker_init, multiprocessing_context="spawn")
    it = iter(loader)
    gen = torch.Generator(device=DEV)

    def masks_for(Bs, step):
        return [make_mask(int(B["n_cells"]), args.mask_ratio, DEV, gen.manual_seed(step * 9973 + j)) for j, B in enumerate(Bs)]

    def dense_clean(B):                                       # CLEAN target -> dense (n_cells, n_slot)
        y = torch.zeros(int(B["n_cells"]), N_SLOT, device=DEV)
        y[B["cell"], B["slot"]] = B["target"]
        return y

    def loss_one(z, B, mk):                                   # masked DENOISING: predict CLEAN target at masked active slots
        cm = mk; om = B["occ"][cm].bool()
        y = B["inp"] if args.noisy_target else dense_clean(B)     # noisy self-sup vs clean denoising
        if args.head == "flow":
            return flow_loss(head, z[cm].float(), y[cm], om)
        recon = head(z[cm].float())
        return ((recon - y[cm]) ** 2)[om].mean()

    @torch.no_grad()
    def evaluate(items, n=60):                                # var_expl vs CLEAN target, SAME metric as FMModel
        model.eval(); head.eval()
        se = np.zeros(N_BAND); cnt = np.zeros(N_BAND); vr_n = np.zeros(N_BAND); vr_d = np.zeros(N_BAND)
        for p in items[:n]:
            B = move(D.get_cached(p, device="cpu"), DEV)
            mk = make_mask(int(B["n_cells"]), args.mask_ratio, DEV, gen.manual_seed(12345))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.forward_feat(B, mk).float()
            om = B["occ"][mk].bool().cpu().numpy()
            tgt = (B["inp"] if args.noisy_target else dense_clean(B))[mk].cpu().numpy()
            bd = B["band_id"][mk].cpu().numpy()
            if args.head == "flow":
                s = flow_sample(head, z[mk], N_SLOT, steps=args.flow_steps, k=args.samples).cpu().numpy()  # (k,nm,slot)
                pred = s.mean(0)
            else:
                pred = head(z[mk]).cpu().numpy(); s = None
            for b in range(N_BAND):
                rows = bd == b
                if not rows.any(): continue
                mm = om[rows]; tt = tgt[rows]; pp = pred[rows]
                se[b] += ((pp - tt) ** 2)[mm].sum(); cnt[b] += mm.sum()
                if s is not None:
                    ss = s[:, rows]; vr_n[b] += ss.var(0)[mm].sum(); vr_d[b] += (tt[mm] ** 2).sum()
        model.train(); head.train()
        out = {}; VARB = NOISY_VAR_BAND if args.noisy_target else TGT_VAR_BAND
        for b in range(N_BAND):
            mse = se[b] / max(cnt[b], 1)
            out[b] = dict(ve=(VARB[b] - mse) / VARB[b],                     # SAME var_expl form as FMModel
                          vr=(vr_n[b] / max(vr_d[b], 1e-9)) if args.head == "flow" else float("nan"))
        return out

    ckpt_path = os.path.join(here, args.ckpt)

    def save_ckpt(step):
        tmp = ckpt_path + ".tmp"
        torch.save(dict(model=model.state_dict(), head=head.state_dict(), opt=opt.state_dict(),
                        step=step, steps=args.steps, d=args.d, M=args.M, depth=args.depth, head_kind=args.head,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                        np_rng=np.random.get_state(), py_rng=random.getstate()), tmp)
        os.replace(tmp, ckpt_path)

    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEV)
        if "opt" not in ck:
            print(f"  (ckpt {args.ckpt} old-format -> FRESH)", flush=True)
        else:
            try:
                model.load_state_dict(ck["model"]); head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"])
                start_step = int(ck["step"])
                try:
                    tr = ck["torch_rng"]; torch.set_rng_state(tr.cpu().to(torch.uint8) if torch.is_tensor(tr) else tr)
                    torch.cuda.set_rng_state_all(ck["cuda_rng"]); np.random.set_state(ck["np_rng"]); random.setstate(ck["py_rng"])
                except Exception as _e:
                    print(f"  (RNG restore skipped: {_e})", flush=True)
                print(f"RESUMED from step {start_step}/{args.steps} (ckpt {args.ckpt})", flush=True)
                if start_step >= args.steps:
                    open(ckpt_path + ".done", "w").close(); print("already complete; nothing to do."); return
            except Exception as _e:
                print(f"  (resume failed: {_e} -> FRESH)", flush=True); start_step = 0

    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        try: Bs = next(it)
        except StopIteration: it = iter(loader); Bs = next(it)
        Bs = [move(B, DEV) for B in Bs]
        masks = masks_for(Bs, step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zs = model.forward_batch(Bs, masks, ckpt=not args.no_ckpt)
        loss = torch.stack([loss_one(z, B, mk) for z, B, mk in zip(zs, Bs, masks)]).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0); opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  step {step}: loss {float(loss):.4f} lr={lr_at(step):.1e} {(time.time()-t0)/max(step-start_step,1)*1000:.0f} ms/step", flush=True)
        if step % args.eval_every == 0:
            M = evaluate(test_b); ve = np.mean([M[b]["ve"] for b in range(N_BAND)])
            extra = (" varRatio=%.3f" % np.mean([M[b]["vr"] for b in range(N_BAND)])) if args.head == "flow" else ""
            print(f"   [eval {step}] {'noisy' if args.noisy_target else 'clean'}VarExpl ov={ve*100:.1f}%{extra} | "
                  + " ".join(f"{VB[b]}(ve={M[b]['ve']*100:.0f}" + (",vr=%.2f" % M[b]['vr'] if args.head == "flow" else "") + ")" for b in range(N_BAND)), flush=True)
            save_ckpt(step)
        elif step % args.ckpt_every == 0:                      # frequent cheap checkpoint (preemption safety)
            save_ckpt(step)
    save_ckpt(args.steps); open(ckpt_path + ".done", "w").close(); print("done")


if __name__ == "__main__":
    main()
