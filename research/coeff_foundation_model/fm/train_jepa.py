"""I-JEPA-style LATENT-TARGET objective (A/B vs MAE-MSE). Same architecture, masking,
RoPE, decoder — only the TARGET changes:
  student.forward_feat(B, mask)[masked]   (predict masked-token latents from visible context)
  == LN( teacher.encode(B) )[masked].detach()   (EMA-teacher's full-context latents)
Loss = smooth-L1 in LATENT space. Teacher = EMA of student. No coefficient reconstruction.
The literature's proven lever for frozen-representation quality (I-JEPA pixel 40.7 -> latent 66.9)."""
import argparse, os, json, glob, copy, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import data as D
D.init_pipeline_cpu()
from model import FMModel
from torch.utils.data import DataLoader
DEV = "cuda"


def move(B, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in B.items()}


def sigreg(z, K=256, nt=16, tmax=5.0, nsub=8192):
    """SIGReg (LeJEPA): push the embedding cloud toward isotropic N(0,I) via K random 1D
    projections + an Epps-Pulley characteristic-function test (Cramer-Wold). Provable
    anti-collapse. z:(N,d) -> scalar. Per-dim standardize first so it targets isotropy+Gaussianity."""
    N, d = z.shape
    if N > nsub:
        z = z[torch.randperm(N, device=z.device)[:nsub]]
    z = (z - z.mean(0)) / (z.std(0) + 1e-6)
    u = torch.randn(d, K, device=z.device); u = u / (u.norm(dim=0, keepdim=True) + 1e-8)
    p = z @ u                                                     # (n,K) marginals along random dirs
    t = torch.linspace(0.2, tmax, nt, device=z.device)
    tp = p[:, :, None] * t[None, None, :]
    re = torch.cos(tp).mean(0); im = torch.sin(tp).mean(0)        # empirical characteristic fn
    g = torch.exp(-0.5 * t * t)                                   # target = N(0,1) CF (real); imag 0
    return (((re - g[None, :]) ** 2 + im ** 2) * g[None, :]).sum(-1).mean()


@torch.no_grad()
def rankme(z):
    """RankMe (arXiv:2210.02885): effective rank of features = exp(entropy of singular-value
    distribution). Head-free unsupervised representation-quality proxy (higher = richer)."""
    z = (z - z.mean(0)).float()
    s = torch.linalg.svdvals(z); pk = s / (s.sum() + 1e-12)
    return float(torch.exp(-(pk * torch.log(pk + 1e-12)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    _pre, _ = ap.parse_known_args()
    cfg = {}
    if _pre.config:
        import yaml
        cfg = yaml.safe_load(open(_pre.config)) or {}
    ap.add_argument("--cache_dir", default="../artifacts/fm_cache_tpc")
    ap.add_argument("--events", type=int, default=21000); ap.add_argument("--val_n", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--dec_blocks", type=int, default=4)
    ap.add_argument("--mask", type=float, default=0.75); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--mup", type=int, default=1); ap.add_argument("--d_base", type=int, default=128)
    ap.add_argument("--momentum", type=float, default=0.996)
    ap.add_argument("--sigreg", type=float, default=0.0, help="LeJEPA SIGReg weight (0=plain I-JEPA)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=6000); ap.add_argument("--snap_every", type=int, default=0)
    ap.add_argument("--tag", default="jepa"); ap.add_argument("--ckpt", default="ckpt_jepa.pt")
    ap.add_argument("--resume", action="store_true")
    ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    student = FMModel(128, 4, 6, n_wirefeat=1, d=a.d, blocks=a.blocks, dec_blocks=a.dec_blocks, heads=a.heads,
                      cond="film", dec_mode="cross", nll=False, mup=bool(a.mup), d_base=a.d_base).to(DEV)
    teacher = copy.deepcopy(student).to(DEV)
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(student.param_groups(a.lr), lr=a.lr, weight_decay=1e-4)
    _ratio = [pg["lr"] / a.lr for pg in opt.param_groups]
    npar = sum(p.numel() for p in student.parameters())

    files = sorted(glob.glob(os.path.join(a.cache_dir, "ev_*.npz")),
                   key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p)))))[:a.events]
    test_b, train_b = files[:a.val_n], files[a.val_n:]
    print(f"JEPA d={a.d} heads={a.heads} hd={a.d//a.heads} params={npar/1e6:.1f}M | train={len(train_b)} "
          f"val={len(test_b)} mask={a.mask} mom0={a.momentum}", flush=True)

    start = 0
    if a.resume and os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location=DEV)
        student.load_state_dict(ck["model"]); teacher.load_state_dict(ck.get("teacher", ck["model"]))
        try:
            opt.load_state_dict(ck["opt"])
        except (ValueError, KeyError) as e:
            print(f"  opt skip: {e}", flush=True)
        start = ck["step"]; print(f"RESUMED {start}", flush=True)

    # async DataLoader: parallel CPU token-assembly overlapped with GPU compute (was per-step
    # synchronous get_cached -> IO-bound). batch_size=None: each item is one full per-event token set.
    train_loader = DataLoader(D.CachedTPC(train_b), batch_size=None, shuffle=True, num_workers=8,
                              persistent_workers=True, prefetch_factor=4, pin_memory=True,
                              worker_init_fn=D._worker_init, multiprocessing_context="spawn")
    data_iter = iter(train_loader)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / a.warmup
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    def jepa_loss(B, mask, want_ctx=False):
        out = student.forward_feat(B, mask, return_ctx=want_ctx)     # (N,d) latents [+ visible-ctx encoder feats]
        pred, ctx = out if want_ctx else (out, None)
        with torch.no_grad():
            tgt = F.layer_norm(teacher.encode(B).float(), (a.d,))   # EMA-teacher full-context latents, normalized
        lp = F.smooth_l1_loss(pred[mask].float(), tgt[mask].detach())
        return (lp, ctx) if want_ctx else lp

    curve = open("jepa_curve.jsonl", "a"); t0 = time.time()
    for step in range(start + 1, a.steps + 1):
        for pg, r in zip(opt.param_groups, _ratio):
            pg["lr"] = lr_at(step) * r
        try:
            B = next(data_iter)
        except StopIteration:                                    # next epoch (loader reshuffles)
            data_iter = iter(train_loader); B = next(data_iter)
        B = move(B, DEV)
        mask = torch.rand(B["inp"].shape[0], device=DEV) < a.mask
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.sigreg > 0:
                lp, ctx = jepa_loss(B, mask, want_ctx=True)      # reuse visible-ctx encoder feats (already in graph)
                ls = sigreg(ctx.float())                          # NO redundant full re-encode (was the 2.6x slowdown)
            else:
                lp = jepa_loss(B, mask); ls = torch.zeros((), device=DEV)
            loss = lp + a.sigreg * ls
        opt.zero_grad(); loss.backward(); opt.step()
        mm = 1.0 - (1.0 - a.momentum) * 0.5 * (1 + math.cos(math.pi * min(step / a.steps, 1.0)))  # -> 1.0
        with torch.no_grad():
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.mul_(mm).add_(ps.detach(), alpha=1 - mm)
        if step % 500 == 0:
            print(f"step {step}: loss {float(loss):.4f} pred {float(lp):.4f} sig {float(ls):.4f} lr={lr_at(step):.1e} mom={mm:.4f} "
                  f"{(time.time()-t0)/(step-start)*1000:.0f}ms/step", flush=True)
        if a.eval_every and step % a.eval_every == 0:
            g = torch.Generator(device=DEV).manual_seed(11); vs = []
            with torch.no_grad():
                for pv in test_b[:40]:
                    Bv = D.get_cached(pv, device=DEV)
                    mv = torch.rand(Bv["inp"].shape[0], generator=g, device=DEV) < a.mask
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        vs.append(float(jepa_loss(Bv, mv)))
            vl = float(np.mean(vs))
            with torch.no_grad():                                # COLLAPSE monitor: per-dim std across tokens (~0 = collapsed)
                fz = teacher.encode(D.get_cached(test_b[0], device=DEV)).float()
                fstd = float(fz.std(0).mean()); frank = float((fz.std(0) > 0.01 * fz.std(0).max()).sum())
                rme = rankme(fz)
            print(f"  [eval {step}] val_loss {vl:.4f}  feat_std {fstd:.4f}  RankMe {rme:.1f}/{a.d}  eff_dims {frank:.0f} "
                  f"{'<<< COLLAPSE' if fstd < 0.02 else ''}", flush=True)
            curve.write(json.dumps(dict(tag=a.tag, step=step, val_loss=vl, feat_std=fstd, rankme=rme, eff_dims=frank)) + "\n"); curve.flush()
            sd = student.state_dict()
            torch.save(dict(model=sd, teacher=teacher.state_dict(), opt=opt.state_dict(), step=step, d=a.d), a.ckpt)
        if a.snap_every and step % a.snap_every == 0:        # independent of eval_every (alignment-bug fix)
            torch.save(dict(model=student.state_dict(), step=step, d=a.d), a.ckpt.replace(".pt", f"_snap{step}.pt"))
    open(a.ckpt + ".done", "w").write("done")


if __name__ == "__main__":
    main()
