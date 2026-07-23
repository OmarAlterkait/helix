"""Batched direct-supervised deconvolution with the Perceiver + flow head.
Batches B events/step (deep stack batched over (B,M,d); cross-in/decode looped),
MEAN loss over events (LR batch-independent), worker DataLoader (not data-bound),
and the CORRECTED var-ratio metric (per-row predictive variance / target variance).

Run: torchless single-GPU. python deconv_perc_batch.py --events 20000 --batch 16 --depth 24 --steps 120000
"""
import sys, os, math, time, argparse, random, numpy as np, torch
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from data import DEV, N_SLOT, N_BAND
from perceiver_model import PerceiverDeconv
from flow_head import FlowHead, flow_loss, flow_sample
VB = ["A4", "D4", "D3", "D2"]


def move(B, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in B.items()}


def _collate(x):          # keep the list of variable-N event dicts (no stacking); picklable for spawn
    return x


class CachedCharge(torch.utils.data.Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return D.get_cached_charge(*self.items[i], device="cpu")


@torch.no_grad()
def band_scales(items):
    sq = np.zeros(N_BAND); n = np.zeros(N_BAND)
    for it in items[:200]:
        B = D.get_cached_charge(*it, device="cpu")
        c = B["target_charge"].numpy(); bd = B["band_id"][B["cell"]].numpy()
        np.add.at(sq, bd, c ** 2); np.add.at(n, bd, 1)
    return np.sqrt(sq / np.maximum(n, 1)) + 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=20000); ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=24); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--flow_steps", type=int, default=4); ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=4000); ap.add_argument("--ckpt", default="ckpt_pb.pt")
    ap.add_argument("--no_ckpt", action="store_true", help="disable gradient checkpointing (faster at small batch, more memory)")
    ap.add_argument("--resume", action="store_true", help="resume from --ckpt if it exists (model+head+opt+step+RNG); start fresh otherwise")
    ap.add_argument("--config", default=None, help="YAML config: its keys become argparse defaults; explicit CLI flags still override (mirrors JAXTPC --production-config)")
    _pre, _ = ap.parse_known_args()
    if _pre.config:
        import yaml
        with open(_pre.config) as _f:
            _cfg = yaml.safe_load(_f) or {}
        for _k in ("script", "hours"):                # SLURM-driver keys, not argparse
            _cfg.pop(_k, None)
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in _cfg.items()})
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); D.init_pipeline_cpu()
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_cache_tpc"))
    qdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_charge_tpc"))

    model = PerceiverDeconv(N_SLOT, N_BAND, 6, d=args.d, M=args.M, depth=args.depth, heads=args.heads).to(DEV)
    head = FlowHead(args.d, N_SLOT).to(DEV)
    npar = (sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())) / 1e6
    print(f"Perceiver-batch d={args.d} M={args.M} depth={args.depth} batch={args.batch} workers={args.workers} params={npar:.1f}M", flush=True)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)

    def lr_at(s):
        if s < args.warmup: return args.lr * s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    have = [i for i in range(args.events)
            if os.path.exists(f"{qdir}/ev_charge_{i:05d}.npz") and os.path.exists(f"{cdir}/ev_{i:05d}.npz")]
    paths = [(f"{cdir}/ev_{i:05d}.npz", f"{qdir}/ev_charge_{i:05d}.npz") for i in have]
    nte = max(args.batch, int(len(paths) * 0.1)); test_b, train_b = paths[:nte], paths[nte:]
    S = torch.tensor(band_scales(train_b), dtype=torch.float32, device=DEV)
    print(f"train={len(train_b)} test={len(test_b)} S={S.cpu().numpy().round(1)}", flush=True)

    from torch.utils.data import DataLoader
    loader = DataLoader(CachedCharge(train_b), batch_size=args.batch, shuffle=True, collate_fn=_collate,
                        num_workers=args.workers, persistent_workers=True, prefetch_factor=4,
                        worker_init_fn=D._worker_init, multiprocessing_context="spawn")
    it = iter(loader)

    def dense_target(B):
        n = int(B["n_cells"])
        y1 = torch.zeros(n, N_SLOT, device=DEV); sm = torch.zeros(n, N_SLOT, dtype=torch.bool, device=DEV)
        tgt = torch.asinh(B["target_charge"] / S[B["band_id"][B["cell"]]])
        y1[B["cell"], B["slot"]] = tgt; sm[B["cell"], B["slot"]] = True
        return y1, sm

    @torch.no_grad()
    def evaluate(items):
        model.eval(); head.eval()
        P = [[] for _ in range(N_BAND)]; T = [[] for _ in range(N_BAND)]
        for it_ in items:
            B = move(D.get_cached_charge(*it_, device="cpu"), DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.forward_feat(B).float()
            sv = flow_sample(head, z, N_SLOT, steps=args.flow_steps, k=args.samples)
            s_act = sv[:, B["cell"], B["slot"]].cpu().numpy()
            t = torch.asinh(B["target_charge"] / S[B["band_id"][B["cell"]]]).cpu().numpy()
            bd = B["band_id"][B["cell"]].cpu().numpy()
            for b in range(N_BAND):
                mm = bd == b; P[b].append(s_act[:, mm]); T[b].append(t[mm])
        model.train(); head.train()
        out = {}
        for b in range(N_BAND):
            s = np.concatenate(P[b], axis=1); t = np.concatenate(T[b])
            mean = s.mean(0); r2 = 1 - ((mean - t) ** 2).mean() / (t.var() + 1e-9)
            vr = float(s.var(axis=0).mean() / (t.var() + 1e-9))      # FIX 6d: per-row predictive var / target var
            out[b] = dict(r2=r2, vr=vr)
        return out

    ckpt_path = os.path.join(here, args.ckpt)

    def save_ckpt(step):                                   # atomic full-state checkpoint (resume-able)
        tmp = ckpt_path + ".tmp"
        torch.save(dict(model=model.state_dict(), head=head.state_dict(), opt=opt.state_dict(),
                        step=step, steps=args.steps, d=args.d, M=args.M, depth=args.depth,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                        np_rng=np.random.get_state(), py_rng=random.getstate()), tmp)
        os.replace(tmp, ckpt_path)                         # rename is atomic -> never a half-written ckpt

    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEV)
        if "opt" not in ck:                                   # stale/old-format checkpoint -> can't resume cleanly
            print(f"  (ckpt {args.ckpt} has no optimizer state [old format] -> starting FRESH)", flush=True)
        else:
            try:
                model.load_state_dict(ck["model"]); head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"])
                start_step = int(ck["step"])
                try:
                    tr = ck["torch_rng"]
                    torch.set_rng_state(tr.cpu().to(torch.uint8) if torch.is_tensor(tr) else tr)
                    torch.cuda.set_rng_state_all(ck["cuda_rng"]); np.random.set_state(ck["np_rng"]); random.setstate(ck["py_rng"])
                except Exception as _e:
                    print(f"  (RNG restore skipped: {_e})", flush=True)
                print(f"RESUMED from step {start_step}/{args.steps} (ckpt {args.ckpt})", flush=True)
                if start_step >= args.steps:
                    print("already complete; nothing to do."); return
            except Exception as _e:                            # architecture/shape mismatch etc.
                print(f"  (resume failed: {_e} -> starting FRESH)", flush=True); start_step = 0       # self-terminate: ends the job

    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        try: Bs = next(it)
        except StopIteration: it = iter(loader); Bs = next(it)
        Bs = [move(B, DEV) for B in Bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zs = model.forward_batch(Bs, ckpt=not args.no_ckpt)
        losses = [flow_loss(head, z.float(), *dense_target(B)) for z, B in zip(zs, Bs)]
        loss = torch.stack(losses).mean()                            # FIX 5a: MEAN over events
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0); opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  step {step}: floss {float(loss):.4f} lr={lr_at(step):.1e} {(time.time()-t0)/step*1000:.0f} ms/step", flush=True)
        if step % args.eval_every == 0:
            M = evaluate(test_b[:60]); ov_r2 = np.mean([M[b]["r2"] for b in range(N_BAND)]); ov_vr = np.mean([M[b]["vr"] for b in range(N_BAND)])
            print(f"   [eval {step}] R2 ov={ov_r2*100:.1f}% varRatio={ov_vr:.3f} | "
                  + " ".join(f"{VB[b]}(r2={M[b]['r2']*100:.0f},vr={M[b]['vr']:.2f})" for b in range(N_BAND)), flush=True)
            save_ckpt(step)
    save_ckpt(args.steps)
    print("done")


if __name__ == "__main__":
    main()
