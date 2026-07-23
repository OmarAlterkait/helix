"""DDP MAE trainer (standardized recipe) — event-sharded data-parallel across GPUs.

Each rank processes 1 event/step; gradients all-reduced => global batch = world events
(the systems-agent lever for the "1 event ~= 1 independent sample" problem). Same
FMModel + losses + muP + no-decay + numeric-sort + arch-metadata + probe snapshots as
the single-GPU train.py; adds --nll and --seed for the objective x seed matrix.

Launch (via slurm/train.sh with `gpus: N` in the config):
  torchrun --standalone --nproc_per_node=N mae_ddp.py --config configs/mae_mse_s0.yaml --resume
"""
import sys, os, json, time, argparse, math, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import data as D
from data import N_SLOT, N_BAND, N_PLANE
from model import FMModel, losses_fused, losses
from train import make_mask, perband_mse, nll_eval, TGT_VAR_BAND


def move(B, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in B.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    _pre, _ = ap.parse_known_args()
    cfg = {}
    if _pre.config:
        import yaml
        cfg = yaml.safe_load(open(_pre.config)) or {}
        for _k in ("script", "hours", "gpus"):
            cfg.pop(_k, None)
    ap.add_argument("--cache_dir", default="../artifacts/fm_cache_tpc")
    ap.add_argument("--events", type=int, default=21000); ap.add_argument("--val_n", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--dec_blocks", type=int, default=4); ap.add_argument("--dec_mode", default="cross")
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--head_dim", type=int, default=0)
    ap.add_argument("--ffn_mult", type=int, default=4); ap.add_argument("--cond", default="film")
    ap.add_argument("--film", default="band,plane,wire")
    ap.add_argument("--mask", type=float, default=0.75); ap.add_argument("--mask_mode", default="random")
    ap.add_argument("--n_planes", type=int, default=1)
    ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--mup", type=int, default=1); ap.add_argument("--d_base", type=int, default=128)
    ap.add_argument("--nll", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wire_rope", type=int, default=1)   # 0 = time-only RoPE (audit bug-2 fix)
    ap.add_argument("--compile", type=int, default=0)     # torch.compile the model
    ap.add_argument("--fused", type=int, default=1)       # 1=losses_fused (dense) 0=losses (gather) [ablation]
    ap.add_argument("--cellt", default="canonical")       # canonical | centroid (debiased survivor-max fine time)
    ap.add_argument("--plane_frac", type=float, default=0.0)  # fraction of steps using whole-plane masking (force triangulation)
    ap.add_argument("--serial", type=int, default=0)      # grouped 3-order serial encoder+decoder (PTv3-style)
    ap.add_argument("--rope_split", type=int, default=1)  # serial: axial within-plane, time-only cross-plane
    ap.add_argument("--gp", type=int, default=1024); ap.add_argument("--gd", type=int, default=2048)
    ap.add_argument("--init_from", default="")            # warm-start model weights from this ckpt (step stays 0)
    ap.add_argument("--holdout_lo", type=int, default=0); ap.add_argument("--holdout_hi", type=int, default=0)  # exclude probe-label ids from train
    ap.add_argument("--lr_mode", default="cosine")        # cosine | const (WSD stable) | decay (WSD cooldown)
    ap.add_argument("--ema", type=float, default=0.0)     # EMA decay (e.g. 0.9999 ~ 7k-step half-life); 0 = off
    ap.add_argument("--pw", type=int, default=16); ap.add_argument("--pt", type=int, default=8)  # patch (wires x band-ticks)
    ap.add_argument("--eval_every", type=int, default=6000); ap.add_argument("--eval_n", type=int, default=60)
    ap.add_argument("--snap_every", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt", default="ckpt_mae_ddp.pt"); ap.add_argument("--tag", default="mae_ddp")
    ap.add_argument("--resume", action="store_true")
    if cfg:
        valid = {a.dest for a in ap._actions}
        unknown = [k for k in cfg if k.replace("-", "_") not in valid]
        if unknown:
            raise SystemExit(f"config {_pre.config} has unknown key(s): {unknown}")
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    args = ap.parse_args()

    local = int(os.environ["LOCAL_RANK"]); rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local); dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    torch.manual_seed(args.seed); np.random.seed(args.seed * 1000 + rank)   # DDP broadcasts init; data order per-rank
    D.init_pipeline_cpu()
    # patch size: set the vit_tpc globals (main proc: for eval assembly + model n_slot) AND export
    # to env so the SPAWNed DataLoader workers set the same (spawn re-imports -> defaults otherwise).
    import vit_tpc as _vtp
    _vtp.PW, _vtp.PT, _vtp.N_SLOT = args.pw, args.pt, args.pw * args.pt
    D.N_SLOT = args.pw * args.pt
    os.environ["FM_PW"], os.environ["FM_PT"] = str(args.pw), str(args.pt)
    if args.cellt != "canonical": os.environ["FM_CELLT"] = args.cellt   # tokenizer RoPE-time coord (workers inherit)
    n_slot = args.pw * args.pt
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.abspath(os.path.join(here, args.cache_dir))

    film = tuple(args.film.split(",")) if args.film else ()
    heads = (args.d // args.head_dim) if args.head_dim > 0 else args.heads
    Mcls, xkw = FMModel, {}
    if args.serial:
        from model_serial import SerialFMModel
        Mcls, xkw = SerialFMModel, dict(rope_split=bool(args.rope_split), gp=args.gp, gd=args.gd)
    model = Mcls(n_slot, N_BAND, N_PLANE, n_wirefeat=1, d=args.d, blocks=args.blocks,
                    dec_blocks=args.dec_blocks, heads=heads, film=film, nll=bool(args.nll),
                    ffn_mult=args.ffn_mult, cond=args.cond, dec_mode=args.dec_mode,
                    mup=bool(args.mup), d_base=args.d_base, wire_rope=bool(args.wire_rope), **xkw).to(dev)
    ddp = DDP(torch.compile(model) if args.compile else model, device_ids=[local])   # eval/save use uncompiled `model`
    opt = torch.optim.AdamW(model.param_groups(args.lr, weight_decay=args.wd), lr=args.lr, betas=(0.9, 0.95))
    _ratio = [pg["lr"] / args.lr for pg in opt.param_groups]              # preserve muP per-group ratio under the scheduler
    arch = dict(d=args.d, blocks=args.blocks, dec_blocks=args.dec_blocks, heads=heads,
                nll=bool(args.nll), cond=args.cond, dec_mode=args.dec_mode, mup=bool(args.mup),
                d_base=args.d_base, ffn_mult=args.ffn_mult, film=args.film, wire_rope=bool(args.wire_rope),
                pw=args.pw, pt=args.pt, n_slot=n_slot)

    _eid = lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))))
    files = sorted(glob.glob(os.path.join(cdir, "ev_*.npz")), key=_eid)[:args.events]
    if args.holdout_hi > args.holdout_lo:                                 # drop probe-label ids from train (anti-leakage)
        n0 = len(files); files = [p for p in files if not (args.holdout_lo <= _eid(p) <= args.holdout_hi)]
        if rank == 0: print(f"holdout ids [{args.holdout_lo},{args.holdout_hi}]: dropped {n0-len(files)} events", flush=True)
    test_files, train_files = files[:args.val_n], files[args.val_n:]
    my_train = train_files[rank::world]                                  # event-shard per rank
    ckpt_path = os.path.join(here, args.ckpt); start = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"])
        try: opt.load_state_dict(ck["opt"])
        except (ValueError, KeyError) as e:
            if rank == 0: print(f"  opt skip: {e}", flush=True)
        start = ck["step"]
        if rank == 0: print(f"RESUMED at step {start}", flush=True)
    elif args.init_from:
        ck = torch.load(os.path.join(here, args.init_from), map_location=dev)
        miss, unexp = model.load_state_dict(ck["model"], strict=False)
        if rank == 0: print(f"WARM-START from {args.init_from}: missing={len(miss)} unexpected={len(unexp)}", flush=True)

    # EMA of weights (rank 0 only; DDP keeps params identical across ranks). Saved in ckpts/snaps
    # so probes can evaluate the "virtually annealed" model mid-flat-phase.
    ema_sd = None
    if args.ema > 0 and rank == 0:
        ck0 = locals().get("ck")
        if args.resume and ck0 is not None and "ema" in ck0:
            ema_sd = {k: v.to(dev).float() for k, v in ck0["ema"].items()}
            print("EMA resumed from ckpt", flush=True)
        else:
            ema_sd = {k: v.detach().float().clone() for k, v in model.state_dict().items()}
            print(f"EMA init (decay={args.ema})", flush=True)

    loader = DataLoader(D.CachedTPC(my_train), batch_size=None, shuffle=True, num_workers=args.workers,
                        persistent_workers=True, prefetch_factor=4, pin_memory=True,
                        worker_init_fn=D._worker_init, multiprocessing_context="spawn")
    it = iter(loader)

    def lr_at(s):
        if s < args.warmup: return args.lr * s / args.warmup
        if args.lr_mode == "const":                                    # WSD stable phase: flat, no horizon baked in
            return args.lr
        if args.lr_mode == "decay":                                    # WSD cooldown: 1-sqrt to ~0 over `steps`
            p = (s - args.warmup) / max(1, args.steps - args.warmup)
            return args.lr * max(1e-3, 1.0 - math.sqrt(min(p, 1.0)))
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    def _ve(pbd): return float(np.mean([(TGT_VAR_BAND[b] - pbd[f"b{b}"]) / TGT_VAR_BAND[b] for b in range(N_BAND)]))

    if rank == 0:
        npar = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"DDP MAE d={args.d} enc={args.blocks} dec={args.dec_blocks} heads={heads} head_dim={args.d//heads} "
              f"nll={bool(args.nll)} world={world} seed={args.seed} | train={len(train_files)} shard={len(my_train)} "
              f"| {args.steps} steps (global batch={world} ev) | wd={args.wd} lr={args.lr} | params={npar:.1f}M", flush=True)
    eval_b = [move(D.get_cached(f), dev) for f in test_files[:args.eval_n]] if rank == 0 else []

    t0 = time.time(); _clip_n = _clip_d = 0
    for step in range(start + 1, args.steps + 1):
        for pg, r in zip(opt.param_groups, _ratio): pg["lr"] = lr_at(step) * r
        try: B = next(it)
        except StopIteration: it = iter(loader); B = next(it)
        B = move(B, dev)
        _mode = "plane" if (args.plane_frac > 0 and np.random.rand() < args.plane_frac) else args.mask_mode
        m = make_mask(B, _mode, args.mask, args.n_planes)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = ddp(B, m)
            _lossfn = losses_fused if args.fused else losses
            bce, val = _lossfn(occ, mu, lv, B, m, vis_w=0.0, noisy=False)
        loss = bce + val
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ema_sd is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point: ema_sd[k].mul_(args.ema).add_(v.float(), alpha=1 - args.ema)
                    else: ema_sd[k] = v.clone()
        if rank == 0:
            _clip_n += int(gn > 1.0); _clip_d += 1                    # clip-activation fraction (C8 diagnostic)
        if rank == 0 and (step % 200 == 0 or step == 1):
            print(f"  step {step:>6}: loss {float(loss):.4f} (bce {float(bce):.4f} val {float(val):.4f}) "
                  f"lr={lr_at(step):.1e} gn={float(gn):.2f} clip={_clip_n/max(_clip_d,1):.2f} "
                  f"{(time.time()-t0)/(step-start)*1000:.0f} ms/step", flush=True)
            _clip_n = _clip_d = 0
        if rank == 0 and args.eval_every and step % args.eval_every == 0:
            prm, pbd, bse, pvm, pvb = perband_mse(model, eval_b, args.mask_mode, args.mask, args.n_planes)
            vnll = nll_eval(model, eval_b, args.mask_mode, args.mask, args.n_planes)
            pl_ve = None
            if args.plane_frac > 0:                                  # triangulation-reconstruction trajectory
                _, ppbd, _, _, _ = perband_mse(model, eval_b, "plane", args.mask, args.n_planes)
                pl_ve = _ve(ppbd)
            print(f"   [eval {step}] masked var_expl={_ve(pbd)*100:.1f}% mse={prm:.3f}"
                  f"{'' if vnll is None else f' nll={vnll:.3f}'}"
                  f"{'' if pl_ve is None else f' | PLANE var_expl={pl_ve*100:.1f}%'}", flush=True)
            with open(os.path.join(here, "fm_curve.jsonl"), "a") as f:
                f.write(json.dumps(dict(tag=args.tag, d=args.d, step=step, var_expl=_ve(pbd),
                                        mse=prm, val_nll=vnll, plane_var_expl=pl_ve, per_band=pbd)) + "\n")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, **({'ema': ema_sd} if ema_sd is not None else {}), **arch), ckpt_path)
        if rank == 0 and args.snap_every and step % args.snap_every == 0:      # KEPT probe-trajectory snapshot
            torch.save(dict(model=model.state_dict(), step=step, **({'ema': ema_sd} if ema_sd is not None else {}), **arch),
                       ckpt_path.replace(".pt", f"_snap{step}.pt"))
    if rank == 0:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=args.steps, **({'ema': ema_sd} if ema_sd is not None else {}), **arch), ckpt_path)
        open(ckpt_path + ".done", "w").close(); print("done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
