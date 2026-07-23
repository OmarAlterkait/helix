"""Overfit-first trainer for the FM architecture.

Stage 1 (--overfit): memorize a tiny set (default 2 events) for many steps with a
FIXED mask — train loss should crater toward 0. This validates the model + loss +
backprop end-to-end. Then scale: more events, bigger model, fresh masks.

Run:
  python train.py --overfit                      # sanity: can it fit 2 events?
  python train.py --events 64 --steps 5000 --d 256 --blocks 8   # scale up
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import data as D
from model import FMModel, losses
from data import DEV, N_SLOT, N_BAND, N_PLANE

# measured predict-mean MSE per band (A4,D4,D3,D2) = target variance; var_explained ceiling
TGT_VAR_BAND = {0: 14.00, 1: 13.83, 2: 11.51, 3: 6.11}      # CLEAN-coeff variance
NOISY_VAR_BAND = {0: 15.06, 1: 14.23, 2: 11.99, 3: 7.17}    # NOISY-coeff variance (self-supervised target)


def move(B, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in B.items()}


def load_b(item):
    """item is either a cached npz path (lazy, RAM-safe) or an already-assembled batch."""
    return D.get_cached(item) if isinstance(item, str) else item


def make_mask(B, mode, ratio, n_planes, gen=None):
    """mode: random (frac of tokens) | plane (whole plane(s)) | block (wire-slab/plane)."""
    n = B["n_cells"]; dev = B["plane_id"].device
    rnd = (lambda *s: torch.rand(*s, generator=gen, device=dev)) if (gen is not None and gen.device.type == dev.type) \
        else (lambda *s: torch.rand(*s, device=dev))
    if mode == "random":
        return rnd(n) < ratio
    gid = B["plane_id"]
    if mode == "plane":                       # cross-plane: hide whole plane(s)
        gids = torch.unique(gid)
        perm = gids[torch.randperm(len(gids), device=dev)]
        pick = perm[:n_planes]
        return torch.isin(gid, pick)
    if mode == "block":                       # contiguous wire-slab per plane (~ratio wide)
        wp = B["wire_pos"]; m = torch.zeros(n, dtype=torch.bool, device=dev)
        for g in torch.unique(gid):
            sel = gid == g
            w = wp[sel]; lo, hi = float(w.min()), float(w.max()) + 1
            win = (hi - lo) * ratio
            start = lo + float(rnd(1)) * max(hi - lo - win, 0.0)
            m[sel] = (w >= start) & (w < start + win)
        return m
    raise ValueError(mode)


@torch.no_grad()
def perband_mse(model, batches, mode, ratio, n_planes, masks=None, noisy=False):
    """Per-band MSE split by MASKED (inference) vs VISIBLE (denoising) active slots.
    Returns (m_prim, m_perband, base, v_prim, v_perband)."""
    model.eval()
    seM = np.zeros(N_BAND); cntM = np.zeros(N_BAND); base = np.zeros(N_BAND)
    seV = np.zeros(N_BAND); cntV = np.zeros(N_BAND)
    for i, B in enumerate(batches):
        B = move(load_b(B), DEV)
        if masks is not None:
            m = masks[i]
        else:
            g = torch.Generator(device=DEV).manual_seed(7 + i)
            m = make_mask(B, mode, ratio, n_planes, g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, m)
        tgt_full = (B["inp"][B["cell"], B["slot"]] if noisy else B["target"])    # noisy-self-sup vs clean
        err = ((mu[B["cell"], B["slot"]] - tgt_full) ** 2).float().cpu().numpy()
        bl_ = ((B["inp"][B["cell"], B["slot"]] - B["target"]) ** 2).float().cpu().numpy()  # noise floor (vs clean)
        bd = B["band_id"][B["cell"]].cpu().numpy()
        mr = m[B["cell"]].cpu().numpy(); vr = ~mr
        np.add.at(seM, bd[mr], err[mr]); np.add.at(cntM, bd[mr], 1); np.add.at(base, bd[mr], bl_[mr])
        np.add.at(seV, bd[vr], err[vr]); np.add.at(cntV, bd[vr], 1)
    model.train()
    pbM = seM / np.maximum(cntM, 1); pbV = seV / np.maximum(cntV, 1)
    return (float(pbM.mean()), {f"b{b}": float(pbM[b]) for b in range(N_BAND)},
            float((base / np.maximum(cntM, 1)).mean()),
            float(pbV.mean()), {f"b{b}": float(pbV[b]) for b in range(N_BAND)})


@torch.no_grad()
def nll_eval(model, batches, mode, ratio, n_planes):
    """Mean Gaussian NLL (nats/coeff) on MASKED active slots — the LIKELIHOOD scaling metric
    (proper scoring rule; unlike var-explained it does NOT saturate). None if not an nll head."""
    model.eval(); tot = 0.0; cnt = 0; c2 = 0.5 * float(np.log(2 * np.pi))
    for i, B in enumerate(batches):
        B = move(load_b(B), DEV)
        g = torch.Generator(device=DEV).manual_seed(7 + i); m = make_mask(B, mode, ratio, n_planes, g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, m)
        if lv is None:
            model.train(); return None
        cm = m[B["cell"]]
        pred = mu[B["cell"], B["slot"]][cm].float(); tgt = B["target"][cm].float()
        logv = lv[B["cell"], B["slot"]][cm].float().clamp(-8, 8)
        nll = 0.5 * ((pred - tgt) ** 2 * torch.exp(-logv) + logv) + c2
        tot += float(nll.sum()); cnt += int(cm.sum())
    model.train(); return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--events", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4, help="attention heads (legacy; used only if --head_dim==0)")
    ap.add_argument("--head_dim", type=int, default=0, help="if >0, FIX head_dim and derive heads=d//head_dim (field-standard ViT + flash-optimal + muP-clean across width); else use --heads")
    ap.add_argument("--dec_blocks", type=int, default=2)
    ap.add_argument("--mask", type=float, default=0.5)
    ap.add_argument("--mask_mode", default="random", choices=["random","plane","block"])
    ap.add_argument("--n_planes", type=int, default=1)
    ap.add_argument("--film", default="band,plane,wire")
    ap.add_argument("--nll", action="store_true")
    ap.add_argument("--cap", type=int, default=30000)
    ap.add_argument("--cache_dir", default=None, help="cached sparse-coeff dir (scale run)")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--val_n", type=int, default=0, help="FIXED held-out val count (first val_n events); >0 overrides test_frac -> clean data-scaling (val constant, train = events-val_n)")
    ap.add_argument("--warmup", type=int, default=0, help="LR warmup steps (then cosine decay)")
    ap.add_argument("--eval_every", type=int, default=0, help="periodic eval cadence (learning curve)")
    ap.add_argument("--eval_n", type=int, default=80, help="# held-out events for periodic eval")
    ap.add_argument("--snap_every", type=int, default=0, help="also save KEPT step-tagged ckpt_TAG_snap<step>.pt every N steps (probe-trajectory)")
    ap.add_argument("--tag", default="", help="run tag for the curve log")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers for parallel CPU token-assembly (overlaps with GPU)")
    ap.add_argument("--vis_w", type=float, default=0.0,
                    help="weight on the VISIBLE-token denoising loss (0 = masked-only)")
    ap.add_argument("--noisy_target", action="store_true",
                    help="self-supervised: predict the NOISY input coeff (no clean truth); else predict CLEAN target")
    ap.add_argument("--ffn_mult", type=int, default=4, help="FFN expansion (cheap capacity lever)")
    ap.add_argument("--cond", default="film", choices=["film", "adaln"], help="conditioning: input-FiLM vs per-layer AdaLN-Zero")
    ap.add_argument("--dec_mode", default="self", choices=["self", "cross"], help="decoder: full-attn over all N (self) vs CrossMAE masked-x-attend-visible (cheaper)")
    ap.add_argument("--mup", action="store_true", help="Maximal Update Parametrization (LR transfers across width d); tune --lr at --d_base once, reuse at any --d")
    ap.add_argument("--wd", type=float, default=0.05, help="weight decay (field-standard SSL ~0.05; DECOUPLED from muP LR + no-decay on 1-D params)")
    ap.add_argument("--d_base", type=int, default=128, help="muP proxy width d_base (the width the LR was tuned at)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (fuses elementwise, ~1.3x; dynamic=True for variable N)")
    ap.add_argument("--ckpt", default="", help="checkpoint path (save every eval; auto from --tag if empty)")
    ap.add_argument("--resume", action="store_true", help="resume from --ckpt if it exists")
    ap.add_argument("--config", default=None, help="YAML config -> argparse defaults (CLI overrides); for SLURM driver")
    _pre, _ = ap.parse_known_args()
    if _pre.config:
        import yaml
        with open(_pre.config) as _f:
            _cfg = yaml.safe_load(_f) or {}
        for _k in ("script", "hours"):
            _cfg.pop(_k, None)
        _valid = {a.dest for a in ap._actions}                 # guard: a mistyped YAML key would
        _unknown = [k for k in _cfg if k.replace("-", "_") not in _valid]   # silently no-op via set_defaults
        if _unknown:
            raise SystemExit(f"config {_pre.config} has unknown key(s) (typo?): {_unknown}")
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in _cfg.items()})
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    # cached runs need only nw (geom) -> CPU init; avoids the GPU forward-pipeline build
    # (and its pimm_data import, which churns). Full init only for the on-the-fly path.
    D.init_pipeline_cpu() if args.cache_dir else D.init_pipeline()

    film = tuple(args.film.split(",")) if args.film else ()
    heads = (args.d // args.head_dim) if args.head_dim > 0 else args.heads   # fix head_dim -> derive heads (muP-clean across width)
    model = FMModel(N_SLOT, N_BAND, N_PLANE, n_wirefeat=1, d=args.d, blocks=args.blocks, heads=heads,
                    dec_blocks=args.dec_blocks, film=film, nll=args.nll, ffn_mult=args.ffn_mult,
                    cond=args.cond, dec_mode=args.dec_mode, mup=args.mup, d_base=args.d_base).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"FMModel d={args.d} enc={args.blocks} dec={args.dec_blocks} heads={heads} head_dim={args.d//heads} "
          f"cond={args.cond} nll={args.nll} params={npar/1e6:.2f}M | mode={'OVERFIT' if args.overfit else 'scale'} events={args.events}")
    # arch metadata saved in EVERY checkpoint -> reload can't silently mismatch heads/nll/mup/etc.
    arch = dict(d=args.d, blocks=args.blocks, dec_blocks=args.dec_blocks, heads=heads,
                nll=args.nll, cond=args.cond, dec_mode=args.dec_mode, mup=args.mup,
                d_base=args.d_base, ffn_mult=args.ffn_mult, film=args.film)
    # muP: per-category LR groups (hidden LR = lr/m; input/output/bias/LN = lr). With
    # mup=False, param_groups collapses to a single lr==args.lr group (identical behavior).
    # decoupled weight decay: per-group wd set inside param_groups so effective decay is
    # width-invariant under muP (do NOT also pass wd to the constructor -> would double-set).
    opt = torch.optim.AdamW(model.param_groups(args.lr, weight_decay=args.wd), lr=args.lr, betas=(0.9, 0.95))
    _lr_ratio = [pg["lr"] / args.lr for pg in opt.param_groups]   # per-group muP ratio (1 or 1/m); preserved by the scheduler
    ckpt_path = args.ckpt or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          f"ckpt_{args.tag or 'run'}.pt")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEV)
        model.load_state_dict(ck["model"]); start_step = ck["step"]
        try:                                                   # opt state may mismatch (e.g. param-group count changed by muP edit)
            opt.load_state_dict(ck["opt"])
        except (ValueError, KeyError) as e:
            print(f"  (opt state incompatible: {e} -> keeping model+step, FRESH optimizer)", flush=True)
        print(f"RESUMED from {ckpt_path} at step {start_step}", flush=True)
    if args.compile:                                           # compile AFTER load (params unchanged; keep _orig_mod for ckpt)
        model = torch.compile(model, dynamic=True)
        print("torch.compile ON (dynamic=True)", flush=True)
    # warmup (linear) -> cosine decay to 10% of peak over the run
    import math
    def lr_at(step):
        if args.warmup and step < args.warmup:
            return args.lr * step / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

    # load events: cached (scale) or on-the-fly (small). held-out split for scale.
    if args.cache_dir:
        import glob
        # NUMERIC sort: filenames aren't fixed-width (events run past 99999 -> ev_100000),
        # so plain sorted() is lexicographic and scrambles the train/val split (MAE vs JEPA
        # would then train on different subsets). Sort by the integer event id.
        files = sorted(glob.glob(os.path.join(args.cache_dir, "ev_*.npz")),
                       key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p)))))[:args.events]
        if args.overfit:
            print(f"loading {len(files)} cached events (overfit: GPU-resident)...", flush=True)
            allb = [move(D.get_cached(f), DEV) for f in files]    # tiny set -> resident on GPU
        elif args.workers > 0:
            print(f"{len(files)} cached events via {args.workers}-worker DataLoader "
                  f"(parallel CPU assembly, overlaps GPU)...", flush=True)
            allb = files                                          # paths; workers assemble in parallel
        elif len(files) <= 400:                                   # fits cgroup RAM (~60MB/ev assembled)
            print(f"staging {len(files)} cached events to CPU (fast path)...", flush=True)
            allb = [move(D.get_cached(f), "cpu") for f in files]  # assemble once, move to GPU per step
        else:
            print(f"lazy-loading {len(files)} cached events per step (RAM-safe; npz paths only)...", flush=True)
            allb = files                                          # paths; assembled on demand per step
    else:
        print("pre-extracting events on-the-fly...", flush=True)
        allb = [move(D.get_event(i, cap=args.cap), "cpu") for i in range(args.events)]
    if args.overfit:
        train_b, test_b = allb, []                            # already GPU-resident
    else:
        ntest = args.val_n if args.val_n > 0 else max(1, int(len(allb) * args.test_frac))  # fixed val -> clean data-scaling
        test_b, train_b = allb[:ntest], allb[ntest:]
    batches = train_b
    print(f"train={len(train_b)} test={len(test_b)} events", flush=True)
    # overfit: FIXED mask per batch, computed once on the GPU-resident batch
    masks = None
    if args.overfit:
        masks = [make_mask(B, args.mask_mode, args.mask, args.n_planes,
                           torch.Generator(device=DEV).manual_seed(1000 + i))
                 for i, B in enumerate(batches)]

    # parallel CPU token-assembly via DataLoader workers (overlaps with GPU compute).
    # batch_size=None: each item is already one full per-event token set (no collation).
    train_loader = data_iter = None
    if args.workers > 0 and not args.overfit:
        from torch.utils.data import DataLoader
        train_loader = DataLoader(D.CachedTPC(train_b), batch_size=None, shuffle=True,
                                  num_workers=args.workers, persistent_workers=True,
                                  prefetch_factor=4, pin_memory=True,    # MAE has mem headroom
                                  worker_init_fn=D._worker_init,
                                  multiprocessing_context="spawn")
        data_iter = iter(train_loader)

    # periodic-eval subsets: held-out VAL + a fixed TRAIN subset -> the train-vs-val GAP
    # (scaling diagnostic: train<<val = overfit/data-limited; train~=val = underfit/objective-or-capacity-limited)
    eval_b = test_b[:args.eval_n] if (test_b and args.eval_every) else []
    train_eval_b = train_b[:args.eval_n] if args.eval_every else []     # SAME masked metric on TRAINING data
    curve_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fm_curve.jsonl")
    VARB = NOISY_VAR_BAND if args.noisy_target else TGT_VAR_BAND
    def _ve(pbd):
        return float(np.mean([(VARB[b] - pbd[f"b{b}"]) / VARB[b] for b in range(N_BAND)]))
    def log_curve(step):
        prm, pbd, bse, pvm, pvb = perband_mse(model, eval_b, args.mask_mode, args.mask, args.n_planes, noisy=args.noisy_target)
        veM, veV = _ve(pbd), _ve(pvb)                      # VAL: inference ceiling vs denoising ceiling
        veT = None
        if train_eval_b:
            _, tpbd, _, _, _ = perband_mse(model, train_eval_b, args.mask_mode, args.mask, args.n_planes, noisy=args.noisy_target)
            veT = _ve(tpbd)                                # TRAIN var-explained (same masked metric) -> the GAP
        vnll = nll_eval(model, eval_b, args.mask_mode, args.mask, args.n_planes)          # likelihood metric (nll heads)
        tnll = nll_eval(model, train_eval_b, args.mask_mode, args.mask, args.n_planes) if train_eval_b else None
        gap = (f" | TRAIN={veT*100:.1f}% gap={(veT-veM)*100:+.1f}" if veT is not None else "")
        nlls = (f" | NLL val={vnll:.3f} train={tnll if tnll is None else round(tnll,3)}" if vnll is not None else "")
        print(f"    [eval @ {step}] masked: mse={prm:.3f} var_expl={veM*100:.1f}%{gap} | "
              f"visible(denoise): mse={pvm:.3f} var_expl={veV*100:.1f}% | xNoise={prm/max(bse,1e-9):.1f}{nlls}", flush=True)
        with open(curve_path, "a") as f:
            f.write(json.dumps(dict(tag=args.tag, d=args.d, blocks=args.blocks, events=args.events,
                    mask=args.mask, vis_w=args.vis_w, step=step, mse=prm, var_expl=veM, train_var_expl=veT,
                    val_nll=vnll, train_nll=tnll, vis_mse=pvm, vis_var_expl=veV, per_band=pbd, vis_per_band=pvb)) + "\n")
        return veM

    t0 = time.time(); order = list(range(len(batches)))
    for step in range(start_step + 1, args.steps + 1):
        for pg, r in zip(opt.param_groups, _lr_ratio): pg["lr"] = lr_at(step) * r   # keep muP per-group ratio
        if train_loader is not None:                                   # workers assemble in parallel
            try:
                B = next(data_iter)
            except StopIteration:                                      # next epoch (loader reshuffles)
                data_iter = iter(train_loader); B = next(data_iter)
            B = move(B, DEV)
        else:
            if not args.overfit and (step - 1) % len(batches) == 0:    # reshuffle each epoch
                order = np.random.permutation(len(batches)).tolist()
            bi = order[(step - 1) % len(batches)]
            B = batches[bi] if args.overfit else move(load_b(batches[bi]), DEV)
        m = masks[bi] if args.overfit else make_mask(B, args.mask_mode, args.mask, args.n_planes)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, m)
            bce, val = losses(occ, mu, lv, B, m, vis_w=args.vis_w, noisy=args.noisy_target)
        loss = bce + val
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  step {step:>5}: loss {float(loss):.4f} (bce {float(bce):.4f} val {float(val):.4f}) "
                  f"lr={lr_at(step):.1e} {(time.time()-t0)/(step-start_step)*1000:.0f} ms/step", flush=True)
        if args.eval_every and eval_b and (step % args.eval_every == 0):
            log_curve(step)
            _sd = (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()  # clean (uncompiled) keys
            torch.save(dict(model=_sd, opt=opt.state_dict(), step=step, **arch),
                       ckpt_path)                                   # resilience + weights for the probe
        if args.snap_every and step % args.snap_every == 0:        # KEPT step-tagged snapshot (independent of eval_every)
            _ss = (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()
            torch.save(dict(model=_ss, step=step, **arch),
                       ckpt_path.replace(".pt", f"_snap{step}.pt"))

    # report
    out = dict(mode="overfit" if args.overfit else "scale", events=args.events,
               steps=args.steps, d=args.d, blocks=args.blocks, dec_blocks=args.dec_blocks,
               lr=args.lr, warmup=args.warmup, params=npar,
               mask_mode=args.mask_mode, mask=args.mask, n_planes=args.n_planes)
    out["vis_w"] = args.vis_w
    tr_eval = batches if args.overfit else batches[:200]    # bound the final eval (lazy path is serial)
    prim, pb, base, vprim, vpb = perband_mse(model, tr_eval, args.mask_mode, args.mask, args.n_planes, masks=masks, noisy=args.noisy_target)
    out.update(train_primary_mse=prim, train_per_band=pb, classical_baseline=base,
               train_ratio_to_classical=prim / max(base, 1e-9),
               train_visible_mse=vprim, train_visible_per_band=vpb)
    if test_b:
        tprim, tpb, tbase, tvprim, tvpb = perband_mse(model, test_b[:200], args.mask_mode, args.mask, args.n_planes, noisy=args.noisy_target)
        out.update(test_primary_mse=tprim, test_per_band=tpb, test_classical_baseline=tbase,
                   test_ratio_to_classical=tprim / max(tbase, 1e-9),
                   generalization_gap=tprim - prim,
                   test_visible_mse=tvprim, test_visible_per_band=tvpb)
    print(json.dumps(out, indent=1))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fm_results.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")
    if not args.overfit:                                      # unconditional final weights (last step may fall between evals)
        _fsd = (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()
        torch.save(dict(model=_fsd, opt=opt.state_dict(), step=args.steps, **arch), ckpt_path)
    open(ckpt_path + ".done", "w").close(); print("done")    # marker so the SLURM babysitter stops resubmitting


if __name__ == "__main__":
    main()
