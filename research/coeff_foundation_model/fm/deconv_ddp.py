"""DDP deconvolution trainer (2+ GPUs). Each rank: 1 event/step (+ optional grad-accum),
SDPA attention (fastest per the batch benchmark), gradients all-reduced across GPUs.
Throughput should scale ~linearly with #GPUs. Run:
  torchrun --nproc_per_node=2 deconv_ddp.py --d 768 --blocks 12 --dec_blocks 4 --steps 8000 --init mae
"""
import sys, os, time, argparse, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import data as D
from data import N_SLOT, N_BAND
from model import FMModel

VB = ["A4", "D4", "D3", "D2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=20000); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--d", type=int, default=768); ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--dec_blocks", type=int, default=4); ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--init", default="mae"); ap.add_argument("--mae_ckpt", default="ckpt_dual44M.pt")
    ap.add_argument("--ckpt", default="ckpt_deconv_ddp.pt"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval_every", type=int, default=2000); ap.add_argument("--test_frac", type=float, default=0.1)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl"); torch.cuda.set_device(local); dev = torch.device("cuda", local)
    torch.manual_seed(0); np.random.seed(0)
    D.init_pipeline_cpu()
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_cache_tpc"))
    qdir = os.path.abspath(os.path.join(here, "..", "artifacts", "fm_charge_tpc"))
    is0 = rank == 0
    def log(*a):
        if is0: print(*a, flush=True)

    model = FMModel(N_SLOT, N_BAND, 6, n_wirefeat=1, d=args.d, blocks=args.blocks, dec_blocks=args.dec_blocks).to(dev)
    if args.init == "mae" and not (args.resume and os.path.exists(os.path.join(here, args.ckpt))):
        model.load_state_dict(torch.load(os.path.join(here, args.mae_ckpt), map_location=dev)["model"])
    ddp = DDP(model, device_ids=[local], find_unused_parameters=True)   # occ_head unused in deconv
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ckpt_path = os.path.join(here, args.ckpt); start = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev); model.load_state_dict(ck["model"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        start = ck["step"]; log(f"RESUMED at {start}")
    log(f"DDP world={world} d={args.d} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
        f"accum={args.accum} (effective batch/step = {world*args.accum} events)")

    def lr_at(s):
        if args.warmup and s < args.warmup: return args.lr * s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    have = [i for i in range(args.events)
            if os.path.exists(f"{qdir}/ev_charge_{i:05d}.npz") and os.path.exists(f"{cdir}/ev_{i:05d}.npz")]
    nte = max(1, int(len(have) * args.test_frac)); test, train = have[:nte], have[nte:]
    shard = train[rank::world]                                # each GPU gets its own slice
    log(f"events: {len(have)} (train {len(train)} / test {len(test)}); shard/rank ~{len(shard)}")

    def load(i):
        B = D.get_cached_charge(f"{cdir}/ev_{i:05d}.npz", f"{qdir}/ev_charge_{i:05d}.npz")
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in B.items()}

    # per-band charge scale (rank 0 computes on a subset, broadcast)
    S = torch.zeros(N_BAND, device=dev)
    if is0:
        sq = np.zeros(N_BAND); n = np.zeros(N_BAND)
        for i in train[:150]:
            B = load(i); c = B["target_charge"].cpu().numpy(); bd = B["band_id"][B["cell"]].cpu().numpy()
            np.add.at(sq, bd, c**2); np.add.at(n, bd, 1)
        S = torch.tensor(np.sqrt(sq/np.maximum(n,1))+1e-6, dtype=torch.float32, device=dev)
    dist.broadcast(S, 0); log(f"S={S.cpu().numpy().round(1)}")

    def tgt(B): return torch.asinh(B["target_charge"] / S[B["band_id"][B["cell"]]])

    @torch.no_grad()
    def evalr2():
        model.eval(); bd_, e_, t_ = [], [], []
        for i in test[:80]:
            B = load(i); m = torch.zeros(int(B["n_cells"]), dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): _, mu, _ = model(B, m)
            pr = mu[B["cell"], B["slot"]].float(); t = tgt(B)
            bd_.append(B["band_id"][B["cell"]].cpu().numpy()); e_.append(((pr-t)**2).cpu().numpy()); t_.append(t.cpu().numpy())
        model.train()
        bd = np.concatenate(bd_); e = np.concatenate(e_); tt = np.concatenate(t_)
        return {b: 1 - e[bd==b].mean()/(tt[bd==b].var()+1e-9) for b in range(N_BAND)}

    t0 = time.time(); seen = 0
    for step in range(start + 1, args.steps + 1):
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for a in range(args.accum):
            i = shard[(step * args.accum + a) % len(shard)]; B = load(i); seen += 1
            ctx = ddp.no_sync() if a < args.accum - 1 else torch.enable_grad()
            with ctx:
                m = torch.zeros(int(B["n_cells"]), dtype=torch.bool, device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): _, mu, _ = ddp(B, m)
                (((mu[B["cell"], B["slot"]].float() - tgt(B)) ** 2).mean() / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if is0 and (step % 200 == 0 or step == start + 1):
            ev_s = seen * world / (time.time() - t0)
            log(f"  step {step}: {(time.time()-t0)/(step-start)*1000:.0f} ms/step  {ev_s:.1f} events/s (all GPUs)")
        if is0 and step % args.eval_every == 0:
            r2 = evalr2(); ov = np.mean([r2[b] for b in range(N_BAND)])
            log(f"   [eval {step}] charge R²: overall={ov*100:.1f}% | " + " ".join(f"{VB[b]}={r2[b]*100:.1f}%" for b in range(N_BAND)))
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, d=args.d,
                            blocks=args.blocks, dec_blocks=args.dec_blocks), ckpt_path)
        dist.barrier()
    log("done"); dist.destroy_process_group()


if __name__ == "__main__":
    main()
