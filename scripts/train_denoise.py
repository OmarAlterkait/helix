#!/usr/bin/env python3
"""Train the supervised denoiser (helix.model.denoise) -- from scratch or fine-tuned.

    torchrun --nnodes N --nproc_per_node 4 ... scripts/train_denoise.py \\
        --arch-from <artifact> [--init <artifact>] --truth-root <dir> --out <dir> \\
        [--n-events N] [--steps S] [--lr 1e-3] [--lr-head 1e-3]

One event per GPU. ``--arch-from`` names an FM eval artifact whose architecture
the encoder takes (its weights are NOT loaded unless ``--init`` names one too):
``--init`` is the label-efficiency arm, its absence the from-scratch arm, and the
two are otherwise identical. Validation is MSE on fixed cells of fixed ``val``
events, every ``--val-every`` steps; ``best.pt`` keeps the lowest. Training
resumes from ``<out>/last.pt`` when it exists.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arch-from", required=True, help="FM eval artifact whose arch the encoder uses")
    ap.add_argument("--init", default=None, help="FM eval artifact to load encoder weights from")
    ap.add_argument("--corpus-root", default=None, help="default: parent of HELIX_CORPUS")
    ap.add_argument("--runs", nargs="*", default=None, help="default: every run under the corpus root")
    ap.add_argument("--truth-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-events", type=int, default=None, help="train on a fixed subset of this size")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3, help="encoder base LR (muP-scaled by width)")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-events", type=int, default=64)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--win-per-event", type=int, default=0,
                    help="charge-free noise windows per event, every cell of each (the floor's bg windows)")
    ap.add_argument("--cov-per-sig", type=float, default=2.0,
                    help="empty cells drawn uniformly inside token footprints, per charge cell")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="FM arch overrides")
    return ap.parse_args()


def to_device(item, dev):
    import torch
    B = {k: torch.as_tensor(v).to(dev, non_blocking=True) if isinstance(v, np.ndarray) else v
         for k, v in item["B"].items()}
    B["n_cells"] = B["plane_id"].shape[0]
    return (B, torch.as_tensor(item["idx"]).to(dev), torch.as_tensor(item["aux"]).to(dev),
            torch.as_tensor(item["y"]).to(dev))


def main():
    a = parse()
    import ast
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from helix.data.denoise import DenoiseEvents, split_events
    from helix.model.artifact import load
    from helix.model.denoise import build_denoise
    from helix.paths import root

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lrank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lrank); dev = torch.device("cuda", lrank)
    say = (lambda *m: print(*m, flush=True)) if rank == 0 else (lambda *m: None)

    corpus_root = a.corpus_root or str(root("HELIX_CORPUS").parent)
    runs = a.runs or sorted(r for r in os.listdir(corpus_root) if r.startswith("run_"))
    overrides = dict(varlen=True, fused_qk=True, compile_blocks=True)
    for kv in a.override:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            overrides[k] = v
    arch = load(a.arch_from).arch
    sd = load(a.init).state_dict if a.init else None
    torch.manual_seed(a.seed)
    model = build_denoise(arch, sd, overrides=overrides).to(dev)
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    say(f"[denoise] arch {a.arch_from} init {a.init or 'scratch'} trainable {n_train_p/1e6:.1f}M world {world}")

    train = DenoiseEvents(corpus_root, runs, a.truth_root, "train", n_events=a.n_events, subset_seed=a.subset_seed,
                          cov_per_sig=a.cov_per_sig, win_per_event=a.win_per_event)
    val_items = split_events(corpus_root, runs[:1], "val")[:a.val_events]
    val = DenoiseEvents(corpus_root, runs, a.truth_root, "val", items=val_items, sample_seed=1234,
                        cov_per_sig=a.cov_per_sig, win_per_event=a.win_per_event)
    say(f"[denoise] train events {len(train)}  val events {len(val)}  runs {len(runs)}")

    ddp = DDP(model, device_ids=[lrank], find_unused_parameters=False)
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.lr_head, a.wd), betas=(0.9, 0.95))
    base = [g["lr"] for g in opt.param_groups]

    def lr_at(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    os.makedirs(a.out, exist_ok=True)
    step, best = 0, float("inf")
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck.get("best", best)
        say(f"[denoise] resumed at step {step}")
    meta = dict(arch=arch, overrides=overrides, args=vars(a), q0=train.q0, runs=runs)

    def save(path, extra=None):
        if rank:
            return
        tmp = path + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best=best, meta=meta,
                        **(extra or {})), tmp)
        os.replace(tmp, path)

    @torch.no_grad()
    def validate():
        model.eval()
        tot = torch.zeros(4, device=dev, dtype=torch.float64)      # sse, n, sse_pos, n_pos
        for i in range(rank, len(val), world):
            B, idx, aux, y = to_device(val[i], dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(B, idx, aux).float()
            e = (p - y) ** 2; pos = y > 0
            tot += torch.stack([e.sum(), torch.tensor(float(len(y)), device=dev),
                                e[pos].sum(), pos.sum().double()]).double()
        dist.all_reduce(tot)
        model.train()
        return (tot[0] / tot[1]).item(), (tot[2] / tot[3].clamp(min=1)).item(), ((tot[0] - tot[2]) / (tot[1] - tot[3]).clamp(min=1)).item()

    class Shard(torch.utils.data.IterableDataset):
        def __iter__(self):
            wi = torch.utils.data.get_worker_info()
            nw, wid = (wi.num_workers, wi.id) if wi else (1, 0)
            # Each rank takes a stride of a fresh permutation per pass; each worker
            # cycles through its rank's share from its own offset, so no worker is
            # ever empty (a small labeled set can hold fewer events than workers).
            ep, k = 0, wid
            while True:
                share = np.random.default_rng((a.seed, ep)).permutation(len(train))[rank::world]
                if not len(share):
                    share = np.array([rank % len(train)])
                while k < len(share):
                    yield train[int(share[k])]
                    k += nw
                k -= len(share); ep += 1

    loader = torch.utils.data.DataLoader(Shard(), batch_size=None, num_workers=a.workers,
                                         persistent_workers=True, prefetch_factor=4)
    it = iter(loader)
    model.train()
    t0, acc, n_acc = time.time(), 0.0, 0
    while step < a.steps:
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * lr_at(step)
        B, idx, aux, y = to_device(next(it), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = ddp(B, idx, aux).float()
        loss = torch.nn.functional.mse_loss(p, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        step += 1; acc += loss.item(); n_acc += 1
        if step % 50 == 0:
            say(f"step {step}/{a.steps} loss {acc / n_acc:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                f"{(time.time() - t0) / n_acc:.3f}s/step")
            t0, acc, n_acc = time.time(), 0.0, 0
        if step % a.val_every == 0 or step == a.steps:
            v, vp, vn = validate()
            say(f"[val] step {step} mse {v:.5f} (charge cells {vp:.5f}, empty cells {vn:.5f})")
            if v < best:
                best = v; save(os.path.join(a.out, "best.pt"))
            if rank == 0:
                with open(os.path.join(a.out, "val.jsonl"), "a") as fh:
                    fh.write(json.dumps(dict(step=step, mse=v, mse_charge=vp, mse_empty=vn)) + "\n")
        if step % a.ckpt_every == 0 or step == a.steps:
            save(last)
    save(os.path.join(a.out, "final.pt"))
    say(f"[denoise] done: best val mse {best:.5f}")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
