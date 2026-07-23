"""WHERE does the per-step time go (the 'slowdown'), and is it loader-bound or
compute-bound? Breaks the real training step (with the actual DataLoader+workers)
into: next() wait / H2D / mask / fwd+loss / bwd+clip+opt. Run at 8 vs 16 workers
(does the loader keep up?) and simulate grad-accum 'batch' (amortize opt/python)."""
import glob, time, collections, torch
import data as D
from model import FMModel, losses
from torch.utils.data import DataLoader
dev = "cuda"


def move(B):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in B.items()}


def mk(B, r=0.75):
    return torch.rand(B["inp"].shape[0], device=B["inp"].device) < r


def step(model, opt, B, m, do_opt=True):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        occ, mu, lv = model(B, m); bce, val = losses(occ, mu, lv, B, m, noisy=True); loss = bce + val
    loss.backward()
    if do_opt:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    return loss


def main():
    D.init_pipeline_cpu()
    paths = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[:3000]
    model = FMModel(128, D.N_BAND, 6, d=512, blocks=12, dec_blocks=4, heads=8, dec_mode="cross").to(dev)
    opt = torch.optim.AdamW(model.parameters(), 4e-4)
    print(f"GPU {torch.cuda.get_device_name()} | profiling CrossMAE-decoder FMModel\n")

    for nw in (8, 16):
        loader = DataLoader(D.CachedTPC(paths), batch_size=None, shuffle=True, num_workers=nw,
                            persistent_workers=True, prefetch_factor=4, pin_memory=True,
                            worker_init_fn=D._worker_init, multiprocessing_context="spawn")
        it = iter(loader)
        for _ in range(8):
            B = move(next(it)); step(model, opt, B, mk(B))
        torch.cuda.synchronize()
        T = collections.defaultdict(float); ITERS = 50

        t0 = time.perf_counter()
        for _ in range(ITERS):
            B = next(it); B = move(B); step(model, opt, B, mk(B))
        torch.cuda.synchronize()
        real = (time.perf_counter() - t0) / ITERS * 1000

        for _ in range(ITERS):
            t = time.perf_counter(); B = next(it); T["next_wait"] += time.perf_counter() - t
            t = time.perf_counter(); B = move(B); torch.cuda.synchronize(); T["h2d"] += time.perf_counter() - t
            t = time.perf_counter(); m = mk(B); torch.cuda.synchronize(); T["mask"] += time.perf_counter() - t
            t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                occ, mu, lv = model(B, m); bce, val = losses(occ, mu, lv, B, m, noisy=True); loss = bce + val
            torch.cuda.synchronize(); T["fwd+loss"] += time.perf_counter() - t
            t = time.perf_counter(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); T["bwd+opt"] += time.perf_counter() - t
        br = " ".join(f"{k}={T[k]/ITERS*1000:5.1f}" for k in ("next_wait", "h2d", "mask", "fwd+loss", "bwd+opt"))
        print(f"workers={nw:2d}: REAL={real:6.1f} ms/step | breakdown(ms, serialized): {br}")
        del loader, it

    print("\ngrad-accum (one opt.step per K events; isolates per-step overhead amortization):")
    loader = DataLoader(D.CachedTPC(paths), batch_size=None, shuffle=True, num_workers=16,
                        persistent_workers=True, prefetch_factor=4, pin_memory=True,
                        worker_init_fn=D._worker_init, multiprocessing_context="spawn")
    it = iter(loader)
    for _ in range(8):
        B = move(next(it)); step(model, opt, B, mk(B))
    torch.cuda.synchronize()
    for K in (1, 4):
        t0 = time.perf_counter(); n = 0
        for _ in range(48 // K):
            opt.zero_grad(set_to_none=True)
            for j in range(K):
                B = move(next(it)); step(model, opt, B, mk(B), do_opt=False); n += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        torch.cuda.synchronize()
        print(f"  K={K}: {(time.perf_counter()-t0)/n*1000:.1f} ms/event")


if __name__ == "__main__":
    main()
