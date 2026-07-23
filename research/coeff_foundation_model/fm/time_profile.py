"""Step-TIMING profile: is the FMModel compute-bound at one event (so batching
won't help wall-clock and we must CUT compute), or under-utilized (batching helps)?
Also: does the CrossMAE decoder actually shrink the step, and is data-loading the
real bottleneck vs the pure-GPU compute floor?"""
import glob, time, torch
import data as D
from model import FMModel, losses
D.init_pipeline_cpu()
dev = "cuda"
paths = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))
B = D.get_cached(paths[0], device=dev)
N = B["inp"].shape[0]
mask = (torch.rand(N, device=dev) < 0.75)
print(f"GPU {torch.cuda.get_device_name()} | event N={N} tokens\n")


def make(dec_mode):
    torch.manual_seed(0)
    m = FMModel(128, D.N_BAND, 6, d=512, blocks=12, dec_blocks=4, heads=8, dec_mode=dec_mode).to(dev)
    return m, torch.optim.AdamW(m.parameters(), 1e-4)


def timed(m, opt, Bx, mk, iters=10):
    def one():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = m(Bx, mk); bce, val = losses(occ, mu, lv, Bx, mk); loss = bce + val
        loss.backward(); opt.step()
    for _ in range(3): one()                       # warmup
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


for dm in ("self", "cross"):
    m, opt = make(dm)
    t = timed(m, opt, B, mask)
    print(f"decoder={dm:5s}: {t*1000:6.1f} ms/step  ({N/t/1e6:.2f} M tok/s)")

# data-loading floor: how long to load+assemble one cached event (CPU) + move to GPU?
t0 = time.perf_counter()
for p in paths[1:6]:
    b = D.get_cached(p, device=dev); torch.cuda.synchronize()
print(f"\ndata load+assemble+toGPU: {(time.perf_counter()-t0)/5*1000:.1f} ms/event "
      f"(vs the ~190 ms/step the real run shows -> if << step, compute-bound)")

# N-scaling: 1 vs 2 events worth of tokens (concat a 2nd event) -> does step time ~double?
B2 = D.get_cached(paths[1], device=dev)
def cat(a, b, keys):
    out = {}
    nc = a["inp"].shape[0]
    for k in a:
        if k == "cell": out[k] = torch.cat([a[k], b[k] + nc]);
        elif k == "slot": out[k] = torch.cat([a[k], b[k]])
        elif k == "n_cells": out[k] = a[k] + b[k]
        elif torch.is_tensor(a[k]) and a[k].shape and a[k].shape[0] in (nc, a["target"].shape[0]):
            out[k] = torch.cat([a[k], b[k]])
        else: out[k] = a[k]
    return out
try:
    Bc = cat(B, B2, None); Nc = Bc["inp"].shape[0]; mc = (torch.rand(Nc, device=dev) < 0.75)
    m, opt = make("self")
    t1 = timed(m, opt, B, mask); t2 = timed(m, opt, Bc, mc)
    print(f"\nN-scaling (self): N={N} -> {t1*1000:.1f}ms ; 2 events N={Nc} -> {t2*1000:.1f}ms "
          f"(ratio {t2/t1:.2f}x; ~2x = compute-bound, <2x = under-utilized at 1 event)")
except Exception as e:
    print(f"\nN-scaling test failed: {str(e)[:90]}")
