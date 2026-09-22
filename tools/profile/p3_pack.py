"""P3 — the three ways to raise the effective batch on ONE GPU, measured.

  seq   : K sequential steps, grads accumulated   (works today, no model change)
  pack  : K events concatenated into one forward  (what MULTI_EVENT_BATCHING
          defers; measured here for cost only -- attention leaks across events)
  ceil  : how many real events fit before OOM

Also isolates the CPU-side floor (python + kernel launch) from GPU time, which
the token-count sweep showed is what the small-N regime is actually measuring.
"""
from __future__ import annotations
import argparse, gc, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, pack, peak_mem, timeit, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--kmax", type=int, default=8)
ap.add_argument("--iters", type=int, default=6)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")

evs, cpu_t = load_events(A.kmax + 4)
evs_d = [to_device(e, dev) for e in evs]
ncell = [int(e["plane_id"].shape[0]) for e in evs_d]
print("n_cells:", ncell)

model = build(device=dev)
opt = torch.optim.AdamW(model.param_groups(1.1e-3, weight_decay=0.05), betas=(0.9, 0.95))
R = {"n_cells": ncell, "cpu_pipeline_s": {k: float(np.mean(v)) for k, v in cpu_t.items()}}

def step_seq(K):
    """K events, one at a time, gradients accumulated -> one optimizer step."""
    opt.zero_grad(set_to_none=True)
    for i in range(K):
        B = evs_d[i]
        with torch.autocast("cuda", torch.bfloat16):
            o = model(B)
        (o["loss"] / K).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

packed = {}
def step_pack(K):
    opt.zero_grad(set_to_none=True)
    B = packed[K]
    with torch.autocast("cuda", torch.bfloat16):
        o = model(B)
    o["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

rows = []
for K in range(1, A.kmax + 1):
    row = dict(K=K, cells=sum(ncell[:K]))
    try:
        t = timeit(lambda: step_seq(K), warmup=2, iters=A.iters)
        with peak_mem() as pm:
            step_seq(K)
        row.update(seq_ms=t["ms_med"], seq_peak_MiB=pm["peak_alloc_MiB"],
                   seq_ev_per_s=K / (t["ms_med"] / 1e3))
    except torch.cuda.OutOfMemoryError:
        row["seq_oom"] = True
        torch.cuda.empty_cache()
    gc.collect(); torch.cuda.empty_cache()
    try:
        packed[K] = pack(evs_d[:K])
        t = timeit(lambda: step_pack(K), warmup=2, iters=A.iters)
        with peak_mem() as pm:
            step_pack(K)
        row.update(pack_ms=t["ms_med"], pack_peak_MiB=pm["peak_alloc_MiB"],
                   pack_ev_per_s=K / (t["ms_med"] / 1e3))
        row["speedup_pack_vs_seq"] = row.get("seq_ms", float("nan")) / t["ms_med"]
    except torch.cuda.OutOfMemoryError:
        row["pack_oom"] = True
        packed.pop(K, None)
        torch.cuda.empty_cache()
    rows.append(row)
    print(f"  K={K} cells={row['cells']:7d} "
          f"seq={row.get('seq_ms', float('nan')):8.1f}ms/{row.get('seq_peak_MiB', 0):8.0f}MiB "
          f"pack={row.get('pack_ms', float('nan')):8.1f}ms/{row.get('pack_peak_MiB', 0):8.0f}MiB "
          f"x{row.get('speedup_pack_vs_seq', float('nan')):.2f}", flush=True)
    opt.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache()
R["pack_vs_seq"] = rows

# ------------------------------------------------- CPU floor (launch-bound?)
print("== CPU-side cost of one step (python + launches, GPU async) ==", flush=True)
B = evs_d[0]
def cpu_only():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        o = model(B)
    o["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
for _ in range(3):
    cpu_only()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(A.iters):
    cpu_only()
t_cpu = (time.perf_counter() - t0) / A.iters * 1e3
torch.cuda.synchronize()
t1 = time.perf_counter()
t_wall = (t1 - t0) / A.iters * 1e3
R["cpu_floor"] = dict(cpu_issue_ms=t_cpu, wall_ms=t_wall,
                      gpu_bound=t_wall > t_cpu)
print(f"  cpu issue {t_cpu:.1f} ms   wall {t_wall:.1f} ms")

# ------------------------------------------- how many events actually fit
print("== memory ceiling: largest packed token set before OOM ==", flush=True)
src = evs_d[int(np.argmax(ncell))]
from common import synth_from
lo, hi, best = 1000, 400000, 0
while lo <= hi:
    mid = (lo + hi) // 2
    try:
        Bs = to_device(synth_from({k: v.cpu() for k, v in src.items() if torch.is_tensor(v)}, mid), dev)
        Bs["n_cells"] = mid
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            o = model(Bs)
        o["loss"].backward()
        opt.step()
        torch.cuda.synchronize()
        best = mid; lo = mid + 5000
        print(f"   {mid} OK (peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB)", flush=True)
    except torch.cuda.OutOfMemoryError:
        hi = mid - 5000
        print(f"   {mid} OOM", flush=True)
    finally:
        opt.zero_grad(set_to_none=True)
        for v in list(locals().get("Bs", {}) or {}):
            pass
        Bs = None; o = None
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
R["max_cells_one_gpu"] = dict(cells=best, events_equiv=best / float(np.mean(ncell)))
print("max cells:", best, "=", R["max_cells_one_gpu"]["events_equiv"], "events")

emit("p3_pack", R)
