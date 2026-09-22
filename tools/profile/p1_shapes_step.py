"""P1 — what a batch IS, and where one training step spends time and memory.

Real corpus events, production (m113/8run) architecture, bf16 autocast, the
same AdamW+clip the config uses.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import (M113, ModuleProfiler, build, emit, leaf_rollup, load_events,
                    n_params, pack, peak_mem, timeit, to_device)

ap = argparse.ArgumentParser()
ap.add_argument("--events", type=int, default=64)
ap.add_argument("--iters", type=int, default=10)
A = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
dev = "cuda"
R = {}

# --------------------------------------------------------------- 1. shapes
print("== loading events ==", flush=True)
evs, cpu_t = load_events(A.events)
R["cpu_pipeline"] = dict(
    read_s_mean=float(np.mean(cpu_t["read_s"])), read_s_med=float(np.median(cpu_t["read_s"])),
    tokenize_s_mean=float(np.mean(cpu_t["tokenize_s"])),
    tokenize_s_med=float(np.median(cpu_t["tokenize_s"])),
    total_s_mean=float(np.mean(cpu_t["read_s"]) + np.mean(cpu_t["tokenize_s"])))
print(json.dumps(R["cpu_pipeline"], indent=1))

ncells = np.array([e["plane_id"].shape[0] for e in evs])
nact = np.array([e["cell"].shape[0] for e in evs])
R["tokens"] = dict(
    n_events=len(evs),
    n_cells=dict(mean=float(ncells.mean()), med=float(np.median(ncells)),
                 min=int(ncells.min()), max=int(ncells.max()),
                 p95=float(np.percentile(ncells, 95)), std=float(ncells.std())),
    n_active=dict(mean=float(nact.mean()), min=int(nact.min()), max=int(nact.max())),
    slot_occupancy=float(nact.mean() / (ncells.mean() * 128)),
    valid_frac=float(evs[0]["valid"].float().mean()),
)
print(json.dumps(R["tokens"], indent=1))

# per-key bytes for the median event
mid = evs[int(np.argsort(ncells)[len(evs)//2])]
kb = {}
for k, v in mid.items():
    if torch.is_tensor(v):
        kb[k] = dict(shape=list(v.shape), dtype=str(v.dtype),
                     MiB=v.numel()*v.element_size()/2**20)
R["batch_bytes"] = dict(keys=dict(sorted(kb.items(), key=lambda kv: -kv[1]["MiB"])),
                        total_MiB=sum(v["MiB"] for v in kb.values()),
                        n_cells=int(mid["plane_id"].shape[0]))
print("batch total MiB", R["batch_bytes"]["total_MiB"], "n_cells", R["batch_bytes"]["n_cells"])

# --------------------------------------------------------------- 2. model
model = build(device=dev)
R["model"] = dict(params=n_params(model), params_MiB=n_params(model)*4/2**20,
                  cfg={k: v for k, v in M113.items() if k != "type"})
print("params", R["model"]["params"])

B = to_device(mid, dev)
N = B["plane_id"].shape[0]

# --------------------------------------------------------- 3. stage memory
print("== stage-by-stage peak memory (bf16 autocast, forward only) ==", flush=True)
from helix.model.loss import losses_cat
stages = {}
torch.cuda.empty_cache()
gen = torch.Generator(device=dev); gen.manual_seed(0)

def stage(name, fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    b0 = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    stages[name] = dict(peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                        delta_peak_MiB=(torch.cuda.max_memory_allocated()-b0)/2**20,
                        live_after_MiB=(torch.cuda.memory_allocated()-b0)/2**20)
    print(f"  {name:22s} peak={stages[name]['peak_MiB']:9.1f} "
          f"dpeak={stages[name]['delta_peak_MiB']:9.1f} live={stages[name]['live_after_MiB']:9.1f}")
    return out

with torch.autocast("cuda", torch.bfloat16):
    m = stage("mask", lambda: model.make_mask(dict(B, n_cells=N), gen=gen))
    feat = stage("forward_feat", lambda: model.forward_feat(B, m))
    occ = stage("occ_head", lambda: model.occ_head(feat) * model.readout_mult)
    val = stage("val_head", lambda: (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins))
    lo = stage("losses_cat", lambda: losses_cat(occ, val, B, m, model.bin_edges, vis_w=model.vis_w))
loss = lo[0] + lo[1]
stage("backward", lambda: loss.backward())
R["stage_memory_fwd"] = stages
R["mask_frac"] = float(m.float().mean())
del feat, occ, val, lo, loss
model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()

# ------------------------------------------------ 4. full step: time + memory
opt = torch.optim.AdamW(model.param_groups(1.1e-3, weight_decay=0.05), betas=(0.9, 0.95))

def full_step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        out = model(B)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

def fwd_only():
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        model(B)

def fwd_bwd():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        out = model(B)
    out["loss"].backward()

print("== step timing ==", flush=True)
R["timing"] = dict(fwd_nograd=timeit(fwd_only, iters=A.iters),
                   fwd_bwd=timeit(fwd_bwd, iters=A.iters),
                   full_step=timeit(full_step, iters=A.iters))
for k, v in R["timing"].items():
    print(f"  {k:12s} {v['ms_med']:8.2f} ms (min {v['ms_min']:.2f})")

with peak_mem() as pm:
    full_step()
R["full_step_memory"] = dict(pm)
R["full_step_memory"]["n_cells"] = N
print("full step peak alloc MiB", pm["peak_alloc_MiB"], "reserved", pm["peak_reserved_MiB"])

# ---------------------------------------------- 5. per-module attribution
print("== per-module (forward) ==", flush=True)
opt.zero_grad(set_to_none=True)
with ModuleProfiler(model) as mp:
    with torch.autocast("cuda", torch.bfloat16):
        out = model(B)
    rows = mp.finish()
out["loss"].backward()
opt.zero_grad(set_to_none=True)
R["modules_leaf_by_type"] = leaf_rollup(rows)
R["modules_top"] = dict(sorted(((k, v) for k, v in rows.items()),
                               key=lambda kv: -kv[1]["ms"])[:40])
for t, v in R["modules_leaf_by_type"].items():
    print(f"  {t:16s} calls={v['calls']:5d} ms={v['ms']:8.2f} alloc={v['alloc_MiB']:9.1f} MiB")

# ---------------------------------------------- 6. torch.profiler kernels
print("== torch.profiler ==", flush=True)
from torch.profiler import profile, ProfilerActivity
for _ in range(3):
    full_step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True, with_stack=False) as prof:
    for _ in range(3):
        full_step()
    torch.cuda.synchronize()
ka = prof.key_averages()
tot = sum(e.self_device_time_total for e in ka)
top = sorted(ka, key=lambda e: -e.self_device_time_total)[:30]
R["kernels"] = [dict(name=e.key, self_cuda_ms=e.self_device_time_total/1e3/3,
                     pct=100*e.self_device_time_total/max(tot,1), calls=e.count//3,
                     cuda_mem_MiB=getattr(e, "self_device_memory_usage", 0)/2**20/3)
                for e in top]
print(f"{'kernel':60s} {'ms/step':>9s} {'%':>6s} {'calls':>6s}")
for e in R["kernels"]:
    print(f"{e['name'][:60]:60s} {e['self_cuda_ms']:9.3f} {e['pct']:6.2f} {e['calls']:6d}")
tr = os.path.join(os.environ.get("PROF_OUT", "."), "p1_trace.json")
try:
    prof.export_chrome_trace(tr); print("trace ->", tr)
except Exception as e:
    print("trace export failed", e)

emit("p1_shapes_step", R)
