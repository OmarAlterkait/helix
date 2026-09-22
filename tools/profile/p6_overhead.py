"""P6 — the CPU side. P3 measured 120 ms of CPU issue against a 122 ms step:
the loop is launch-bound as much as it is GPU-bound. This attributes the CPU
time by phase, counts kernel launches, and measures what removes them.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time, traceback
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, peak_mem, timeit, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=10)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")

evs, _ = load_events(2)
model = build(device=dev)
B = to_device({k: v for k, v in evs[0].items() if torch.is_tensor(v)}, dev)
B["n_cells"] = B["plane_id"].shape[0]
gen = torch.Generator(device=dev); gen.manual_seed(0)
m = model.make_mask(B, mode="random", gen=gen)
N = B["n_cells"]
from helix.model.loss import losses_cat
R = {"n_cells": int(N)}


def cpu_ms(fn, iters=None, warmup=3):
    """Wall time with the GPU left async -> the CPU's own issue cost."""
    iters = iters or A.iters
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return dict(cpu_ms=(t1 - t0) / iters * 1e3, wall_ms=(t2 - t0) / iters * 1e3)


opt = torch.optim.AdamW(model.param_groups(1.1e-3, weight_decay=0.05), betas=(0.9, 0.95))

def f_fwd():
    with torch.autocast("cuda", torch.bfloat16):
        feat = model.forward_feat(B, m)
        occ = model.occ_head(feat) * model.readout_mult
        val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
        bce, vl = losses_cat(occ, val, B, m, model.bin_edges, vis_w=0.0)
    return bce + vl

def f_trunk():
    with torch.autocast("cuda", torch.bfloat16):
        return model.forward_feat(B, m)

def f_fwdbwd():
    opt.zero_grad(set_to_none=True)
    f_fwd().backward()

def f_clip():
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

def f_optstep():
    opt.step()

def f_full():
    opt.zero_grad(set_to_none=True)
    f_fwd().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

print("== CPU issue time by phase (GPU async) ==", flush=True)
f_fwdbwd()   # populate grads for clip/step
phases = {}
for nm, fn in (("trunk_fwd", f_trunk), ("fwd(+heads+loss)", f_fwd),
               ("fwd+bwd", f_fwdbwd), ("clip_grad_norm", f_clip),
               ("opt.step", f_optstep), ("full_step", f_full)):
    phases[nm] = cpu_ms(fn)
    print(f"  {nm:18s} cpu={phases[nm]['cpu_ms']:8.2f} ms  wall={phases[nm]['wall_ms']:8.2f} ms")
R["phases_cpu"] = phases

# ------------------------------------------------------ kernel launch count
from torch.profiler import ProfilerActivity, profile
for _ in range(3):
    f_full()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True) as prof:
    for _ in range(3):
        f_full()
    torch.cuda.synchronize()
evts = prof.events()
launches = sum(1 for e in evts if getattr(e, "device_type", None) is not None
               and str(getattr(e, "device_type", "")).endswith("CUDA")
               and getattr(e, "self_device_time_total", 0) > 0)
ka = prof.key_averages()
n_dev_kernels = sum(e.count for e in ka if e.self_device_time_total > 0) / 3
n_aten = sum(e.count for e in ka if e.key.startswith("aten::")) / 3
R["launches"] = dict(device_kernel_calls_per_step=n_dev_kernels, aten_calls_per_step=n_aten)
print(f"  device kernel calls/step ~ {n_dev_kernels:.0f};  aten calls/step ~ {n_aten:.0f}")
top_cpu = sorted(ka, key=lambda e: -e.self_cpu_time_total)[:20]
R["top_cpu_ops"] = [dict(name=e.key, self_cpu_ms=e.self_cpu_time_total/1e3/3, calls=e.count//3)
                    for e in top_cpu]
print(f"\n{'op (by self CPU)':52s} {'ms/step':>9s} {'calls':>7s}")
for e in R["top_cpu_ops"]:
    print(f"  {e['name'][:50]:52s} {e['self_cpu_ms']:9.3f} {e['calls']:7d}")

# ------------------------------------------------------------- interventions
print("\n== interventions ==", flush=True)
base = timeit(f_full, warmup=3, iters=A.iters)
with peak_mem() as pmb:
    f_full()
R["variants"] = {"base": dict(ms=base["ms_med"], peak_MiB=pmb["peak_alloc_MiB"],
                              **cpu_ms(f_full))}
print(f"  base            {base['ms_med']:8.2f} ms")

def record(name, fn, extra=None):
    try:
        t = timeit(fn, warmup=5, iters=A.iters)
        c = cpu_ms(fn)
        with peak_mem() as pm:
            fn()
        R["variants"][name] = dict(ms=t["ms_med"], peak_MiB=pm["peak_alloc_MiB"], **c,
                                   speedup=R["variants"]["base"]["ms"] / t["ms_med"], **(extra or {}))
        print(f"  {name:16s} {t['ms_med']:8.2f} ms  cpu {c['cpu_ms']:7.2f}  "
              f"peak {pm['peak_alloc_MiB']:8.0f} MiB  x{R['variants']['base']['ms']/t['ms_med']:.2f}", flush=True)
    except Exception as e:
        R["variants"][name] = dict(error=repr(e), tb=traceback.format_exc()[-800:])
        print(f"  {name:16s} FAILED {e}", flush=True)
    gc.collect(); torch.cuda.empty_cache()

# 1. fused AdamW
opt_f = torch.optim.AdamW(model.param_groups(1.1e-3, weight_decay=0.05),
                          betas=(0.9, 0.95), fused=True)
def full_fused():
    opt_f.zero_grad(set_to_none=True)
    f_fwd().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    opt_f.step()
record("fused_adamw", full_fused)

# 2. no clip (isolate its cost)
def full_noclip():
    opt.zero_grad(set_to_none=True)
    f_fwd().backward()
    opt.step()
record("no_clip", full_noclip)

# 3. compiled RoPE only
import helix.model.serial as S
_ar = S.apply_rope
try:
    S.apply_rope = torch.compile(_ar, dynamic=True)
    record("compiled_rope", f_full)
finally:
    S.apply_rope = _ar

# 4. torch.compile(model) default / reduce-overhead
for mode in ("default", "max-autotune-no-cudagraphs"):
    try:
        cm = torch.compile(model, dynamic=True, mode=None if mode == "default" else mode)
        def cstep():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                o = cm(B, tok_mask=m)
            o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        record(f"compile_{mode}", cstep)
    except Exception as e:
        R["variants"][f"compile_{mode}"] = dict(error=repr(e))
        print(f"  compile_{mode} FAILED {e}")
    torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()

emit("p6_overhead", R)
