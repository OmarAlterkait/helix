"""P7 — what the freed memory buys: width/depth at batch 1, vs more events.

Same event, same optimizer, production head and the cheap head, across model
sizes. Reports params, step ms, peak MiB, and how many events would fit.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, traceback
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, n_params, peak_mem, timeit, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=6)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")
GPU_GiB = torch.cuda.get_device_properties(0).total_memory / 2**30

evs, _ = load_events(2)
src = {k: v for k, v in evs[0].items() if torch.is_tensor(v)}
B = to_device(src, dev)
B["n_cells"] = B["plane_id"].shape[0]
N = B["n_cells"]
print("n_cells", N)

SIZES = [
    ("d512_b12  (production)", dict(d=512,  blocks=12, dec_blocks=4, heads=8)),
    ("d512_b24",               dict(d=512,  blocks=24, dec_blocks=8, heads=8)),
    ("d768_b12",               dict(d=768,  blocks=12, dec_blocks=4, heads=12)),
    ("d768_b24",               dict(d=768,  blocks=24, dec_blocks=8, heads=12)),
    ("d1024_b12",              dict(d=1024, blocks=12, dec_blocks=4, heads=16)),
    ("d1024_b24",              dict(d=1024, blocks=24, dec_blocks=8, heads=16)),
    ("d1536_b24",              dict(d=1536, blocks=24, dec_blocks=8, heads=24)),
]
R = {"n_cells": int(N), "gpu_GiB": GPU_GiB, "rows": {}}

for name, ov in SIZES:
    try:
        model = build(ov, device=dev)
        opt = torch.optim.AdamW(model.param_groups(1.1e-3, weight_decay=0.05), betas=(0.9, 0.95))
        gen = torch.Generator(device=dev); gen.manual_seed(0)
        m = model.make_mask(B, mode="random", gen=gen)
        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                o = model(B, tok_mask=m)
            o["loss"].backward()
            opt.step()
        t = timeit(step, warmup=2, iters=A.iters)
        with peak_mem() as pm:
            step()
        p = n_params(model)
        # optimizer state is already resident at this point (2 moments fp32 + params)
        row = dict(params_M=p/1e6, ms=t["ms_med"], peak_MiB=pm["peak_alloc_MiB"],
                   MiB_per_cell=pm["peak_alloc_MiB"]/N,
                   events_fit=(GPU_GiB*1024*0.92 - (pm["peak_alloc_MiB"] - pm["delta_MiB"]))
                              / max(pm["delta_MiB"], 1),
                   tok_per_s=N/(t["ms_med"]/1e3))
        R["rows"][name] = row
        print(f"  {name:24s} {row['params_M']:7.1f}M  {row['ms']:8.1f} ms  "
              f"peak {row['peak_MiB']:9.0f} MiB  {row['tok_per_s']:,.0f} tok/s", flush=True)
    except torch.cuda.OutOfMemoryError:
        R["rows"][name] = dict(oom=True)
        print(f"  {name:24s} OOM", flush=True)
    except Exception as e:
        R["rows"][name] = dict(error=repr(e))
        print(f"  {name:24s} ERR {e}", flush=True)
    finally:
        model = opt = None
        gc.collect(); torch.cuda.empty_cache()

emit("p7_capacity", R)
