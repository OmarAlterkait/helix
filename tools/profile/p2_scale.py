"""P2 — cost vs token count, on THIS GPU.

MULTI_EVENT_BATCHING.md's whole argument rests on a us/cell curve measured on a
2080 Ti and says to re-measure if training moves to A100/H100. This is that
re-measurement, plus the memory curve the original did not record, plus a
trunk-vs-head split so the saturation point is attributed.
"""
from __future__ import annotations
import argparse, gc, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, peak_mem, synth_from, timeit, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--cells", type=str,
                default="2000,4000,8000,16000,24000,32000,40000,48000,64000,80000,96000,128000,160000,192000,256000")
ap.add_argument("--iters", type=int, default=8)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")

evs, _ = load_events(4)
src = evs[0]
print("source event n_cells", src["plane_id"].shape[0])

VARIANTS = {
    "prod_cat128":  dict(),                      # production: categorical head K=128
    "cat32":        dict(n_bins=32),
    "gauss_nll":    dict(n_bins=0, nll=True, loss_fused=True),
    "trunk_only":   None,                        # forward_feat only, no head/loss
}

R = {"variants": {}, "source_n_cells": int(src["plane_id"].shape[0])}
cells = [int(c) for c in A.cells.split(",")]

for vname, ov in VARIANTS.items():
    print(f"\n==== {vname} ====", flush=True)
    model = build(ov if ov is not None else {}, device=dev)
    opt = torch.optim.AdamW(model.param_groups(1.1e-3, weight_decay=0.05), betas=(0.9, 0.95))
    rows = []
    for nc in cells:
        try:
            B = to_device(synth_from(src, nc), dev)
            B["n_cells"] = nc
            gen = torch.Generator(device=dev); gen.manual_seed(0)
            m = model.make_mask(B, mode="random", gen=gen)

            if ov is None:
                def fwd():
                    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                        model.forward_feat(B, m)
                def fb():
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", torch.bfloat16):
                        x = model.forward_feat(B, m)
                    x.float().pow(2).mean().backward()
            else:
                def fwd():
                    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                        model(B, tok_mask=m)
                def fb():
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", torch.bfloat16):
                        o = model(B, tok_mask=m)
                    o["loss"].backward()

            tf = timeit(fwd, warmup=2, iters=A.iters)
            with peak_mem() as pmf:
                fwd()
            tb = timeit(fb, warmup=2, iters=A.iters)
            with peak_mem() as pmb:
                fb()
            opt.zero_grad(set_to_none=True)
            row = dict(n_cells=nc, fwd_ms=tf["ms_med"], fb_ms=tb["ms_med"],
                       fwd_us_per_cell=tf["ms_med"]*1e3/nc,
                       fb_us_per_cell=tb["ms_med"]*1e3/nc,
                       fwd_peak_MiB=pmf["peak_alloc_MiB"], fb_peak_MiB=pmb["peak_alloc_MiB"],
                       fb_delta_MiB=pmb["delta_MiB"],
                       tok_per_s=nc/(tb["ms_med"]/1e3))
            rows.append(row)
            print(f"  N={nc:7d} fwd={row['fwd_ms']:8.2f}ms ({row['fwd_us_per_cell']:5.2f} us/cell) "
                  f"f+b={row['fb_ms']:8.2f}ms ({row['fb_us_per_cell']:5.2f}) "
                  f"peak_fb={row['fb_peak_MiB']:8.1f}MiB tok/s={row['tok_per_s']:,.0f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  N={nc:7d} OOM", flush=True)
            rows.append(dict(n_cells=nc, oom=True))
            torch.cuda.empty_cache()
            break
        finally:
            for v in ("B", "m"):
                if v in dir():
                    pass
            gc.collect(); torch.cuda.empty_cache()
    R["variants"][vname] = rows
    del model, opt
    gc.collect(); torch.cuda.empty_cache()

emit("p2_scale", R)
