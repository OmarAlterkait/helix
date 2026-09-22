"""P13 — what the missing setuptools is actually costing.

The image's venv has no pip, setuptools, pkg_resources or wheel (uv built it
without them), so Inductor's wrapper build fails and torch.compile(model) dies
with ModuleNotFoundError. This puts a setuptools wheel on sys.path and retries,
so the answer is measured rather than asserted.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time, traceback
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, peak_mem, timeit, to_device
from kit import Pre, head_bmm
import helix.model.serial as S
from helix.model.loss import losses_cat

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=8)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")
R = {}
try:
    import setuptools
    R["setuptools"] = setuptools.__version__
except Exception as e:
    R["setuptools"] = f"MISSING ({e})"
print("setuptools:", R["setuptools"], flush=True)

evs, _ = load_events(3)
evs_d = [to_device({k: v for k, v in e.items() if torch.is_tensor(v)}, dev) for e in evs]
for e in evs_d:
    e["n_cells"] = e["plane_id"].shape[0]
model = build(device=dev)
masks = []
for i, e in enumerate(evs_d):
    g = torch.Generator(device=dev); g.manual_seed(100 + i)
    masks.append(model.make_mask(e, mode="random", gen=g))
opt = torch.optim.AdamW(model.param_groups(0.0, weight_decay=0.05), betas=(0.9, 0.95))
_ar = S.apply_rope
B, m = evs_d[0], masks[0]


def head_base(model, feat, B, m):
    occ = model.occ_head(feat) * model.readout_mult
    val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
    return losses_cat(occ, val, B, m, model.bin_edges, vis_w=0.0)


def mkstep(mod, head):
    def st():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            feat = mod.forward_feat(B, m) if hasattr(mod, "forward_feat") else mod(B, m)
            b, v = head(model, feat, B, m)
        (b + v).backward()
        opt.step()
    return st


def measure(tag, st, extra=None, warmup=5):
    try:
        t0 = time.perf_counter()
        for _ in range(warmup):
            st()
        torch.cuda.synchronize()
        warm = time.perf_counter() - t0
        t = timeit(st, warmup=2, iters=A.iters)
        with peak_mem() as pm:
            st()
        # cpu issue cost with the queue already deep
        for _ in range(2):
            st()
        torch.cuda.synchronize(); c0 = time.perf_counter()
        for _ in range(A.iters):
            st()
        cpu = (time.perf_counter() - c0) / A.iters * 1e3
        torch.cuda.synchronize()
        R["rows"][tag] = dict(ms=t["ms_med"], peak_MiB=pm["peak_alloc_MiB"],
                              cpu_ms=cpu, warmup_s=warm, **(extra or {}))
        print(f"  {tag:34s} {t['ms_med']:8.2f} ms  cpu {cpu:7.2f}  "
              f"peak {pm['peak_alloc_MiB']:8.0f} MiB  (warmup {warm:.1f}s)", flush=True)
    except Exception as e:
        R["rows"][tag] = dict(error=repr(e), tb=traceback.format_exc()[-1200:])
        print(f"  {tag:34s} FAILED {type(e).__name__}: {e}", flush=True)
    finally:
        opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()


R["rows"] = {}
print("== eager references ==", flush=True)
measure("eager base", mkstep(model, head_base), warmup=3)
S.apply_rope = Pre(cast=torch.bfloat16)
measure("eager + sparse head + rope", mkstep(model, head_bmm), warmup=3)
S.apply_rope = _ar

print("\n== torch.compile ==", flush=True)
for tag, kw, head in (
    ("compile(model) dynamic", dict(dynamic=True), head_base),
    ("compile(model) dynamic + sparse", dict(dynamic=True), head_bmm),
    ("compile(forward_feat) dynamic", dict(dynamic=True), head_base),
):
    torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
    try:
        if tag.startswith("compile(forward_feat)"):
            cf = torch.compile(model.forward_feat, **kw)
            def st():
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", torch.bfloat16):
                    feat = cf(B, m)
                    b, v = head(model, feat, B, m)
                (b + v).backward()
                opt.step()
        else:
            cm = torch.compile(model, **kw)
            if "sparse" in tag:
                S.apply_rope = Pre(cast=torch.bfloat16)
            def st():
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", torch.bfloat16):
                    o = cm(B, tok_mask=m)
                o["loss"].backward()
                opt.step()
        measure(tag, st)
    except Exception as e:
        R["rows"][tag] = dict(error=repr(e), tb=traceback.format_exc()[-1200:])
        print(f"  {tag:34s} SETUP FAILED {e}", flush=True)
    finally:
        S.apply_rope = _ar

# ---- does it survive a DIFFERENT event size? (recompiles are the real risk)
print("\n== shape churn: 3 different events round-robin ==", flush=True)
torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
try:
    cm = torch.compile(model, dynamic=True)
    def rr():
        for i in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                o = cm(evs_d[i], tok_mask=masks[i])
            o["loss"].backward()
            opt.step()
    t0 = time.perf_counter()
    for _ in range(3):
        rr()
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0
    t = timeit(rr, warmup=1, iters=4)
    R["rows"]["compiled round-robin 3 events"] = dict(ms_per_event=t["ms_med"]/3, warmup_s=warm)
    print(f"  compiled round-robin: {t['ms_med']/3:.2f} ms/event (warmup {warm:.1f}s)")
except Exception as e:
    R["rows"]["compiled round-robin 3 events"] = dict(error=repr(e))
    print("  round-robin FAILED", e)

def eager_rr():
    for i in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            o = model(evs_d[i], tok_mask=masks[i])
        o["loss"].backward()
        opt.step()
torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
t = timeit(eager_rr, warmup=2, iters=4)
R["rows"]["eager round-robin 3 events"] = dict(ms_per_event=t["ms_med"]/3)
print(f"  eager round-robin:    {t['ms_med']/3:.2f} ms/event")

emit("p13_compile", R)
