"""P9 — the two things P5/P6 pointed at.

  (a) apply_rope is 31% of trunk device time -- more than the MLP and ~6x the
      SDPA it feeds. It recomputes cos/sin of the SAME angles in all 32 calls
      and promotes bf16 activations to fp32 on the way.
  (b) 4,007 kernel launches and 77 cudaStreamSynchronize per step. Every sync
      stalls CPU run-ahead on a loop already at its launch limit.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, traceback, warnings
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, peak_mem, timeit, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=10)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")

import helix.model.serial as S
from helix.model.fm import rope_angles
from helix.model.loss import losses_cat

evs, _ = load_events(2)
model = build(device=dev)
B = to_device({k: v for k, v in evs[0].items() if torch.is_tensor(v)}, dev)
B["n_cells"] = B["plane_id"].shape[0]
gen = torch.Generator(device=dev); gen.manual_seed(0)
m = model.make_mask(B, mode="random", gen=gen)
# lr=0: the optimizer still does all of its work (so the step cost is real) but
# leaves the weights untouched, which is what lets every variant below be
# compared against ONE reference forward. With a real lr the first variant's
# timing loop trains the model and every later "deviation" is just drift.
opt = torch.optim.AdamW(model.param_groups(0.0, weight_decay=0.05), betas=(0.9, 0.95))
R = {"n_cells": int(B["n_cells"])}
_orig_rope = S.apply_rope


def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        feat = model.forward_feat(B, m)
        occ = model.occ_head(feat) * model.readout_mult
        val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
        bce, vl = losses_cat(occ, val, B, m, model.bin_edges, vis_w=0.0)
    (bce + vl).backward()
    opt.step()


# ---------------------------------------------------------- (a) rope variants
class Pre:
    """cos/sin computed ONCE per angle tensor instead of once per block.

    Cached on the angle tensor itself -- an id()-keyed dict is unsafe here,
    because CPython reuses the id of a freed tensor and the next forward then
    reads another event's table (measured: max|d| ~5 on a feature of magnitude
    ~1.4). A production version would build the tables in forward_feat and pass
    them down; this keeps the call signature so the two are comparable.
    """
    def __init__(self, cast=None):
        self.cast = cast
        self.key = "_hx_rope_bf16" if cast is not None else "_hx_rope_fp32"

    def __call__(self, x, ang_t, ang_w):
        h2 = x.shape[-1] // 2
        def tab(a):
            v = getattr(a, self.key, None)
            if v is None:
                c = torch.cos(a)[:, None, :].repeat_interleave(2, -1)
                s = torch.sin(a)[:, None, :].repeat_interleave(2, -1)
                if self.cast is not None:
                    c, s = c.to(self.cast), s.to(self.cast)
                setattr(a, self.key, (c, s))
                v = (c, s)
            return v
        def rot(v, cs):
            c, s = cs
            v2 = torch.stack([-v[..., 1::2], v[..., 0::2]], -1).reshape_as(v)
            return v * c + v2 * s
        xt = rot(x[..., :h2], tab(ang_t))
        xw = rot(x[..., h2:], tab(ang_w)) if ang_w is not None else x[..., h2:]
        return torch.cat([xt, xw], -1)


def with_rope(fn, tag):
    S.apply_rope = fn
    try:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            out = model.forward_feat(B, m).float()
        t = timeit(step, warmup=3, iters=A.iters)
        with peak_mem() as pm:
            step()
        R["rope"][tag] = dict(ms=t["ms_med"], peak_MiB=pm["peak_alloc_MiB"],
                              max_abs_dev=float((out - ref).abs().max()),
                              speedup=R["rope"]["shipped"]["ms"] / t["ms_med"]
                              if "shipped" in R["rope"] else 1.0)
        print(f"  {tag:22s} {t['ms_med']:8.2f} ms  peak {pm['peak_alloc_MiB']:8.0f} MiB  "
              f"max|d|={R['rope'][tag]['max_abs_dev']:.3e}  "
              f"x{R['rope'][tag]['speedup']:.3f}", flush=True)
    except Exception as e:
        R["rope"][tag] = dict(error=repr(e), tb=traceback.format_exc()[-700:])
        print(f"  {tag:22s} FAILED {e}")
    finally:
        S.apply_rope = _orig_rope
        opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()


print("== (a) RoPE ==", flush=True)
R["rope"] = {}
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    ref = model.forward_feat(B, m).float().clone()
with_rope(_orig_rope, "shipped")
with_rope(Pre(), "precomputed_fp32")
with_rope(Pre(cast=torch.bfloat16), "precomputed_bf16")
try:
    with_rope(torch.compile(_orig_rope, dynamic=True), "compiled")
except Exception as e:
    print("compiled rope failed", e)
torch._dynamo.reset()
try:
    p = Pre()
    with_rope(torch.compile(p.__call__, dynamic=True), "precomputed+compiled")
except Exception as e:
    print("pre+compiled failed", e)
torch._dynamo.reset()

# --------------------------------------- (a2) the grouped-attention sync fix
print("\n== (a2) uniform_attn without the per-call host sync ==", flush=True)


def uniform_attn_nosync(q, k, v, order, g):
    """serial.uniform_attn, with `x[order[-1]]` -> `x[order[-1:]]`.

    A 0-d tensor index calls item(); that is one cudaStreamSynchronize per grp()
    call, i.e. 3 per block, 48 per step. A 1-element slice indexes on device.
    """
    T, h, hd = q.shape; npad = ((T + g - 1) // g) * g; nb = npad // g
    last = order[-1:]
    def grp(x):
        b = x.new_empty(npad, h, hd); b[:T] = x[order]
        if npad > T: b[T:] = x[last]
        return b.view(nb, g, h, hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(grp(q), grp(k.to(q.dtype)), grp(v.to(q.dtype)))
    o = o.permute(0, 2, 1, 3).reshape(npad, h, hd)[:T]
    out = o.new_empty(T, h, hd); out[order] = o
    return out


def grouped_cross_nosync(q, k, v, oq, ok, g):
    Tq, h, hd = q.shape; Tk = k.shape[0]; nb = (max(Tq, Tk) + g - 1) // g
    lq, lk = oq[-1:], ok[-1:]
    def grp(x, order, T, last):
        gg = (T + nb - 1) // nb; npad = nb * gg
        b = x.new_empty(npad, h, hd); b[:T] = x[order]
        if npad > T: b[T:] = x[last]
        return b.view(nb, gg, h, hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(grp(q, oq, Tq, lq), grp(k.to(q.dtype), ok, Tk, lk),
                                       grp(v.to(q.dtype), ok, Tk, lk))
    gq = ((Tq + nb - 1) // nb); o = o.permute(0, 2, 1, 3).reshape(nb * gq, h, hd)[:Tq]
    out = o.new_empty(Tq, h, hd); out[oq] = o
    return out


_ou, _og = S.uniform_attn, S.grouped_cross
R["nosync"] = {}
for tag, use in (("shipped", False), ("nosync_attn", True), ("nosync+rope_bf16", True)):
    if use:
        S.uniform_attn, S.grouped_cross = uniform_attn_nosync, grouped_cross_nosync
    if tag == "nosync+rope_bf16":
        S.apply_rope = Pre(cast=torch.bfloat16)
    try:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            out = model.forward_feat(B, m).float()
        t = timeit(step, warmup=3, iters=A.iters)
        import time as _t
        for _ in range(3):
            step()
        torch.cuda.synchronize(); t0 = _t.perf_counter()
        for _ in range(A.iters):
            step()
        cpu = (_t.perf_counter() - t0) / A.iters * 1e3
        torch.cuda.synchronize()
        R["nosync"][tag] = dict(ms=t["ms_med"], cpu_ms=cpu,
                                max_abs_dev=float((out - ref).abs().max()))
        print(f"  {tag:20s} {t['ms_med']:8.2f} ms  cpu {cpu:7.2f} ms  "
              f"max|d|={R['nosync'][tag]['max_abs_dev']:.3e}", flush=True)
    except Exception as e:
        R["nosync"][tag] = dict(error=repr(e))
        print(f"  {tag:20s} FAILED {e}")
    finally:
        S.uniform_attn, S.grouped_cross = _ou, _og
        S.apply_rope = _orig_rope
        opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------- (b) host syncs, located
print("\n== (b) host synchronisation points ==", flush=True)
hits = {}
def show(message, category, filename, lineno, file=None, line=None):
    st = [f for f in traceback.extract_stack()[:-1]
          if "/helix/" in f.filename or "/prof/" in f.filename]
    key = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in st[-4:])
    hits[key] = hits.get(key, 0) + 1

torch.cuda.set_sync_debug_mode("warn")
old = warnings.showwarning
warnings.showwarning = show
warnings.simplefilter("always")
try:
    step()
finally:
    warnings.showwarning = old
    torch.cuda.set_sync_debug_mode("default")
R["sync_sites"] = dict(sorted(hits.items(), key=lambda kv: -kv[1]))
print(f"  {sum(hits.values())} sync events at {len(hits)} sites")
for k, v in list(R["sync_sites"].items())[:25]:
    print(f"   {v:4d}x  {k}")

emit("p9_rope_sync", R)
