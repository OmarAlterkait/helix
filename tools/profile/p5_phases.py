"""P5 — where the GPU time actually goes, by PHASE of the serial forward.

The kernel table from P1 says GEMMs + flash attention are a minority of device
time. This attributes the rest: RoPE construction, the per-layer argsort
schedule, the grouped-attention gather/scatter, the heads, the loss.

Instrumented copies of `uniform_attn` / `grouped_cross` / `_self` / `_cross` are
byte-for-byte the shipped versions plus record_function() scopes; an equivalence
check against the real ones runs first, so a drift in serial.py fails here.
"""
from __future__ import annotations
import argparse, gc, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, timeit, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=3)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")

import helix.model.serial as S
from helix.model.fm import apply_rope as _apply_rope, rope_angles as _rope_angles

_orig = dict(uniform_attn=S.uniform_attn, grouped_cross=S.grouped_cross,
             _self=S._self, _cross=S._cross,
             apply_rope=S.apply_rope, rope_angles=S.rope_angles)


def uniform_attn_i(q, k, v, order, g):
    T, h, hd = q.shape; npad = ((T + g - 1) // g) * g; nb = npad // g
    def grp(x):
        b = x.new_empty(npad, h, hd); b[:T] = x[order]
        if npad > T: b[T:] = x[order[-1]]
        return b.view(nb, g, h, hd).permute(0, 2, 1, 3)
    with record_function("HX/attn_gather"):
        qq, kk, vv = grp(q), grp(k.to(q.dtype)), grp(v.to(q.dtype))
    with record_function("HX/sdpa_self"):
        o = F.scaled_dot_product_attention(qq, kk, vv)
    with record_function("HX/attn_scatter"):
        o = o.permute(0, 2, 1, 3).reshape(npad, h, hd)[:T]
        out = o.new_empty(T, h, hd); out[order] = o
    return out


def grouped_cross_i(q, k, v, oq, ok, g):
    Tq, h, hd = q.shape; Tk = k.shape[0]; nb = (max(Tq, Tk) + g - 1) // g
    def grp(x, order, T):
        gg = (T + nb - 1) // nb; npad = nb * gg
        b = x.new_empty(npad, h, hd); b[:T] = x[order]
        if npad > T: b[T:] = x[order[-1]]
        return b.view(nb, gg, h, hd).permute(0, 2, 1, 3)
    with record_function("HX/attn_gather"):
        qq = grp(q, oq, Tq); kk = grp(k.to(q.dtype), ok, Tk); vv = grp(v.to(q.dtype), ok, Tk)
    with record_function("HX/sdpa_cross"):
        o = F.scaled_dot_product_attention(qq, kk, vv)
    with record_function("HX/attn_scatter"):
        gq = ((Tq + nb - 1) // nb); o = o.permute(0, 2, 1, 3).reshape(nb * gq, h, hd)[:Tq]
        out = o.new_empty(Tq, h, hd); out[oq] = o
    return out


def _self_i(blk, x, at, aw, order, g, c=None):
    T, d = x.shape
    with record_function("HX/ln+qkv"):
        if blk.adaln:
            sa, ba, ga, sm, bm, gm = blk.ada(c).chunk(6, -1)
            hh = blk.n1(x) * (1 + sa) + ba
        else:
            hh = blk.n1(x)
        q, k, v = blk.qkv(hh).chunk(3, -1)
    with record_function("HX/rope_apply"):
        q = S.apply_rope(q.view(T, blk.h, blk.hd), at, aw)
        k = S.apply_rope(k.view(T, blk.h, blk.hd), at, aw)
    o = S.uniform_attn(q, k, v.view(T, blk.h, blk.hd), order, g)
    with record_function("HX/proj"):
        ao = blk.proj(o.reshape(T, d))
        x = x + (ga * ao if blk.adaln else ao)
    with record_function("HX/mlp"):
        if blk.adaln:
            return x + gm * blk.mlp(blk.n2(x) * (1 + sm) + bm)
        return x + blk.mlp(blk.n2(x))


def _cross_i(blk, q, kv, qat, qaw, kat, kaw, oq, okv, g, c=None):
    Tq, Tk = q.shape[0], kv.shape[0]
    with record_function("HX/ln+qkv"):
        if blk.adaln:
            sa, ba, ga, sm, bm, gm = blk.ada(c).chunk(6, -1)
            hq = blk.nq(q) * (1 + sa) + ba
        else:
            hq = blk.nq(q)
        qh0 = blk.q(hq).view(Tq, blk.h, blk.hd)
        k, v = blk.kv(blk.nk(kv)).chunk(2, -1)
    with record_function("HX/rope_apply"):
        qh = S.apply_rope(qh0, qat, qaw)
        kh = S.apply_rope(k.view(Tk, blk.h, blk.hd), kat, kaw)
    o = S.grouped_cross(qh, kh, v.view(Tk, blk.h, blk.hd), oq, okv, g)
    with record_function("HX/proj"):
        ao = blk.proj(o.reshape(Tq, blk.h * blk.hd))
        q = q + (ga * ao if blk.adaln else ao)
    with record_function("HX/mlp"):
        if blk.adaln:
            return q + gm * blk.mlp(blk.n2(q) * (1 + sm) + bm)
        return q + blk.mlp(blk.n2(q))


def rope_angles_i(*a, **k):
    with record_function("HX/rope_angles"):
        return _orig["rope_angles"](*a, **k)


def apply_rope_i(*a, **k):
    return _orig["apply_rope"](*a, **k)


# ---------------------------------------------------------------- equivalence
evs, _ = load_events(2)
model = build(device=dev)
B = to_device({k: v for k, v in evs[0].items() if torch.is_tensor(v)}, dev)
B["n_cells"] = B["plane_id"].shape[0]
gen = torch.Generator(device=dev); gen.manual_seed(0)
m = model.make_mask(B, mode="random", gen=gen)
print("n_cells", B["n_cells"], "masked", float(m.float().mean()))

with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    ref = model.forward_feat(B, m).float().clone()
S.uniform_attn, S.grouped_cross, S._self, S._cross = (
    uniform_attn_i, grouped_cross_i, _self_i, _cross_i)
S.rope_angles = rope_angles_i
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    got = model.forward_feat(B, m).float()
dmax = float((ref - got).abs().max())
print("instrumented-vs-shipped max|d| =", dmax)
assert dmax == 0.0, "instrumented copies drifted from serial.py"

# ---------------------------------------------------------------------- run
from helix.model.loss import losses_cat
opt = torch.optim.AdamW(model.param_groups(1.1e-3, weight_decay=0.05), betas=(0.9, 0.95))

def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        with record_function("HX/mask"):
            mm = m
        with record_function("HX/sched"):
            pass
        with record_function("HX/encode+decode"):
            feat = model.forward_feat(B, mm)
        with record_function("HX/heads"):
            occ = model.occ_head(feat) * model.readout_mult
            val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
        with record_function("HX/loss"):
            bce, vl = losses_cat(occ, val, B, mm, model.bin_edges, vis_w=0.0)
            loss = bce + vl
    with record_function("HX/backward"):
        loss.backward()
    with record_function("HX/optim"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

for _ in range(3):
    step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False, profile_memory=False, acc_events=True) as prof:
    for _ in range(A.iters):
        step()
    torch.cuda.synchronize()

ka = prof.key_averages()
it = A.iters
total_dev = sum(e.self_device_time_total for e in ka) / 1e3 / it
R = {"total_device_ms_per_step": total_dev, "n_cells": int(B["n_cells"])}

# --- named regions (inclusive) -------------------------------------------
reg = {e.key: dict(incl_ms=e.device_time_total/1e3/it, self_ms=e.self_device_time_total/1e3/it,
                   calls=e.count/it, cpu_ms=e.cpu_time_total/1e3/it)
       for e in ka if e.key.startswith("HX/")}
R["regions"] = dict(sorted(reg.items(), key=lambda kv: -kv[1]["incl_ms"]))
print(f"\ntotal device ms/step = {total_dev:.2f}")
print(f"{'region':22s} {'incl ms':>9s} {'self ms':>9s} {'% incl':>7s} {'calls':>7s} {'cpu ms':>8s}")
for k, v in R["regions"].items():
    print(f"{k:22s} {v['incl_ms']:9.2f} {v['self_ms']:9.2f} {100*v['incl_ms']/total_dev:7.2f} "
          f"{v['calls']:7.0f} {v['cpu_ms']:8.2f}")

# --- kernel classes -------------------------------------------------------
CLASS = [
    ("gemm",    ("ampere_", "cutlass", "gemm", "aten::mm", "aten::addmm", "aten::bmm", "aten::linear")),
    ("flash",   ("flash", "_flash_attention", "scaled_dot_product")),
    ("copy",    ("aten::copy_", "Memcpy", "aten::cat", "aten::contiguous", "aten::clone", "direct_copy")),
    ("index",   ("index", "gather", "scatter", "take", "nonzero", "aten::masked", "bucketize")),
    ("sort",    ("sort", "Sort", "radix", "cub::")),
    ("norm",    ("layer_norm", "LayerNorm", "GammaBeta", "Gamma")),
    ("reduce",  ("reduce", "sum", "mean", "Reduce")),
    ("elemwise",("elementwise", "aten::mul", "aten::add", "aten::sub", "aten::div",
                 "aten::exp", "aten::cos", "aten::sin", "aten::pow", "aten::fill",
                 "aten::zero", "gelu", "sigmoid", "where", "repeat", "stack")),
    ("softmax", ("softmax", "cross_entropy", "log_softmax", "nll_loss")),
    ("optim",   ("Adam", "adam", "foreach", "clip", "norm_")),
]
buckets = {n: 0.0 for n, _ in CLASS}
buckets["other"] = 0.0
other = {}
for e in ka:
    t = e.self_device_time_total / 1e3 / it
    if t <= 0:
        continue
    for n, pats in CLASS:
        if any(p in e.key for p in pats):
            buckets[n] += t
            break
    else:
        buckets["other"] += t
        other[e.key] = other.get(e.key, 0) + t
R["kernel_classes"] = dict(sorted(buckets.items(), key=lambda kv: -kv[1]))
R["other_kernels"] = dict(sorted(other.items(), key=lambda kv: -kv[1])[:20])
print(f"\n{'class':12s} {'ms/step':>9s} {'%':>7s}")
for k, v in R["kernel_classes"].items():
    print(f"{k:12s} {v:9.2f} {100*v/total_dev:7.2f}")
print("\ntop unclassified:")
for k, v in list(R["other_kernels"].items())[:12]:
    print(f"  {k[:70]:70s} {v:8.3f}")

emit("p5_phases", R)
