"""P15 — the GPU is 95% busy at 25% efficiency. What is it busy WITH?

Three parts:
  1. this card's achieved HBM bandwidth, as the denominator for everything else;
  2. finer instrumentation than p5 -- `_emb`, `_sched`, the vis/mask subsetting
     and the final index_copy were 13.2 ms of "unnamed" there;
  3. an encoder that carries the residual stream ALREADY PERMUTED AND PADDED, so
     a block costs ONE gather instead of three gathers plus a scatter.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time, traceback
import numpy as np, torch, torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, peak_mem, timeit, to_device
import helix.model.serial as S
from helix.model.fm import rope_angles
from helix.model.loss import losses_cat

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=8)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")
R = {}

# ------------------------------------------------- 1. HBM bandwidth reference
print("== achieved HBM bandwidth ==", flush=True)
bw = {}
for mb in (64, 256, 1024):
    n = mb * 2**20 // 2
    a = torch.empty(n, dtype=torch.bfloat16, device=dev)
    b = torch.empty(n, dtype=torch.bfloat16, device=dev)
    t = timeit(lambda: a.copy_(b), warmup=5, iters=20)
    bw[f"copy_{mb}MiB"] = 2 * mb / 2**10 / (t["ms_med"] / 1e3)      # GiB/s r+w
    t2 = timeit(lambda: a.mul_(1.0001), warmup=5, iters=20)
    bw[f"scale_{mb}MiB"] = 2 * mb / 2**10 / (t2["ms_med"] / 1e3)
    del a, b
    gc.collect(); torch.cuda.empty_cache()
# gather, which is what the grouped attention actually does
n = 32500 * 512
a = torch.randn(32500, 512, dtype=torch.bfloat16, device=dev)
idx = torch.randperm(32500, device=dev)
t = timeit(lambda: a[idx], warmup=5, iters=20)
bw["gather_32500x512"] = 2 * (32500 * 512 * 2) / 2**30 / (t["ms_med"] / 1e3)
R["bandwidth_GiBs"] = bw
for k, v in bw.items():
    print(f"  {k:22s} {v:8.1f} GiB/s")
del a, idx; gc.collect(); torch.cuda.empty_cache()

# ------------------------------------------------------------- setup
evs, _ = load_events(2)
model = build(device=dev)
B = to_device({k: v for k, v in evs[0].items() if torch.is_tensor(v)}, dev)
B["n_cells"] = B["plane_id"].shape[0]
gen = torch.Generator(device=dev); gen.manual_seed(0)
m = model.make_mask(B, mode="random", gen=gen)
T_all = int(B["n_cells"]); T_vis = int((~m).sum()); T_msk = int(m.sum())
R["shapes"] = dict(n_cells=T_all, visible=T_vis, masked=T_msk, d=model.d)
print(f"\nn_cells={T_all} visible={T_vis} masked={T_msk}")
opt = torch.optim.AdamW(model.param_groups(0.0, weight_decay=0.05), betas=(0.9, 0.95))

# --------------------------------------- 2. finer instrumentation of the trunk
_orig = dict(u=S.uniform_attn, gc_=S.grouped_cross, se=S._self, cr=S._cross)


def ffeat_instr(model, Bx, tok_mask):
    """serial.forward_feat, verbatim, with record_function around the parts p5
    lumped into 'unnamed'."""
    N = Bx["inp"].shape[0]
    with record_function("HX/rope_angles"):
        at = rope_angles(Bx["t_phys"], model.d // model.heads, *model.lam_t)
        aw = rope_angles(Bx["wire_pos"], model.d // model.heads, *model.lam_w)
    with record_function("HX/mask_idx"):
        vis = ~tok_mask
        vis_idx = vis.nonzero(as_tuple=True)[0]
        mask_idx = tok_mask.nonzero(as_tuple=True)[0]
    with record_function("HX/emb"):
        xv = model._emb(Bx, vis_idx)
    with record_function("HX/subset_angles"):
        atv, awv = at[vis], aw[vis]
    with record_function("HX/sched"):
        sched = model._sched(Bx["plane_id"][vis], Bx["t_phys"][vis], Bx["wire_pos"][vis])
    c = model._cond(Bx) if model.cond == "adaln" else None
    cv = c[vis] if c is not None else None
    with record_function("HX/encoder"):
        for blk, (o, g, uw) in zip(model.enc, sched):
            xv = S._self(blk, xv, atv, awv if uw else None, o, g, cv)
    with record_function("HX/dec_queries"):
        qm = model.mask_tok.expand(mask_idx.numel(), model.d)
        if model.film is not None:
            g_, b_ = model.film(Bx["band_id"][tok_mask], Bx["plane_id"][tok_mask],
                                Bx["wirefeat"][tok_mask])
            qm = g_ * qm + b_
        qm = (qm + model.band_emb(Bx["band_id"][tok_mask])
              + model.plane_emb(Bx["plane_id"][tok_mask])).to(xv.dtype)
        atm, awm = at[tok_mask], aw[tok_mask]
        oq = torch.argsort(Bx["t_phys"][tok_mask].double())
        okv = torch.argsort(Bx["t_phys"][vis].double())
    with record_function("HX/decoder"):
        for blk in model.dec:
            qm = S._cross(blk, qm, xv, atm, awm, atv, awv, oq, okv, model.gd, None)
    with record_function("HX/scatter_out"):
        x = torch.zeros(N, model.d, dtype=xv.dtype, device=xv.device)
        x = x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm)
        out = model.dec_norm(x)
    return out


print("\n== forward, finer phases ==", flush=True)
for _ in range(3):
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        ffeat_instr(model, B, m)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA], acc_events=True) as pr:
    for _ in range(A.iters):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            ffeat_instr(model, B, m)
    torch.cuda.synchronize()
ka = pr.key_averages()
reg = {e.key: e.device_time_total / 1e3 / A.iters for e in ka if e.key.startswith("HX/")}
tot = reg.get("HX/encoder", 0) + reg.get("HX/decoder", 0) + sum(
    v for k, v in reg.items() if k not in ("HX/encoder", "HX/decoder"))
R["fwd_phases_ms"] = dict(sorted(reg.items(), key=lambda kv: -kv[1]))
print(f"{'phase':22s} {'ms':>8s} {'%':>7s}")
for k, v in R["fwd_phases_ms"].items():
    print(f"  {k:20s} {v:8.3f} {100*v/max(tot,1e-9):7.2f}")

# --------------------------- 3. permuted-residual encoder (one gather / block)
def build_plan(orders, gs, T, device):
    """For each block: the gather index from the PREVIOUS block's padded layout
    into this block's, plus (nb, g). Entry gathers from natural order; exit
    returns to it."""
    plan = []
    prev_inv = None                                     # None == natural order
    ar = torch.arange(T, device=device)
    for o, g in zip(orders, gs):
        npad = ((T + g - 1) // g) * g
        pos = torch.arange(npad, device=device).clamp(max=T - 1)
        tok = o[pos]                                    # token id at each padded slot
        src = tok if prev_inv is None else prev_inv[tok]
        inv = torch.empty(T, dtype=torch.long, device=device)
        inv[o] = ar
        plan.append((src, npad // g, g))
        prev_inv = inv
    return plan, prev_inv


def rope_tables(ang, dtype):
    c = torch.cos(ang)[:, None, :].repeat_interleave(2, -1).to(dtype)
    s = torch.sin(ang)[:, None, :].repeat_interleave(2, -1).to(dtype)
    return c, s


def rope_pre(x, tt, tw):
    h2 = x.shape[-1] // 2
    def rot(v, cs):
        c, s = cs
        v2 = torch.stack([-v[..., 1::2], v[..., 0::2]], -1).reshape_as(v)
        return v * c + v2 * s
    xt = rot(x[..., :h2], tt)
    xw = rot(x[..., h2:], tw) if tw is not None else x[..., h2:]
    return torch.cat([xt, xw], -1)


def _self_perm(blk, xp, tt, tw, nb, g):
    """Block.forward in ALREADY-PERMUTED, ALREADY-PADDED space.

    Everything except attention is row-wise, so it does not care about order;
    the pad rows are exact duplicates of the block's last real token and stay
    duplicates through the block, which is why this is the same computation.
    """
    P, d = xp.shape
    hh = blk.n1(xp)
    q, k, v = blk.qkv(hh).chunk(3, -1)
    q = rope_pre(q.view(P, blk.h, blk.hd), tt, tw)
    k = rope_pre(k.view(P, blk.h, blk.hd), tt, tw)
    v = v.view(P, blk.h, blk.hd)
    shp = lambda t: t.view(nb, g, blk.h, blk.hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(shp(q), shp(k), shp(v))
    o = o.permute(0, 2, 1, 3).reshape(P, d)
    xp = xp + blk.proj(o)
    return xp + blk.mlp(blk.n2(xp))


def encode_perm(model, Bx, tok_mask, dtype=torch.bfloat16):
    vis = ~tok_mask
    vis_idx = vis.nonzero(as_tuple=True)[0]
    xv = model._emb(Bx, vis_idx)
    T = xv.shape[0]
    at = rope_angles(Bx["t_phys"][vis], model.d // model.heads, *model.lam_t)
    aw = rope_angles(Bx["wire_pos"][vis], model.d // model.heads, *model.lam_w)
    sched = model._sched(Bx["plane_id"][vis], Bx["t_phys"][vis], Bx["wire_pos"][vis])
    orders = [o for o, g, uw in sched]; gs = [g for o, g, uw in sched]
    uws = [uw for o, g, uw in sched]
    plan, last_inv = build_plan(orders, gs, T, xv.device)
    xp = xv[plan[0][0]]
    # angle tables per padded layout, built once each (4 distinct layouts here)
    cache = {}
    for i, (blk, (src, nb, g)) in enumerate(zip(model.enc, plan)):
        if i:
            xp = xp[src]
        key = (int(orders[i].data_ptr()), g)
        if key not in cache:
            pos = torch.arange(nb * g, device=xv.device).clamp(max=T - 1)
            tk = orders[i][pos]
            cache[key] = (rope_tables(at[tk], xp.dtype), rope_tables(aw[tk], xp.dtype))
        tt, tw = cache[key]
        xp = _self_perm(blk, xp, tt, tw if uws[i] else None, nb, g)
    return xp[:0].new_empty(0) if False else xp[last_inv], vis_idx, at, aw, vis


print("\n== permuted-residual encoder ==", flush=True)
try:
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        vis = ~m
        ref = model.encode_ref = None
        # reference: shipped encoder over visible tokens only
        vis_idx = vis.nonzero(as_tuple=True)[0]
        xv = model._emb(B, vis_idx)
        at = rope_angles(B["t_phys"], model.d // model.heads, *model.lam_t)
        aw = rope_angles(B["wire_pos"], model.d // model.heads, *model.lam_w)
        atv, awv = at[vis], aw[vis]
        sched = model._sched(B["plane_id"][vis], B["t_phys"][vis], B["wire_pos"][vis])
        for blk, (o, g, uw) in zip(model.enc, sched):
            xv = S._self(blk, xv, atv, awv if uw else None, o, g, None)
        ref = xv.float().clone()
        got, *_ = encode_perm(model, B, m)
        got = got.float()
    d = (ref - got).abs()
    R["perm_encoder"] = dict(max_abs=float(d.max()),
                             max_rel=float(d.max() / ref.abs().max()),
                             mean_abs=float(d.mean()))
    print(f"  max|d| vs shipped encoder = {R['perm_encoder']['max_abs']:.3e} "
          f"(rel {R['perm_encoder']['max_rel']:.3e})")

    def enc_ship():
        with torch.autocast("cuda", torch.bfloat16):
            vis_ = ~m
            vi = vis_.nonzero(as_tuple=True)[0]
            x = model._emb(B, vi)
            a1 = rope_angles(B["t_phys"], model.d // model.heads, *model.lam_t)[vis_]
            a2 = rope_angles(B["wire_pos"], model.d // model.heads, *model.lam_w)[vis_]
            sc = model._sched(B["plane_id"][vis_], B["t_phys"][vis_], B["wire_pos"][vis_])
            for blk, (o, g, uw) in zip(model.enc, sc):
                x = S._self(blk, x, a1, a2 if uw else None, o, g, None)
            return x

    def enc_perm():
        with torch.autocast("cuda", torch.bfloat16):
            return encode_perm(model, B, m)[0]

    for tag, fn in (("shipped encoder", enc_ship), ("permuted encoder", enc_perm)):
        def fwd():
            with torch.no_grad():
                fn()
        def fb():
            opt.zero_grad(set_to_none=True)
            fn().float().pow(2).mean().backward()
        tf = timeit(fwd, warmup=3, iters=A.iters)
        tb = timeit(fb, warmup=3, iters=A.iters)
        with peak_mem() as pm:
            fb()
        R.setdefault("encoder", {})[tag] = dict(fwd_ms=tf["ms_med"], fb_ms=tb["ms_med"],
                                                peak_MiB=pm["peak_alloc_MiB"])
        print(f"  {tag:20s} fwd {tf['ms_med']:7.2f} ms   fwd+bwd {tb['ms_med']:7.2f} ms   "
              f"peak {pm['peak_alloc_MiB']:8.0f} MiB", flush=True)
        opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    e = R["encoder"]
    R["encoder"]["speedup_fb"] = e["shipped encoder"]["fb_ms"] / e["permuted encoder"]["fb_ms"]
    print(f"  -> {R['encoder']['speedup_fb']:.3f}x on the encoder alone")
except Exception as e:
    R["perm_encoder_error"] = traceback.format_exc()[-2000:]
    print("  FAILED\n", traceback.format_exc()[-2000:])

emit("p15_traffic", R)
