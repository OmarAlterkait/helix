"""P16 — the residual stream is fp32, inside a bf16 autocast run.

`_emb` ends with `x + band_emb(band) + plane_emb(plane)`. nn.Embedding is not on
autocast's bf16 list, so it returns fp32 weights, and bf16 + fp32 -> fp32. From
there every block does `x = x + proj(o)` with x fp32 and proj bf16, so the
residual stream stays fp32 for all 16 blocks -- twice the bytes of the largest
tensor in the model, plus a cast at every residual add.

This dumps the dtypes to establish the fact, then measures what carrying the
stream in bf16 costs and saves, on top of the permuted-residual encoder.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time, traceback
import numpy as np, torch, torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, peak_mem, timeit, to_device
from kit import head_bmm
import helix.model.serial as S
from helix.model.fm import rope_angles
from helix.model.loss import losses_cat

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=8)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")
R = {}

evs, _ = load_events(5)
evs_d = [to_device({k: v for k, v in e.items() if torch.is_tensor(v)}, dev) for e in evs]
for e in evs_d:
    e["n_cells"] = e["plane_id"].shape[0]
model = build(device=dev)
masks = []
for i, e in enumerate(evs_d):
    g = torch.Generator(device=dev); g.manual_seed(100 + i)
    masks.append(model.make_mask(e, mode="random", gen=g))
B, m = evs_d[0], masks[0]
opt = torch.optim.AdamW(model.param_groups(0.0, weight_decay=0.05), betas=(0.9, 0.95))

# ------------------------------------------------------- 1. establish the fact
print("== dtypes through one encoder block, under bf16 autocast ==", flush=True)
dt = {}
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    vis = ~m
    vi = vis.nonzero(as_tuple=True)[0]
    x0 = model.embed(torch.cat([B["inp"][vi], B["occ"][vi]], -1))
    dt["embed(Linear) out"] = str(x0.dtype)
    g_, b_ = model.film(B["band_id"][vi], B["plane_id"][vi], B["wirefeat"][vi])
    dt["film gamma"] = str(g_.dtype)
    x1 = g_ * x0 + b_
    dt["after FiLM"] = str(x1.dtype)
    be = model.band_emb(B["band_id"][vi])
    dt["band_emb out"] = str(be.dtype)
    x2 = x1 + be + model.plane_emb(B["plane_id"][vi])
    dt["_emb() out (residual stream)"] = str(x2.dtype)
    blk = model.enc[0]
    hh = blk.n1(x2); dt["LayerNorm out"] = str(hh.dtype)
    q = blk.qkv(hh); dt["qkv(Linear) out"] = str(q.dtype)
    at = rope_angles(B["t_phys"][vis], model.d // model.heads, *model.lam_t)
    dt["rope_angles"] = str(at.dtype)
    qq = S.apply_rope(q.chunk(3, -1)[0].view(-1, blk.h, blk.hd), at, None)
    dt["after apply_rope (q,k)"] = str(qq.dtype)
    o = F.scaled_dot_product_attention(qq.transpose(0, 1)[None], qq.transpose(0, 1)[None],
                                       qq.transpose(0, 1)[None])
    dt["SDPA out"] = str(o.dtype)
    ao = blk.proj(o[0].transpose(0, 1).reshape(-1, model.d)); dt["proj out"] = str(ao.dtype)
    dt["x + proj (residual)"] = str((x2 + ao).dtype)
R["dtypes"] = dt
for k, v in dt.items():
    flag = "  <-- fp32" if v == "torch.float32" else ""
    print(f"  {k:32s} {v}{flag}")
N = int((~m).sum())
R["stream_bytes"] = dict(visible_tokens=N, d=model.d,
                         fp32_MiB=N * model.d * 4 / 2**20, bf16_MiB=N * model.d * 2 / 2**20)
print(f"  residual stream: {N} x {model.d} = "
      f"{R['stream_bytes']['fp32_MiB']:.1f} MiB fp32 vs "
      f"{R['stream_bytes']['bf16_MiB']:.1f} MiB bf16, x16 blocks")

# ------------------------------------------- 2. permuted encoder, both dtypes
def build_plan(orders, gs, T, device):
    plan, prev_inv = [], None
    ar = torch.arange(T, device=device)
    for o, g in zip(orders, gs):
        npad = ((T + g - 1) // g) * g
        tok = o[torch.arange(npad, device=device).clamp(max=T - 1)]
        plan.append((tok if prev_inv is None else prev_inv[tok], npad // g, g))
        inv = torch.empty(T, dtype=torch.long, device=device); inv[o] = ar
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
    P, d = xp.shape
    q, k, v = blk.qkv(blk.n1(xp)).chunk(3, -1)
    q = rope_pre(q.view(P, blk.h, blk.hd), tt, tw)
    k = rope_pre(k.view(P, blk.h, blk.hd), tt, tw)
    shp = lambda t: t.view(nb, g, blk.h, blk.hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(shp(q), shp(k), shp(v.view(P, blk.h, blk.hd)))
    o = o.permute(0, 2, 1, 3).reshape(P, d)
    xp = xp + blk.proj(o)
    return xp + blk.mlp(blk.n2(xp))


def encode_perm(model, Bx, tok_mask, stream_bf16=False):
    vis = ~tok_mask
    vis_idx = vis.nonzero(as_tuple=True)[0]
    xv = model._emb(Bx, vis_idx)
    if stream_bf16:
        xv = xv.to(torch.bfloat16)
    T = xv.shape[0]
    at = rope_angles(Bx["t_phys"][vis], model.d // model.heads, *model.lam_t)
    aw = rope_angles(Bx["wire_pos"][vis], model.d // model.heads, *model.lam_w)
    sched = model._sched(Bx["plane_id"][vis], Bx["t_phys"][vis], Bx["wire_pos"][vis])
    orders = [o for o, g, uw in sched]
    plan, last_inv = build_plan(orders, [g for o, g, uw in sched], T, xv.device)
    xp = xv[plan[0][0]]
    cache = {}
    for i, (blk, (src, nb, g)) in enumerate(zip(model.enc, plan)):
        if i:
            xp = xp[src]
        key = (int(orders[i].data_ptr()), g)
        if key not in cache:
            tk = orders[i][torch.arange(nb * g, device=xv.device).clamp(max=T - 1)]
            cache[key] = (rope_tables(at[tk], xp.dtype), rope_tables(aw[tk], xp.dtype))
        tt, tw = cache[key]
        xp = _self_perm(blk, xp, tt, tw if sched[i][2] else None, nb, g)
    return xp[last_inv], vis_idx, at, aw, vis


def ffeat(model, Bx, tok_mask, perm=False, stream_bf16=False):
    if not perm:
        return model.forward_feat(Bx, tok_mask)
    Nn = Bx["inp"].shape[0]
    xv, vis_idx, atv_all, awv_all, vis = encode_perm(model, Bx, tok_mask, stream_bf16)
    at = rope_angles(Bx["t_phys"], model.d // model.heads, *model.lam_t)
    aw = rope_angles(Bx["wire_pos"], model.d // model.heads, *model.lam_w)
    mask_idx = tok_mask.nonzero(as_tuple=True)[0]
    qm = model.mask_tok.expand(mask_idx.numel(), model.d)
    g_, b_ = model.film(Bx["band_id"][tok_mask], Bx["plane_id"][tok_mask],
                        Bx["wirefeat"][tok_mask])
    qm = g_ * qm + b_
    qm = (qm + model.band_emb(Bx["band_id"][tok_mask])
          + model.plane_emb(Bx["plane_id"][tok_mask])).to(xv.dtype)
    atm, awm = at[tok_mask], aw[tok_mask]
    atv, awv = at[vis], aw[vis]
    oq = torch.argsort(Bx["t_phys"][tok_mask].double())
    okv = torch.argsort(Bx["t_phys"][vis].double())
    for blk in model.dec:
        qm = S._cross(blk, qm, xv, atm, awm, atv, awv, oq, okv, model.gd, None)
    x = torch.zeros(Nn, model.d, dtype=xv.dtype, device=xv.device)
    x = x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm)
    return model.dec_norm(x)


def head_base(model, feat, Bx, mm):
    occ = model.occ_head(feat) * model.readout_mult
    val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
    return losses_cat(occ, val, Bx, mm, model.bin_edges, vis_w=0.0)


print("\n== full step ==", flush=True)
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    ref = ffeat(model, B, m).float().clone()
VARIANTS = [
    ("A  shipped",                      dict(perm=False, stream_bf16=False), head_base),
    ("B  permuted residual",            dict(perm=True,  stream_bf16=False), head_base),
    ("C  B + bf16 stream",              dict(perm=True,  stream_bf16=True),  head_base),
    ("D  C + sparse head",              dict(perm=True,  stream_bf16=True),  head_bmm),
]
R["variants"] = {}
for name, kw, head in VARIANTS:
    try:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            got = ffeat(model, B, m, **kw).float()
        dev_abs = float((ref - got).abs().max())
        def st():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                f = ffeat(model, B, m, **kw)
                b_, v_ = head(model, f, B, m)
            (b_ + v_).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        t = timeit(st, warmup=3, iters=A.iters)
        with peak_mem() as pm:
            st()
        c0 = time.perf_counter()
        for _ in range(A.iters):
            st()
        cpu = (time.perf_counter() - c0) / A.iters * 1e3
        torch.cuda.synchronize()
        R["variants"][name] = dict(ms=t["ms_med"], peak_MiB=pm["peak_alloc_MiB"],
                                   cpu_ms=cpu, max_abs_dev=dev_abs)
        print(f"  {name:24s} {t['ms_med']:8.2f} ms  peak {pm['peak_alloc_MiB']:8.0f} MiB  "
              f"max|d| {dev_abs:.3e}", flush=True)
    except Exception as e:
        R["variants"][name] = dict(error=repr(e), tb=traceback.format_exc()[-800:])
        print(f"  {name:24s} FAILED {e}")
    opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()

a = R["variants"].get("A  shipped", {})
for n, r in R["variants"].items():
    if "ms" in r and "ms" in a:
        r["speedup"] = a["ms"] / r["ms"]
        r["mem"] = r["peak_MiB"] / a["peak_MiB"]
        print(f"  {n:24s} x{r['speedup']:.3f}  mem {r['mem']:.2f}x")

# ------------------------------- 3. busy fraction of the best variant
print("\n== GPU busy fraction, best variant ==", flush=True)
import threading
def busy(tag, st):
    for _ in range(3):
        st()
    torch.cuda.synchronize()
    stop = threading.Event(); sm = []
    def s():
        while not stop.is_set():
            sm.append(torch.cuda.utilization(0)); time.sleep(0.01)
    th = threading.Thread(target=s, daemon=True); th.start()
    t0 = time.perf_counter(); n = 0
    while time.perf_counter() - t0 < 8:
        st(); n += 1
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    stop.set(); th.join()
    with profile(activities=[ProfilerActivity.CUDA], acc_events=True) as pr:
        torch.cuda.synchronize(); p0 = time.perf_counter()
        for _ in range(A.iters):
            st()
        torch.cuda.synchronize(); pw = (time.perf_counter() - p0) * 1e3
    ker = sum(e.self_device_time_total for e in pr.key_averages()) / 1e3
    R.setdefault("busy", {})[tag] = dict(ms=wall / n * 1e3, util=float(np.mean(sm)),
                                         busy_frac=ker / pw)
    print(f"  {tag:24s} {wall/n*1e3:8.1f} ms  util {np.mean(sm):5.1f}%  "
          f"kernel/wall {100*ker/pw:5.1f}%", flush=True)
    opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()

for name, kw, head in (VARIANTS[0], VARIANTS[3]):
    def st():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            f = ffeat(model, B, m, **kw)
            b_, v_ = head(model, f, B, m)
        (b_ + v_).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    busy(name, st)

emit("p16_dtype", R)
