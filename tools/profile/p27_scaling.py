"""P17 — the two things p15/p16 pointed at, done properly.

(a) `apply_rope`, isolated. The shipped form runs ~14 kernels per call on
    strided views: cos, sin, two repeat_interleave, a negate, a stack over
    v[...,1::2]/v[...,0::2], two muls, an add, per HALF, then a cat. Rewritten
    over a (T, h, 32, 2) view the same arithmetic is 7 kernels, fully
    contiguous, with no repeat_interleave and no cat -- and bit-exact.

(b) The DECODER carries the same permutation waste as the encoder, times four:
    all 4 CrossBlocks call grouped_cross with the SAME oq and okv, so the query
    set is gathered 4 times, the key/value set 8 times, and the output scattered
    4 times. Permute once, run all four blocks in permuted space, permute back.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, threading, time, traceback
import numpy as np, torch, torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, pack, peak_mem, timeit, to_device
from kit import head_bmm
import helix.model.serial as S
from helix.model.fm import apply_rope, rope_angles
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
HD = model.d // model.heads


# ============================================================ (a) fused RoPE
def rope_pairs(ang_t, ang_w, dtype):
    """One (T, 1, hd/2) cos and sin covering BOTH halves: time frequencies for
    the first hd/4 pairs, wire for the rest. No repeat_interleave, no cat."""
    if ang_w is None:
        n = ang_t.shape[1]
        c = torch.cat([torch.cos(ang_t), torch.ones_like(ang_t)], -1)
        s = torch.cat([torch.sin(ang_t), torch.zeros_like(ang_t)], -1)
    else:
        c = torch.cat([torch.cos(ang_t), torch.cos(ang_w)], -1)
        s = torch.cat([torch.sin(ang_t), torch.sin(ang_w)], -1)
    return c[:, None, :].to(dtype).contiguous(), s[:, None, :].to(dtype).contiguous()


def rope_fused(x, c, s):
    T, h, hd = x.shape
    v = x.view(T, h, hd // 2, 2)
    a, b = v[..., 0], v[..., 1]
    return torch.stack([a * c - b * s, b * c + a * s], -1).view(T, h, hd)


print("== (a) apply_rope, isolated, on the real shapes ==", flush=True)
R["rope"] = {}
for tag, T in (("encoder (7,625 tok)", int((~m).sum())), ("decoder q (22,919 tok)", int(m.sum()))):
    at = rope_angles(torch.rand(T, device=dev) * 4000, HD, *model.lam_t)
    aw = rope_angles(torch.rand(T, device=dev) * 2000, HD, *model.lam_w)
    for dt in (torch.float32, torch.bfloat16):
        x = torch.randn(T, model.heads, HD, device=dev, dtype=dt)
        c, s = rope_pairs(at, aw, dt)
        ref = apply_rope(x, at, aw)
        got = rope_fused(x, c, s)
        dv = float((ref.float() - got.float()).abs().max())
        t0 = timeit(lambda: apply_rope(x, at, aw), warmup=5, iters=30)
        t1 = timeit(lambda: rope_fused(x, c, s), warmup=5, iters=30)
        try:
            cf = torch.compile(rope_fused, dynamic=True)
            cf(x, c, s)
            t2 = timeit(lambda: cf(x, c, s), warmup=5, iters=30)["ms_med"]
        except Exception:
            t2 = float("nan")
        # bandwidth-ideal: read x + write out (+ tables), at measured HBM speed
        ideal_ms = 2 * x.numel() * x.element_size() / (1285 * 2**30) * 1e3
        key = f"{tag} {str(dt).split('.')[-1]}"
        R["rope"][key] = dict(shipped_ms=t0["ms_med"], fused_ms=t1["ms_med"],
                              compiled_ms=t2, ideal_ms=ideal_ms, max_abs_dev=dv,
                              shipped_pct_of_ideal=100 * ideal_ms / t0["ms_med"],
                              fused_pct_of_ideal=100 * ideal_ms / t1["ms_med"])
        print(f"  {key:34s} shipped {t0['ms_med']:6.3f}  fused {t1['ms_med']:6.3f}  "
              f"compiled {t2:6.3f}  ideal {ideal_ms:6.3f} ms  "
              f"(shipped {100*ideal_ms/t0['ms_med']:4.1f}% of roofline, "
              f"fused {100*ideal_ms/t1['ms_med']:4.1f}%)  max|d|={dv:.1e}", flush=True)
        torch._dynamo.reset()
        del x, c, s, ref, got
        gc.collect(); torch.cuda.empty_cache()


# ================================== (b) permuted encoder AND permuted decoder
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


def _self_perm(blk, xp, c, s, nb, g):
    P, d = xp.shape
    q, k, v = blk.qkv(blk.n1(xp)).chunk(3, -1)
    q = rope_fused(q.view(P, blk.h, blk.hd), c, s)
    k = rope_fused(k.view(P, blk.h, blk.hd), c, s)
    shp = lambda t: t.view(nb, g, blk.h, blk.hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(shp(q), shp(k), shp(v.view(P, blk.h, blk.hd)))
    xp = xp + blk.proj(o.permute(0, 2, 1, 3).reshape(P, d))
    return xp + blk.mlp(blk.n2(xp))


def _cross_perm(blk, qp, kvp, qc, qs, kc, ks, nb, gq, gk):
    """CrossBlock over query/key sets that are ALREADY in oq/okv order and
    padded. All four decoder blocks share oq and okv, so this is done once."""
    Pq, d = qp.shape
    Pk = kvp.shape[0]
    hq = blk.nq(qp)
    qh = rope_fused(blk.q(hq).view(Pq, blk.h, blk.hd), qc, qs)
    k, v = blk.kv(blk.nk(kvp)).chunk(2, -1)
    kh = rope_fused(k.view(Pk, blk.h, blk.hd), kc, ks)
    o = F.scaled_dot_product_attention(
        qh.view(nb, gq, blk.h, blk.hd).permute(0, 2, 1, 3),
        kh.view(nb, gk, blk.h, blk.hd).permute(0, 2, 1, 3),
        v.view(Pk, blk.h, blk.hd).view(nb, gk, blk.h, blk.hd).permute(0, 2, 1, 3))
    qp = qp + blk.proj(o.permute(0, 2, 1, 3).reshape(Pq, d))
    return qp + blk.mlp(blk.n2(qp))


def _pad_idx(order, T, npad, device):
    return order[torch.arange(npad, device=device).clamp(max=T - 1)]


def ffeat_perm(model, Bx, tok_mask, stream_bf16=False, perm_dec=True):
    N = Bx["inp"].shape[0]
    vis = ~tok_mask
    vis_idx = vis.nonzero(as_tuple=True)[0]
    mask_idx = tok_mask.nonzero(as_tuple=True)[0]
    xv = model._emb(Bx, vis_idx)
    if stream_bf16:
        xv = xv.to(torch.bfloat16)
    T = xv.shape[0]
    atv = rope_angles(Bx["t_phys"][vis], HD, *model.lam_t)
    awv = rope_angles(Bx["wire_pos"][vis], HD, *model.lam_w)
    sched = model._sched(Bx["plane_id"][vis], Bx["t_phys"][vis], Bx["wire_pos"][vis])
    orders = [o for o, g, uw in sched]
    plan, last_inv = build_plan(orders, [g for o, g, uw in sched], T, xv.device)
    xp = xv[plan[0][0]]
    cache = {}
    for i, (blk, (src, nb, g)) in enumerate(zip(model.enc, plan)):
        if i:
            xp = xp[src]
        key = (int(orders[i].data_ptr()), g, sched[i][2])
        if key not in cache:
            tk = _pad_idx(orders[i], T, nb * g, xv.device)
            cache[key] = rope_pairs(atv[tk], awv[tk] if sched[i][2] else None, xp.dtype)
        c, s = cache[key]
        xp = _self_perm(blk, xp, c, s, nb, g)
    xv = xp[last_inv]

    # ---- decoder
    qm = model.mask_tok.expand(mask_idx.numel(), model.d)
    g_, b_ = model.film(Bx["band_id"][tok_mask], Bx["plane_id"][tok_mask],
                        Bx["wirefeat"][tok_mask])
    qm = ((g_ * qm + b_) + model.band_emb(Bx["band_id"][tok_mask])
          + model.plane_emb(Bx["plane_id"][tok_mask])).to(xv.dtype)
    atm = rope_angles(Bx["t_phys"][tok_mask], HD, *model.lam_t)
    awm = rope_angles(Bx["wire_pos"][tok_mask], HD, *model.lam_w)
    Tq, Tk = qm.shape[0], xv.shape[0]
    oq = torch.argsort(Bx["t_phys"][tok_mask].double())
    okv = torch.argsort(Bx["t_phys"][vis].double())
    if not perm_dec:
        for blk in model.dec:
            qm = S._cross(blk, qm, xv, atm, awm, atv, awv, oq, okv, model.gd, None)
    else:
        g = model.gd
        nb = (max(Tq, Tk) + g - 1) // g
        gq = (Tq + nb - 1) // nb
        gk = (Tk + nb - 1) // nb
        iq = _pad_idx(oq, Tq, nb * gq, xv.device)
        ik = _pad_idx(okv, Tk, nb * gk, xv.device)
        qc, qs = rope_pairs(atm[iq], awm[iq], qm.dtype)
        kc, ks = rope_pairs(atv[ik], awv[ik], xv.dtype)
        qp = qm[iq]
        kvp = xv[ik]
        for blk in model.dec:
            qp = _cross_perm(blk, qp, kvp, qc, qs, kc, ks, nb, gq, gk)
        inv = torch.empty(Tq, dtype=torch.long, device=xv.device)
        inv[oq] = torch.arange(Tq, device=xv.device)
        qm = qp[inv]
    x = torch.zeros(N, model.d, dtype=xv.dtype, device=xv.device)
    x = x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm)
    return model.dec_norm(x)


def head_base(model, feat, Bx, mm):
    occ = model.occ_head(feat) * model.readout_mult
    val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
    return losses_cat(occ, val, Bx, mm, model.bin_edges, vis_w=0.0)



# ---------------------------------------------------------------- p18 additions
def head_base(model, feat, Bx, mm):
    occ = model.occ_head(feat) * model.readout_mult
    val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
    return losses_cat(occ, val, Bx, mm, model.bin_edges, vis_w=0.0)


def ffeat_masked(model, Bx, tok_mask, stream_bf16=False):
    """ffeat_perm, but returning ONLY the masked rows.

    With vis_w == 0 the objective touches no visible row: `losses_cat` masks the
    BCE by `tok_mask[:, None]` and the value term by `... & mrow`. So the final
    zeros(N, d) + two index_copy + LayerNorm over all N, and both heads over all
    N, spend 25% of their work on rows that are multiplied by zero.
    """
    vis = ~tok_mask
    vis_idx = vis.nonzero(as_tuple=True)[0]
    mask_idx = tok_mask.nonzero(as_tuple=True)[0]
    xv = model._emb(Bx, vis_idx)
    if stream_bf16:
        xv = xv.to(torch.bfloat16)
    T = xv.shape[0]
    atv = rope_angles(Bx["t_phys"][vis], HD, *model.lam_t)
    awv = rope_angles(Bx["wire_pos"][vis], HD, *model.lam_w)
    sched = model._sched(Bx["plane_id"][vis], Bx["t_phys"][vis], Bx["wire_pos"][vis])
    orders = [o for o, g, uw in sched]
    plan, last_inv = build_plan(orders, [g for o, g, uw in sched], T, xv.device)
    xp = xv[plan[0][0]]
    cache = {}
    for i, (blk, (src, nb, g)) in enumerate(zip(model.enc, plan)):
        if i:
            xp = xp[src]
        key = (int(orders[i].data_ptr()), g, sched[i][2])
        if key not in cache:
            tk = _pad_idx(orders[i], T, nb * g, xv.device)
            cache[key] = rope_pairs(atv[tk], awv[tk] if sched[i][2] else None, xp.dtype)
        c, s = cache[key]
        xp = _self_perm(blk, xp, c, s, nb, g)
    xv = xp[last_inv]
    qm = model.mask_tok.expand(mask_idx.numel(), model.d)
    g_, b_ = model.film(Bx["band_id"][tok_mask], Bx["plane_id"][tok_mask],
                        Bx["wirefeat"][tok_mask])
    qm = ((g_ * qm + b_) + model.band_emb(Bx["band_id"][tok_mask])
          + model.plane_emb(Bx["plane_id"][tok_mask])).to(xv.dtype)
    atm = rope_angles(Bx["t_phys"][tok_mask], HD, *model.lam_t)
    awm = rope_angles(Bx["wire_pos"][tok_mask], HD, *model.lam_w)
    Tq, Tk = qm.shape[0], xv.shape[0]
    oq = torch.argsort(Bx["t_phys"][tok_mask].double())
    okv = torch.argsort(Bx["t_phys"][vis].double())
    g = model.gd
    nb = (max(Tq, Tk) + g - 1) // g
    gq = (Tq + nb - 1) // nb; gk = (Tk + nb - 1) // nb
    iq = _pad_idx(oq, Tq, nb * gq, xv.device); ik = _pad_idx(okv, Tk, nb * gk, xv.device)
    qc, qs = rope_pairs(atm[iq], awm[iq], qm.dtype)
    kc, ks = rope_pairs(atv[ik], awv[ik], xv.dtype)
    qp = qm[iq]; kvp = xv[ik]
    for blk in model.dec:
        qp = _cross_perm(blk, qp, kvp, qc, qs, kc, ks, nb, gq, gk)
    inv = torch.empty(Tq, dtype=torch.long, device=xv.device)
    inv[oq] = torch.arange(Tq, device=xv.device)
    return model.dec_norm(qp[inv]), mask_idx


def head_masked(model, fm_, Bx, tok_mask, mask_idx):
    """Both heads and both loss terms, over masked rows only. Same sums."""
    NS, KB = model.n_slot, model.n_bins
    occ = model.occ_head(fm_) * model.readout_mult
    valid_m = Bx["valid"][mask_idx]
    occt_m = Bx["occ"][mask_idx]
    tgt_m = Bx["tgt"][mask_idx]
    bce_e = F.binary_cross_entropy_with_logits(occ, occt_m, reduction="none")
    bce = (bce_e * valid_m).sum() / valid_m.sum().clamp(min=1)
    act = occt_m.bool() & valid_m
    ci, si = act.nonzero(as_tuple=True)
    P = ci.numel()
    from helix.model.loss import bucketize_bins
    binid = bucketize_bins(tgt_m[ci, si][:, None], Bx["band_id"][mask_idx][ci],
                           model.bin_edges, KB).squeeze(1)
    o = torch.argsort(si)
    ci_s, si_s, bin_s = ci[o], si[o], binid[o]
    cnt = torch.bincount(si_s, minlength=NS); mx = int(cnt.max())
    st_ = torch.cat([cnt.new_zeros(1), cnt.cumsum(0)[:-1]])
    col = torch.arange(mx, device=fm_.device)[None, :]
    okm = col < cnt[:, None]
    flat = (st_[:, None] + col).clamp(max=max(P - 1, 0))
    f = fm_[ci_s[flat]]
    W = model.val_head.weight.view(NS, KB, -1); bv = model.val_head.bias.view(NS, KB)
    lg = torch.baddbmm(bv[:, None, :].to(f.dtype), f, W.transpose(1, 2).to(f.dtype)) * model.readout_mult
    ce = F.cross_entropy(lg.reshape(-1, KB).float(), bin_s[flat].reshape(-1), reduction="none")
    return bce, (ce.view(NS, mx) * okm).sum() / float(P)



# ============================= scaling: which axis buys capacity most cheaply
from common import synth_from, n_params
from torch.utils.checkpoint import checkpoint

NCELLS = 32064                      # the corpus median token count
src = {k: v for k, v in evs[0].items() if torch.is_tensor(v)}
Bs = to_device(synth_from(src, NCELLS), dev); Bs["n_cells"] = NCELLS
print(f"fixed batch: {NCELLS} cells (corpus median)", flush=True)


def measure(ov, mask_ratio=0.75, ckpt=False, iters=5):
    """Build a model at these overrides and time one optimised step."""
    mo = build(ov, device=dev)
    p = n_params(mo)
    o = torch.optim.AdamW(mo.param_groups(0.0, weight_decay=0.05), betas=(0.9, 0.95))
    g = torch.Generator(device=dev); g.manual_seed(0)
    mk = mo.make_mask(Bs, mode="random", ratio=mask_ratio, gen=g)
    global model
    keep, model = model, mo
    try:
        def st():
            o.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                f, mi = ffeat_masked(mo, Bs, mk)
                b_, v_ = head_masked(mo, f, Bs, mk, mi)
            (b_ + v_).backward()
            torch.nn.utils.clip_grad_norm_(mo.parameters(), 1.0)
            o.step()
        st(); st()
        state = torch.cuda.memory_allocated() / 2**20      # params+grads+adam resident
        t = timeit(st, warmup=2, iters=iters)
        with peak_mem() as pm:
            st()
        return dict(params_M=p/1e6, ms=t["ms_med"], peak_MiB=pm["peak_alloc_MiB"],
                    state_MiB=state, act_MiB=pm["peak_alloc_MiB"] - state,
                    masked=float(mk.float().mean()))
    finally:
        model = keep
        o.zero_grad(set_to_none=True); del mo, o
        gc.collect(); torch.cuda.empty_cache()


BASE = dict(d=512, blocks=12, dec_blocks=4, heads=8)
AXES = {
    "width d (heads = d/64)": [
        ("d=384",  dict(d=384,  heads=6)),
        ("d=512",  dict()),
        ("d=768",  dict(d=768,  heads=12)),
        ("d=1024", dict(d=1024, heads=16)),
        ("d=1536", dict(d=1536, heads=24)),
        ("d=2048", dict(d=2048, heads=32)),
    ],
    "encoder depth": [
        ("enc=6",  dict(blocks=6)),  ("enc=12", dict()),
        ("enc=24", dict(blocks=24)), ("enc=36", dict(blocks=36)),
        ("enc=48", dict(blocks=48)),
    ],
    "decoder depth": [
        ("dec=2",  dict(dec_blocks=2)), ("dec=4", dict()),
        ("dec=8",  dict(dec_blocks=8)), ("dec=16", dict(dec_blocks=16)),
    ],
    "ffn_mult": [("F=2", dict(ffn_mult=2)), ("F=4", dict()), ("F=8", dict(ffn_mult=8))],
    "n_bins K": [("K=32", dict(n_bins=32)), ("K=64", dict(n_bins=64)),
                 ("K=128", dict()), ("K=256", dict(n_bins=256))],
    "attention group gp/gd": [
        ("gp/gd=512/1024",   dict(gp=512,  gd=1024)),
        ("gp/gd=1024/2048",  dict()),
        ("gp/gd=2048/4096",  dict(gp=2048, gd=4096)),
        ("gp/gd=4096/8192",  dict(gp=4096, gd=8192)),
    ],
}

R["base_cells"] = NCELLS
R["axes"] = {}
for axis, rows in AXES.items():
    print(f"\n== {axis} ==", flush=True)
    print(f"  {'config':20s} {'params':>9s} {'ms':>8s} {'peak':>9s} {'state':>8s} "
          f"{'activ':>9s} {'M/ms':>7s} {'M/GiB':>7s}")
    out = {}
    for tag, ov in rows:
        try:
            r = measure({**BASE, **ov})
            r["params_per_ms"] = r["params_M"] / r["ms"]
            r["params_per_GiB"] = r["params_M"] / (r["peak_MiB"] / 1024)
            out[tag] = r
            print(f"  {tag:20s} {r['params_M']:8.1f}M {r['ms']:8.1f} {r['peak_MiB']:9.0f} "
                  f"{r['state_MiB']:8.0f} {r['act_MiB']:9.0f} "
                  f"{r['params_per_ms']:7.2f} {r['params_per_GiB']:7.1f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            out[tag] = dict(oom=True); print(f"  {tag:20s} OOM", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            out[tag] = dict(error=repr(e)[:180]); print(f"  {tag:20s} ERR {e}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    R["axes"][axis] = out

# mask ratio moves work between encoder and decoder rather than adding it
print("\n== mask ratio (moves tokens between encoder and decoder) ==", flush=True)
out = {}
for mr in (0.50, 0.65, 0.75, 0.85, 0.90):
    r = measure(BASE, mask_ratio=mr)
    out[f"mask={mr}"] = r
    print(f"  mask={mr:.2f}  enc sees {int((1-r['masked'])*NCELLS):6d}  "
          f"dec queries {int(r['masked']*NCELLS):6d}   {r['ms']:7.1f} ms  "
          f"{r['peak_MiB']:8.0f} MiB", flush=True)
R["axes"]["mask ratio"] = out

emit("p27_scaling", R)
