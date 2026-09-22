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

evs, _ = load_events(24)
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



# ================== before/after as a function of event size, on one A100
import argparse as _ap
from common import synth_from

print("\n== before / after vs event size ==", flush=True)
C = lambda f: torch.compile(f, dynamic=True)
_rf, _sp, _cp, _hm = rope_fused, _self_perm, _cross_perm, head_masked


def before_step(Bx, mx):
    with torch.autocast("cuda", torch.bfloat16):
        f = model.forward_feat(Bx, mx)
        return head_base(model, f, Bx, mx)


def after_step(Bx, mx):
    """Everything compiled: blocks, RoPE, head. The head compile needs
    setuptools -- Inductor falls back to its C++ builder there -- so this is the
    'container fixed' number."""
    with torch.autocast("cuda", torch.bfloat16):
        fm_, mi = ffeat_masked(model, Bx, mx)
        return head_masked(model, fm_, Bx, mx, mi)


def after_nohead_step(Bx, mx):
    """Blocks and RoPE compiled, head left eager. This one runs in the image as
    it is today: those regions lower to pure Triton and never touch
    cpp_extension."""
    with torch.autocast("cuda", torch.bfloat16):
        fm_, mi = ffeat_masked(model, Bx, mx)
        return _hm(model, fm_, Bx, mx, mi)


def full(fn, Bx, mx):
    def st():
        opt.zero_grad(set_to_none=True)
        b_, v_ = fn(Bx, mx)
        (b_ + v_).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return st


def measure(fn, Bx, mx, iters=5):
    st = full(fn, Bx, mx)
    t = timeit(st, warmup=2, iters=iters)
    with peak_mem() as pm:
        st()
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        l = float(sum(fn(Bx, mx)))
    return dict(ms=t["ms_med"], peak_MiB=pm["peak_alloc_MiB"], loss=l)


def row(tag, Bx, mx, n_cells, n_coeff):
    r = dict(tag=tag, n_cells=int(n_cells), n_coeff=int(n_coeff))
    for name, fn in (("before", before_step), ("after", after_step),
                     ("after_nohead", after_nohead_step)):
        try:
            r[name] = measure(fn, Bx, mx)
        except torch.cuda.OutOfMemoryError:
            r[name] = dict(oom=True)
        except Exception as e:
            r[name] = dict(error=repr(e)[:300], tb=traceback.format_exc()[-600:])
            print(f"    !! {name} failed: {type(e).__name__}: {e}", flush=True)
            print(traceback.format_exc()[-700:], flush=True)
        gc.collect(); torch.cuda.empty_cache()
    if "ms" in r.get("before", {}) and "ms" in r.get("after", {}):
        r["speedup"] = r["before"]["ms"] / r["after"]["ms"]
        r["mem_ratio"] = r["after"]["peak_MiB"] / r["before"]["peak_MiB"]
        if "ms" in r.get("after_nohead", {}):
            r["speedup_nohead"] = r["before"]["ms"] / r["after_nohead"]["ms"]
            r["mem_ratio_nohead"] = r["after_nohead"]["peak_MiB"] / r["before"]["peak_MiB"]
        r["d_loss"] = r["after"]["loss"] - r["before"]["loss"]
        r["rel_loss"] = r["d_loss"] / r["before"]["loss"]
        print(f"  {tag:10s} cells={r['n_cells']:7d} coeff={r['n_coeff']:8d}  "
              f"before {r['before']['ms']:8.1f} ms /{r['before']['peak_MiB']:8.0f} MiB   "
              f"after {r['after']['ms']:8.1f} ms /{r['after']['peak_MiB']:8.0f} MiB   "
              f"x{r['speedup']:.3f}/{r.get('speedup_nohead', float('nan')):.3f}  "
              f"mem {r['mem_ratio']:.3f}  dloss {r['rel_loss']:+.1e}", flush=True)
    else:
        b = "OOM" if r.get("before", {}).get("oom") else f"{r.get('before',{}).get('ms',0):.0f}ms"
        a = "OOM" if r.get("after", {}).get("oom") else f"{r.get('after',{}).get('ms',0):.0f}ms"
        print(f"  {tag:10s} cells={r['n_cells']:7d} coeff={r['n_coeff']:8d}  "
              f"before {b:>9s}   after {a:>9s}", flush=True)
    return r


# compile the AFTER path once; dynamic=True absorbs the size sweep
globals()["_self_perm"] = C(_sp)
globals()["_cross_perm"] = C(_cp)
globals()["rope_fused"] = C(_rf)
globals()["head_masked"] = C(_hm)

R["real"] = []
print("  --- real events ---", flush=True)
NREAL = min(len(evs_d), 24)
order = sorted(range(NREAL), key=lambda i: int(evs[i]["cell"].shape[0]))
for i in order:
    R["real"].append(row(f"ev{i}", evs_d[i], masks[i],
                         evs_d[i]["n_cells"], evs[i]["cell"].shape[0]))

print("  --- synthetic scaling (a real event tiled) ---", flush=True)
src = {k: v for k, v in evs[0].items() if torch.is_tensor(v)}
coeff_per_cell = evs[0]["cell"].shape[0] / evs[0]["plane_id"].shape[0]
R["synth"] = []
for nc in (4000, 8000, 16000, 32000, 48000, 64000, 96000, 128000, 192000, 256000, 320000):
    try:
        Bs = to_device(synth_from(src, nc), dev); Bs["n_cells"] = nc
        g2 = torch.Generator(device=dev); g2.manual_seed(0)
        ms = model.make_mask(Bs, mode="random", gen=g2)
        R["synth"].append(row(f"N={nc}", Bs, ms, nc, Bs["cell"].shape[0]))
    except torch.cuda.OutOfMemoryError:
        R["synth"].append(dict(tag=f"N={nc}", n_cells=nc, both_oom=True))
        print(f"  N={nc:7d} OOM building batch", flush=True)
    Bs = ms = None
    gc.collect(); torch.cuda.empty_cache()

ok = [r for r in R["real"] + R["synth"] if "speedup" in r]
if ok:
    sp = [r["speedup"] for r in ok]; mr = [r["mem_ratio"] for r in ok]
    spn = [r["speedup_nohead"] for r in ok if "speedup_nohead" in r]
    R["summary_nohead"] = dict(n=len(spn), mean=float(np.mean(spn)) if spn else None,
                               lo=min(spn) if spn else None, hi=max(spn) if spn else None)
    R["summary"] = dict(n=len(ok), speedup_min=min(sp), speedup_max=max(sp),
                        speedup_mean=float(np.mean(sp)),
                        mem_min=min(mr), mem_max=max(mr), mem_mean=float(np.mean(mr)),
                        rel_loss_max=max(abs(r["rel_loss"]) for r in ok))
    if spn:
        print(f"\n  no-head-compile (works in today's image): {min(spn):.3f}-{max(spn):.3f} "
              f"(mean {np.mean(spn):.3f})")
    print(f"  speedup {min(sp):.3f}-{max(sp):.3f} (mean {np.mean(sp):.3f}); "
          f"memory {min(mr):.3f}-{max(mr):.3f} (mean {np.mean(mr):.3f}); "
          f"|rel dloss| <= {R['summary']['rel_loss_max']:.1e}")

# ------------------------------------------------------------------ figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for key, lbl, mk_ in (("real", "real events", "o"), ("synth", "synthetic", "s")):
        rows = [r for r in R[key] if "speedup" in r]
        if not rows:
            continue
        x = [r["n_coeff"] / 1e3 for r in rows]
        ax[0].plot(x, [r["before"]["ms"] for r in rows], mk_ + "-", label=f"before, {lbl}")
        ax[0].plot(x, [r["after"]["ms"] for r in rows], mk_ + "--", label=f"after, {lbl}")
        ax[1].plot(x, [r["before"]["peak_MiB"] / 1024 for r in rows], mk_ + "-", label=f"before, {lbl}")
        ax[1].plot(x, [r["after"]["peak_MiB"] / 1024 for r in rows], mk_ + "--", label=f"after, {lbl}")
        ax[2].plot(x, [r["speedup"] for r in rows], mk_ + "-", label=f"speedup, {lbl}")
        ax[2].plot(x, [1 / r["mem_ratio"] for r in rows], mk_ + "--", label=f"memory saving, {lbl}")
    ax[0].set_ylabel("step time (ms)"); ax[1].set_ylabel("peak GiB")
    ax[2].set_ylabel("before / after"); ax[2].axhline(1, color="k", lw=0.6)
    ax[1].axhline(40, color="r", ls=":", lw=1); ax[1].text(5, 40.6, "A100 40 GB", color="r", fontsize=8)
    for a_ in ax:
        a_.set_xlabel("coefficients per event (thousands)"); a_.set_xscale("log")
        a_.grid(alpha=.3); a_.legend(fontsize=7)
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    fig.suptitle("helix FM step: shipped vs optimised, one A100-40GB", fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.environ.get("PROF_OUT", "."), "p23_before_after.png")
    fig.savefig(out, dpi=130)
    R["figure"] = out
    print("  figure ->", out)
except Exception as e:
    print("  figure failed:", e)

globals()["_self_perm"], globals()["_cross_perm"] = _sp, _cp
globals()["rope_fused"], globals()["head_masked"] = _rf, _hm
emit("p23_before_after", R)
