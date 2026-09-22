"""P10 — a working event-aware packed forward, and what batching is worth once
it is legal.

MULTI_EVENT_BATCHING.md section 6 specifies per-event padding rather than a
dense block-diagonal mask. This implements it for both the grouped self
attention and the grouped cross decoder, checks that

  * K=1 is BIT-IDENTICAL to the shipped path (the acceptance criterion), and
  * a packed forward sliced per event reproduces each event's solo forward,

then measures ms/event and MiB/event against running the events one at a time.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, traceback
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, pack, peak_mem, timeit, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--kmax", type=int, default=8)
ap.add_argument("--iters", type=int, default=6)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")

import helix.model.serial as S
from helix.model.fm import rope_angles

L = lambda *a, **k: torch.tensor(*a, **k)


def _cum0(x):
    return torch.cat([x.new_zeros(1), x.cumsum(0)[:-1]])


def self_plan(counts, g):
    """Per-event padding to a multiple of g. At K=1 this reproduces the shipped
    npad = ceil(T/g)*g with the last token duplicated."""
    dev_ = counts.device
    p = ((counts + g - 1) // g) * g
    eid = torch.repeat_interleave(torch.arange(len(counts), device=dev_), p)
    local = torch.arange(int(p.sum()), device=dev_) - _cum0(p)[eid]
    n = counts[eid]
    src = _cum0(counts)[eid] + torch.minimum(local, n - 1)
    return src, local < n, int(p.sum()) // g


def cross_plan(cq, ck, g):
    """Per-event block count/sizes, exactly as the shipped code computes them
    for a single event; blocks padded to the batch-wide max so one SDPA call
    covers them. No block ever spans two events."""
    dev_ = cq.device
    nb = (torch.maximum(cq, ck) + g - 1) // g
    gq = (cq + nb - 1) // nb
    gk = (ck + nb - 1) // nb
    GQ, GK, NB = int(gq.max()), int(gk.max()), int(nb.sum())
    beid = torch.repeat_interleave(torch.arange(len(cq), device=dev_), nb)
    j = torch.arange(NB, device=dev_) - _cum0(nb)[beid]
    pq = torch.arange(GQ, device=dev_)[None, :]
    pk = torch.arange(GK, device=dev_)[None, :]
    gqe, gke = gq[beid][:, None], gk[beid][:, None]
    flatq = j[:, None] * gqe + torch.minimum(pq, gqe - 1)
    flatk = j[:, None] * gke + torch.minimum(pk, gke - 1)
    srcq = _cum0(cq)[beid][:, None] + torch.minimum(flatq, cq[beid][:, None] - 1)
    srck = _cum0(ck)[beid][:, None] + torch.minimum(flatk, ck[beid][:, None] - 1)
    keepq = (pq < gqe) & (flatq < cq[beid][:, None])
    return srcq, srck, keepq, NB, GQ, GK


_plans = {}


def uniform_attn_ev(q, k, v, order, g):
    T, h, hd = q.shape
    src, keep, nb = _plans["self"][g]
    idx = order[src]
    def grp(x):
        return x[idx].view(nb, g, h, hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(grp(q), grp(k.to(q.dtype)), grp(v.to(q.dtype)))
    o = o.permute(0, 2, 1, 3).reshape(-1, h, hd)
    out = o.new_empty(T, h, hd)
    out[idx[keep]] = o[keep]
    return out


def grouped_cross_ev(q, k, v, oq, ok, g):
    Tq, h, hd = q.shape
    srcq, srck, keepq, NB, GQ, GK = _plans["cross"][g]
    iq, ik = oq[srcq], ok[srck]
    o = F.scaled_dot_product_attention(
        q[iq].permute(0, 2, 1, 3), k.to(q.dtype)[ik].permute(0, 2, 1, 3),
        v.to(q.dtype)[ik].permute(0, 2, 1, 3))
    o = o.permute(0, 2, 1, 3)
    out = o.new_empty(Tq, h, hd)
    out[iq[keepq]] = o[keepq]
    return out


def sched_ev(model, plane, t, wire, batch_idx):
    """`SerialFMModel._sched`, made event-major: within-event key order first,
    then a stable sort by event, so no group spans an event boundary."""
    bp = plane.double()
    def ev(o):
        return o[torch.argsort(batch_idx[o], stable=True)]
    o_pt = ev(torch.argsort(bp * 1e9 + t.double()))
    o_pw = ev(torch.argsort(bp * 1e13 + wire.double() * 1e6 + t.double()))
    o_t = ev(torch.argsort(t.double()))
    o_ts = ev(torch.roll(torch.argsort(t.double()), model.gd // 2))
    dw = not model.rope_split
    cell = [(o_pt, model.gp, True), (o_t, model.gd, dw),
            (o_pw, model.gp, True), (o_ts, model.gd, dw)]
    return (cell * ((len(model.enc) + 3) // 4))[:len(model.enc)]


def forward_feat_ev(model, B, tok_mask, offset):
    """serial.SerialFMModel.forward_feat with per-event grouping."""
    N = B["inp"].shape[0]
    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    bidx = torch.repeat_interleave(torch.arange(len(counts), device=offset.device), counts)
    at = rope_angles(B["t_phys"], model.d // model.heads, *model.lam_t)
    aw = rope_angles(B["wire_pos"], model.d // model.heads, *model.lam_w)
    vis = ~tok_mask
    vis_idx = vis.nonzero(as_tuple=True)[0]
    mask_idx = tok_mask.nonzero(as_tuple=True)[0]
    xv = model._emb(B, vis_idx); atv, awv = at[vis], aw[vis]
    bv = bidx[vis]
    cq = torch.bincount(bidx[tok_mask], minlength=len(counts))
    ck = torch.bincount(bv, minlength=len(counts))
    _plans["self"] = {g: self_plan(ck, g) for g in {model.gp, model.gd}}
    _plans["cross"] = {model.gd: cross_plan(cq, ck, model.gd)}
    sched = sched_ev(model, B["plane_id"][vis], B["t_phys"][vis], B["wire_pos"][vis], bv)
    c = model._cond(B) if model.cond == "adaln" else None
    cv = c[vis] if c is not None else None
    for blk, (o, g, uw) in zip(model.enc, sched):
        xv = S._self(blk, xv, atv, awv if uw else None, o, g, cv)
    qm = model.mask_tok.expand(mask_idx.numel(), model.d)
    if c is not None:
        qm = qm.to(xv.dtype)
    else:
        if model.film is not None:
            g_, b_ = model.film(B["band_id"][tok_mask], B["plane_id"][tok_mask],
                                B["wirefeat"][tok_mask])
            qm = g_ * qm + b_
        qm = (qm + model.band_emb(B["band_id"][tok_mask])
              + model.plane_emb(B["plane_id"][tok_mask])).to(xv.dtype)
    atm, awm = at[tok_mask], aw[tok_mask]
    bm = bidx[tok_mask]
    oq = torch.argsort(B["t_phys"][tok_mask].double())
    oq = oq[torch.argsort(bm[oq], stable=True)]
    okv = torch.argsort(B["t_phys"][vis].double())
    okv = okv[torch.argsort(bv[okv], stable=True)]
    cm = c[tok_mask] if c is not None else None
    for blk in model.dec:
        qm = S._cross(blk, qm, xv, atm, awm, atv, awv, oq, okv, model.gd, cm)
    x = torch.zeros(N, model.d, dtype=xv.dtype, device=xv.device)
    x = x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm)
    return model.dec_norm(x)


# ------------------------------------------------------------------- checks
evs, _ = load_events(A.kmax + 1)
evs_d = [to_device({k: v for k, v in e.items() if torch.is_tensor(v)}, dev) for e in evs]
for e in evs_d:
    e["n_cells"] = e["plane_id"].shape[0]
model = build(device=dev)
R = {"n_cells": [int(e["n_cells"]) for e in evs_d]}
print("n_cells", R["n_cells"])

masks = []
for i, e in enumerate(evs_d):
    g = torch.Generator(device=dev); g.manual_seed(100 + i)
    masks.append(model.make_mask(e, mode="random", gen=g))

_orig = (S.uniform_attn, S.grouped_cross)

def solo(i):
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        return model.forward_feat(evs_d[i], masks[i]).float().clone()

refs = [solo(i) for i in range(len(evs_d))]

S.uniform_attn, S.grouped_cross = uniform_attn_ev, grouped_cross_ev
print("\n== equivalence ==", flush=True)
R["equiv"] = {}
for K in (1, 2, 4, min(8, A.kmax)):
    if K > len(evs_d):
        continue
    Bp = pack(evs_d[:K])
    Bp["n_cells"] = Bp["plane_id"].shape[0]
    mp = torch.cat(masks[:K])
    try:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            got = forward_feat_ev(model, Bp, mp, Bp["offset"]).float()
        s = 0; worst = 0.0; rel = 0.0
        for i in range(K):
            n = int(evs_d[i]["n_cells"])
            d = (got[s:s+n] - refs[i]).abs()
            worst = max(worst, float(d.max()))
            rel = max(rel, float(d.max() / refs[i].abs().max()))
            s += n
        R["equiv"][K] = dict(max_abs=worst, max_rel=rel)
        print(f"  K={K}  max|packed - solo| = {worst:.3e}   rel {rel:.3e}"
              f"{'   BIT-EXACT' if worst == 0 else ''}", flush=True)
    except Exception as e:
        R["equiv"][K] = dict(error=repr(e), tb=traceback.format_exc()[-900:])
        print(f"  K={K} FAILED {e}")

# ---------------------------------------------------------------- throughput
print("\n== throughput, event-aware pack vs sequential accumulation ==", flush=True)
from helix.model.loss import losses_cat
NS, KB = model.n_slot, model.n_bins


def head_bmm(model, feat, B, tok_mask):
    occ = model.occ_head(feat) * model.readout_mult
    mrow = tok_mask[:, None]
    m_occ = mrow & B["valid"]
    bce = (F.binary_cross_entropy_with_logits(occ, B["occ"], reduction="none")
           * m_occ).sum() / m_occ.sum().clamp(min=1)
    from helix.model.loss import bucketize_bins
    act = B["occ"].bool() & B["valid"] & mrow
    ci, si = act.nonzero(as_tuple=True)
    P = ci.numel()
    binid = bucketize_bins(B["tgt"][ci, si][:, None], B["band_id"][ci],
                           model.bin_edges, KB).squeeze(1)
    o = torch.argsort(si)
    ci_s, si_s, bin_s = ci[o], si[o], binid[o]
    cnt = torch.bincount(si_s, minlength=NS); mx = int(cnt.max())
    st = _cum0(cnt)
    col = torch.arange(mx, device=feat.device)[None, :]
    valid = col < cnt[:, None]
    flat = (st[:, None] + col).clamp(max=max(P - 1, 0))
    f = feat[ci_s[flat]]
    W = model.val_head.weight.view(NS, KB, -1)
    bv = model.val_head.bias.view(NS, KB)
    lg = torch.baddbmm(bv[:, None, :].to(f.dtype), f, W.transpose(1, 2).to(f.dtype)) * model.readout_mult
    ce = F.cross_entropy(lg.reshape(-1, KB).float(), bin_s[flat].reshape(-1), reduction="none")
    return bce, (ce.view(NS, mx) * valid).sum() / float(P)


opt = torch.optim.AdamW(model.param_groups(1.1e-3, weight_decay=0.05), betas=(0.9, 0.95))
rows = []
for K in range(1, A.kmax + 1):
    row = dict(K=K, cells=sum(R["n_cells"][:K]))
    # sequential accumulation, shipped attention
    S.uniform_attn, S.grouped_cross = _orig
    def seq():
        opt.zero_grad(set_to_none=True)
        for i in range(K):
            with torch.autocast("cuda", torch.bfloat16):
                b, v = head_bmm(model, model.forward_feat(evs_d[i], masks[i]), evs_d[i], masks[i])
            ((b + v) / K).backward()
        opt.step()
    try:
        t = timeit(seq, warmup=1, iters=max(2, A.iters // 2))
        with peak_mem() as pm:
            seq()
        row.update(seq_ms=t["ms_med"], seq_peak_MiB=pm["peak_alloc_MiB"])
    except torch.cuda.OutOfMemoryError:
        row["seq_oom"] = True
    opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    # event-aware pack
    S.uniform_attn, S.grouped_cross = uniform_attn_ev, grouped_cross_ev
    try:
        Bp = pack(evs_d[:K]); Bp["n_cells"] = Bp["plane_id"].shape[0]
        mp = torch.cat(masks[:K])
        def pk():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                feat = forward_feat_ev(model, Bp, mp, Bp["offset"])
                b, v = head_bmm(model, feat, Bp, mp)
            (b + v).backward()
            opt.step()
        t = timeit(pk, warmup=1, iters=max(2, A.iters // 2))
        with peak_mem() as pm:
            pk()
        row.update(pack_ms=t["ms_med"], pack_peak_MiB=pm["peak_alloc_MiB"],
                   speedup=row.get("seq_ms", float("nan")) / t["ms_med"],
                   ms_per_event=t["ms_med"] / K,
                   MiB_per_event=pm["peak_alloc_MiB"] / K)
    except torch.cuda.OutOfMemoryError:
        row["pack_oom"] = True
    except Exception as e:
        row["pack_err"] = repr(e)
    opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    rows.append(row)
    print(f"  K={K} cells={row['cells']:7d} seq={row.get('seq_ms',float('nan')):8.1f}ms/"
          f"{row.get('seq_peak_MiB',0):8.0f}MiB  pack={row.get('pack_ms',float('nan')):8.1f}ms/"
          f"{row.get('pack_peak_MiB',0):8.0f}MiB  x{row.get('speedup',float('nan')):.2f}"
          f"  {row.get('ms_per_event',float('nan')):6.1f} ms/event", flush=True)
R["throughput"] = rows
S.uniform_attn, S.grouped_cross = _orig
emit("p10_eventaware", R)
