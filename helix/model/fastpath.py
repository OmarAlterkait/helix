"""The permuted-residual forward: one gather per block instead of four.

``serial.uniform_attn`` gathers q, k and v into grouped layout (three gathers of
``(T, d)``) and scatters the output back (one scatter) — four passes over the
largest tensor in the model, per block. But LayerNorm, qkv, the projection, the
MLP and both residual adds are **row-wise**: none of them cares what order the
tokens are in. So the residual stream can simply be CARRIED permuted and padded,
and a block costs one gather that composes the previous block's permutation with
this one's::

    src_i[p] = inv_{i-1}[ order_i[min(p, T-1)] ]

The pad rows are duplicates of the block's last real token and stay duplicates
through the block (every op is row-wise, and they sit in the same attention
group as the token they copy), which is why this is the identical computation.

**The decoder is worse and it is the bigger half.** At ``mask_ratio = 0.75`` the
encoder sees ~25 % of tokens and the decoder runs ~75 % as queries. All four
``CrossBlock``s call ``grouped_cross`` with the SAME ``oq`` and ``okv``, so the
query set is gathered 4x, the key/value set 8x and the output scattered 4x — all
by one permutation. Permuting once before the loop and back after leaves two
gathers and one scatter for the whole decoder.

Measured on an A100 at the corpus median event, against the shipped path:
**1.13x on the full step, 1.27x on the encoder alone, max|delta| = 0**
(``docs/PERFORMANCE.md`` §7b, ``tools/profile/p15_traffic.py``).

WHAT THIS DELIBERATELY DOES NOT CHANGE. The padding contract is reproduced
exactly: ``npad = ceil(T/g)*g`` with the last token duplicated, those pads
attended unmasked. ``serial.grouped_cross`` computes a much tighter pad from the
same inputs, and adopting it here would cut 567 pad rows to 7 — but it is a
numerics change, so it belongs to a run, not to a speedup. See
``docs/REVIEW_FIELD.md`` §1.1.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from helix.model.fm import rope_angles
from helix.model.rope import apply_rope_fused, rope_tables


# --------------------------------------------------------------- block plans

def _pad_idx(order, T, npad):
    """Padded layout: the block's tokens in `order`, then its last one repeated.

    Identical to ``serial.uniform_attn``'s ``b[:T] = x[order]; b[T:] =
    x[order[-1]]``, expressed as one index so the gather is a single kernel —
    and, as a side effect, without the ``x[order[-1]]`` 0-d tensor index that
    costs a host synchronisation per call (48 per step).
    """
    return order[torch.arange(npad, device=order.device).clamp(max=T - 1)]


def self_plan(orders, gs, T, device):
    """Per encoder block: (gather index from the previous padded layout, nb, g),
    plus the inverse of the last permutation so the stream can be returned to
    natural order."""
    plan, prev_inv = [], None
    ar = torch.arange(T, device=device)
    for o, g in zip(orders, gs):
        npad = ((T + g - 1) // g) * g
        tok = _pad_idx(o, T, npad)
        plan.append((tok if prev_inv is None else prev_inv[tok], npad // g, g))
        inv = torch.empty(T, dtype=torch.long, device=device)
        inv[o] = ar
        prev_inv = inv
    return plan, prev_inv


def cross_plan(oq, okv, Tq, Tk, g):
    """``serial.grouped_cross``'s own block geometry, computed once for all four
    decoder blocks instead of once per block."""
    nb = (max(Tq, Tk) + g - 1) // g
    gq = (Tq + nb - 1) // nb
    gk = (Tk + nb - 1) // nb
    return _pad_idx(oq, Tq, nb * gq), _pad_idx(okv, Tk, nb * gk), nb, gq, gk


# ------------------------------------------------------------------- blocks

def self_block(blk, xp, cos, sin, nb, g):
    """``serial._self`` over an already-permuted, already-padded stream."""
    P, d = xp.shape
    if blk.adaln:
        raise NotImplementedError(
            "fast_path does not implement AdaLN conditioning; build with "
            "cond='film' or set fast_path=False")
    q, k, v = blk.qkv(blk.n1(xp)).chunk(3, -1)
    q = apply_rope_fused(q.view(P, blk.h, blk.hd), cos, sin)
    k = apply_rope_fused(k.view(P, blk.h, blk.hd), cos, sin)
    shp = lambda t: t.view(nb, g, blk.h, blk.hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(shp(q), shp(k), shp(v.view(P, blk.h, blk.hd)))
    xp = xp + blk.proj(o.permute(0, 2, 1, 3).reshape(P, d))
    return xp + blk.mlp(blk.n2(xp))


def cross_block(blk, qp, kvp, qcos, qsin, kcos, ksin, nb, gq, gk):
    """``serial._cross`` over query and key sets that are already in oq/okv
    order and padded — done once for the whole decoder stack."""
    if blk.adaln:
        raise NotImplementedError("fast_path does not implement AdaLN conditioning")
    Pq, d = qp.shape
    Pk = kvp.shape[0]
    qh = apply_rope_fused(blk.q(blk.nq(qp)).view(Pq, blk.h, blk.hd), qcos, qsin)
    k, v = blk.kv(blk.nk(kvp)).chunk(2, -1)
    kh = apply_rope_fused(k.view(Pk, blk.h, blk.hd), kcos, ksin)
    o = F.scaled_dot_product_attention(
        qh.view(nb, gq, blk.h, blk.hd).permute(0, 2, 1, 3),
        kh.view(nb, gk, blk.h, blk.hd).permute(0, 2, 1, 3),
        v.view(Pk, blk.h, blk.hd).view(nb, gk, blk.h, blk.hd).permute(0, 2, 1, 3))
    qp = qp + blk.proj(o.permute(0, 2, 1, 3).reshape(Pq, d))
    return qp + blk.mlp(blk.n2(qp))


# ------------------------------------------------------------------ forward

_COMPILED = {}


def _blocks(model):
    """(self_block, cross_block), compiled when the model asks for it.

    One torch.compile per process, dynamic=True: the token count changes every
    step, and with shapes symbolic a new count does not recompile. Every block
    shares ONE graph -- inline_inbuilt_nn_modules makes parameters graph inputs
    rather than guarding on each module -- verified in the image as 1 frame,
    1 graph, 0 recompiles over 12 blocks x 4 token counts. Compiling the whole
    block is what pays (docs/PERFORMANCE.md §7c): the MLP alone is already two
    GEMMs and a GELU, and there is nothing to fuse across.
    """
    if not getattr(model, "compile_blocks", False):
        return self_block, cross_block
    if not _COMPILED:
        _COMPILED["self"] = torch.compile(self_block, dynamic=True)
        _COMPILED["cross"] = torch.compile(cross_block, dynamic=True)
    return _COMPILED["self"], _COMPILED["cross"]


def forward_feat(model, B, tok_mask, masked_only=False):
    """``SerialFMModel.forward_feat``, permuted.

    ``masked_only=True`` returns ``(feat_masked, mask_idx)`` instead of the full
    ``(N, d)``. At ``vis_w == 0`` no visible row enters the objective — the BCE
    is masked by ``tok_mask[:, None]`` and the value term by ``... & mrow`` — so
    the ``zeros(N, d)``, the two ``index_copy``s, the LayerNorm over all N and
    both heads over all N spend a quarter of their work on rows multiplied by
    zero. The probe wants every row, so this is a TRAINING option, never the
    default contract.
    """
    _self_blk, _cross_blk = _blocks(model)
    hd = model.d // model.heads
    vis = ~tok_mask
    vis_idx = vis.nonzero(as_tuple=True)[0]
    mask_idx = tok_mask.nonzero(as_tuple=True)[0]

    xv = model._emb(B, vis_idx)
    T = xv.shape[0]
    atv = rope_angles(B["t_phys"][vis], hd, *model.lam_t)
    awv = rope_angles(B["wire_pos"][vis], hd, *model.lam_w)
    sched = model._sched(B["plane_id"][vis], B["t_phys"][vis], B["wire_pos"][vis])
    orders = [o for o, g, uw in sched]
    plan, last_inv = self_plan(orders, [g for o, g, uw in sched], T, xv.device)

    xp = xv[plan[0][0]]
    tables = {}
    for i, (blk, (src, nb, g)) in enumerate(zip(model.enc, plan)):
        if i:
            xp = xp[src]
        # Four distinct (order, g, wire-on) layouts cycle across the stack, so
        # the tables are built four times per forward rather than 24.
        key = (int(orders[i].data_ptr()), g, sched[i][2])
        if key not in tables:
            tk = _pad_idx(orders[i], T, nb * g)
            tables[key] = rope_tables(atv[tk], awv[tk] if sched[i][2] else None,
                                      xp.dtype)
        cos, sin = tables[key]
        xp = _self_blk(blk, xp, cos, sin, nb, g)
    xv = xp[last_inv]

    qm = model.mask_tok.expand(mask_idx.numel(), model.d)
    if model.film is not None:
        g_, b_ = model.film(B["band_id"][tok_mask], B["plane_id"][tok_mask],
                            B["wirefeat"][tok_mask])
        qm = g_ * qm + b_
    qm = (qm + model.band_emb(B["band_id"][tok_mask])
          + model.plane_emb(B["plane_id"][tok_mask])).to(xv.dtype)

    atm = rope_angles(B["t_phys"][tok_mask], hd, *model.lam_t)
    awm = rope_angles(B["wire_pos"][tok_mask], hd, *model.lam_w)
    Tq, Tk = qm.shape[0], xv.shape[0]
    oq = torch.argsort(B["t_phys"][tok_mask].double())
    okv = torch.argsort(B["t_phys"][vis].double())
    iq, ik, nb, gq, gk = cross_plan(oq, okv, Tq, Tk, model.gd)
    qcos, qsin = rope_tables(atm[iq], awm[iq], qm.dtype)
    kcos, ksin = rope_tables(atv[ik], awv[ik], xv.dtype)
    qp, kvp = qm[iq], xv[ik]
    for blk in model.dec:
        qp = _cross_blk(blk, qp, kvp, qcos, qsin, kcos, ksin, nb, gq, gk)
    inv = torch.empty(Tq, dtype=torch.long, device=xv.device)
    inv[oq] = torch.arange(Tq, device=xv.device)
    qm = qp[inv]

    if masked_only:
        return model.dec_norm(qm), mask_idx
    N = B["inp"].shape[0]
    x = torch.zeros(N, model.d, dtype=xv.dtype, device=xv.device)
    x = x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm)
    return model.dec_norm(x)
