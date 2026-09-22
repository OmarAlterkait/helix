"""Reusable pieces of the interventions, so p11 measures the SAME code p4/p9/p10
measured rather than a retyped cousin."""
from __future__ import annotations
import torch, torch.nn.functional as F
import helix.model.serial as S
from helix.model.fm import rope_angles
from helix.model.loss import bucketize_bins


def _cum0(x):
    return torch.cat([x.new_zeros(1), x.cumsum(0)[:-1]])


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


def head_bmm(model, feat, B, tok_mask):
    NS, KB = model.n_slot, model.n_bins
    occ = model.occ_head(feat) * model.readout_mult
    mrow = tok_mask[:, None]
    m_occ = mrow & B["valid"]
    bce = (F.binary_cross_entropy_with_logits(occ, B["occ"], reduction="none")
           * m_occ).sum() / m_occ.sum().clamp(min=1)
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


