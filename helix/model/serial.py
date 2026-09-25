"""SerialFMModel — the production variant (grouped/serialised attention).

FMModel with a grouped 4-order serial encoder and a grouped-cross decoder. It
reuses ALL FMModel weights (same Block/CrossBlock params), so it warm-starts
from a full-attention checkpoint.

rope_split=True  -> axial (t,wire) RoPE on plane-order layers, TIME-ONLY on
                    drift-order layers + decoder (the fix).
rope_split=False -> global (t,wire) RoPE everywhere (the full-attn behaviour).

Each encoder block attends within groups of ``g`` tokens taken along one of
four orders; the decoder's masked queries cross-attend visible tokens in
drift-time groups. Groups are padded to ``npad = ceil(T/g)*g`` with the last
token repeated, and those pads are attended unmasked.

The residual stream is CARRIED in grouped, padded order. LayerNorm, qkv, the
projection, the MLP and both residual adds are row-wise, so only attention cares
about order, and a block costs one gather composing the previous layout with
this one instead of gathering q, k, v and scattering the output back. The
decoder's query and key orders are the same in every block, so it is permuted
once before the stack and back once after. The arithmetic is the plain
per-block formulation's, bit for bit (tests/test_serial.py keeps that
formulation as the reference).
"""
import torch
import torch.nn.functional as F

from helix.model.fm import FMModel, rope_angles
from helix.model.layers import apply_rope_tables, rope_tables


def _pad_idx(order, T, npad):
    """`order`, then its last token repeated up to `npad`, as one gather index."""
    return order[torch.arange(npad, device=order.device).clamp(max=T - 1)]


def _inverse(order):
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return inv


def _attend(q, k, v, blk, kmask):
    """SDPA over grouped (nb, h, g, hd) tensors, with the block's attention sinks
    appended to every group's keys and ``kmask`` (nb, 1, 1, g) excluding padding."""
    if blk.n_sink:
        nb = k.shape[0]
        k = torch.cat([k, blk.sink_k.to(k.dtype).expand(nb, -1, -1, -1)], 2)
        v = torch.cat([v, blk.sink_v.to(v.dtype).expand(nb, -1, -1, -1)], 2)
        if kmask is not None:
            kmask = torch.cat([kmask, kmask.new_ones(nb, 1, 1, blk.n_sink)], -1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=kmask)


def self_block(blk, x, cos, sin, nb, g, c=None, kmask=None):
    """Block.forward over a stream already in grouped, padded order. It MUST
    mirror every branch Block has, AdaLN included — a missing branch here is
    silently dead conditioning, not an error."""
    P, d = x.shape
    if blk.adaln:
        sa, ba, ga, sm, bm, gm = blk.ada(c).chunk(6, -1)
        hh = blk.n1(x) * (1 + sa) + ba
    else:
        hh = blk.n1(x)
    q, k, v = blk.qkv(hh).chunk(3, -1)
    q = apply_rope_tables(q.view(P, blk.h, blk.hd), cos, sin)
    k = apply_rope_tables(k.view(P, blk.h, blk.hd), cos, sin)
    grp = lambda t: t.view(nb, g, blk.h, blk.hd).permute(0, 2, 1, 3)
    o = _attend(grp(q), grp(k), grp(v.view(P, blk.h, blk.hd)), blk, kmask)
    ao = blk.proj(o.permute(0, 2, 1, 3).reshape(P, d))
    x = x + (ga * ao if blk.adaln else ao)
    if blk.adaln:
        return x + gm * blk.mlp(blk.n2(x) * (1 + sm) + bm)
    return x + blk.mlp(blk.n2(x))


def cross_block(blk, q, kv, qcos, qsin, kcos, ksin, nb, gq, gk, c=None, kmask=None):
    """CrossBlock.forward over query and key sets already in grouped, padded
    order — same mirroring obligation as self_block."""
    Pq, Pk = q.shape[0], kv.shape[0]
    if blk.adaln:
        sa, ba, ga, sm, bm, gm = blk.ada(c).chunk(6, -1)
        hq = blk.nq(q) * (1 + sa) + ba
    else:
        hq = blk.nq(q)
    qh = apply_rope_tables(blk.q(hq).view(Pq, blk.h, blk.hd), qcos, qsin)
    k, v = blk.kv(blk.nk(kv)).chunk(2, -1)
    kh = apply_rope_tables(k.view(Pk, blk.h, blk.hd), kcos, ksin)
    o = F.scaled_dot_product_attention(
        qh.view(nb, gq, blk.h, blk.hd).permute(0, 2, 1, 3),
        kh.view(nb, gk, blk.h, blk.hd).permute(0, 2, 1, 3),
        v.view(Pk, blk.h, blk.hd).view(nb, gk, blk.h, blk.hd).permute(0, 2, 1, 3),
        attn_mask=kmask)
    ao = blk.proj(o.permute(0, 2, 1, 3).reshape(Pq, blk.h * blk.hd))
    q = q + (ga * ao if blk.adaln else ao)
    if blk.adaln:
        return q + gm * blk.mlp(blk.n2(q) * (1 + sm) + bm)
    return q + blk.mlp(blk.n2(q))


_COMPILED = {}


def _detach_dynamo_finalizers_at_exit():
    """Stop dynamo's guard finalizers from running at interpreter exit.

    Each compiled frame's guards hold a weakref.finalize(obj, invalidate) whose
    guard manager keeps a NON-owning pointer to its cache entry. At exit the
    atexit pass runs those finalizers after the entries can already be freed,
    and invalidate() dereferences the dangling pointer: a SIGSEGV after the
    run has finished and its checkpoint is complete (torch 2.10; seen in ~half
    of the runs that loaded weights). Nothing needs invalidating at exit, so
    they are detached first. atexit runs last-registered-first, so weakref's
    own exit hook is forced to register before this one.
    """
    import atexit
    import weakref

    weakref.finalize(_detach_dynamo_finalizers_at_exit, lambda: None)

    def _detach():
        from torch._dynamo.guards import CheckFunctionManager
        for f, info in list(weakref.finalize._registry.items()):
            fn = getattr(info.func, "func", info.func)      # functools.partial
            if getattr(fn, "__func__", None) is CheckFunctionManager.invalidate:
                f.detach()

    atexit.register(_detach)


def _ckpt(fn, model):
    """``fn`` with its activations recomputed in backward when model.act_ckpt."""
    if not (model.act_ckpt and torch.is_grad_enabled()):
        return fn
    from torch.utils.checkpoint import checkpoint
    return lambda *args: checkpoint(fn, *args, use_reentrant=False)


def _blocks(model):
    """(self_block, cross_block), compiled when ``model.compile_blocks``.

    One torch.compile per process with dynamic shapes, so a new token count does
    not recompile, and every block shares one graph (parameters are graph
    inputs). Each function still specialises on a mod-8 alignment of the group
    size, a fusion-size threshold and grad mode: 8 recompiles per rank, all in
    the first few hundred steps, which is exactly dynamo's default limit — and
    past the limit it runs eagerly without saying so. Hence the headroom.

    optimize_ddp is off because dynamo's DDP optimizer splits a compiled graph
    at DDP's gradient-bucket boundaries, and with dynamic shapes that split
    fails to compile (BackendCompilerFailed: 'int' object has no attribute
    'meta'). It only triggers once one block's parameters exceed a 25 MB bucket
    -- d768 and up -- which is why d512 never showed it. Each compiled region is
    one block, so DDP still overlaps its all-reduce across blocks without it.
    """
    if not model.compile_blocks:
        return self_block, cross_block
    if not _COMPILED:
        import torch._dynamo as _dyn
        _dyn.config.optimize_ddp = False
        _detach_dynamo_finalizers_at_exit()
        for knob in ("recompile_limit", "cache_size_limit"):
            if hasattr(_dyn.config, knob):
                setattr(_dyn.config, knob, max(64, getattr(_dyn.config, knob)))
        _COMPILED["self"] = torch.compile(self_block, dynamic=True)
        _COMPILED["cross"] = torch.compile(cross_block, dynamic=True)
    return _COMPILED["self"], _COMPILED["cross"]


class SerialFMModel(FMModel):
    def __init__(self, *args, rope_split=True, gp=1024, gd=2048, **kw):
        super().__init__(*args, **kw)
        self.rope_split, self.gp, self.gd = rope_split, gp, gd

    def _layouts(self, plane, t, wire):
        """The encoder's distinct (order, group size, wire RoPE on) layouts;
        block i uses ``layouts[i % 4]``."""
        bp = plane.double()
        o_pt = torch.argsort(bp * 1e9 + t.double())
        o_pw = torch.argsort(bp * 1e13 + wire.double() * 1e6 + t.double())
        o_t = torch.argsort(t.double()); o_ts = torch.roll(o_t, self.gd // 2)
        dw = not self.rope_split                                   # drift-layer wire RoPE: off if split
        cell = [(o_pt, self.gp, True), (o_t, self.gd, dw), (o_pw, self.gp, True), (o_ts, self.gd, dw)]
        return cell[:len(self.enc)]

    def _emb(self, B, idx=None):
        band, plane = B["band_id"], B["plane_id"]
        sel = slice(None) if idx is None else idx
        x = self.embed(torch.cat([B["inp"][sel], B["occ"][sel]], -1))
        if self.cond == "adaln":
            return x                       # identity enters through AdaLN, not additively
        if self.film is not None:
            g, b = self.film(band[sel], plane[sel], B["wirefeat"][sel]); x = g * x + b
        return x + self.band_emb(band[sel]) + self.plane_emb(plane[sel])

    def _angles(self, B):
        hd = self.d // self.heads
        return (rope_angles(B["t_phys"], hd, *self.lam_t),
                rope_angles(B["wire_pos"], hd, *self.lam_w))

    def _encode(self, B, idx, at, aw, c, layers=()):
        """The encoder over rows ``idx`` of B (None = all) -> (x, {layer: x}),
        both in natural order. ``at``/``aw``/``c`` are already those rows'."""
        sel = slice(None) if idx is None else idx
        x = self._emb(B, idx)
        T = x.shape[0]
        lay = []
        for o, g, uw in self._layouts(B["plane_id"][sel], B["t_phys"][sel], B["wire_pos"][sel]):
            npad = ((T + g - 1) // g) * g
            pi = _pad_idx(o, T, npad)
            cos, sin = rope_tables(at[pi], aw[pi] if uw else None, x.dtype)
            km = ((torch.arange(npad, device=x.device) < T).view(npad // g, 1, 1, g)
                  if self.pad_mask else None)
            lay.append((pi, _inverse(o), npad // g, g, cos, sin,
                        None if c is None else c[pi], km))
        # layout j's padded rows, gathered from layout j-1's padded stream
        step = [lay[j - 1][1][lay[j][0]] for j in range(len(lay))]
        self_blk = _ckpt(_blocks(self)[0], self)
        xp, out = x[lay[0][0]], {}
        for i, blk in enumerate(self.enc):
            j = i % len(lay)
            if i:
                xp = xp[step[j]]
            _, inv, nb, g, cos, sin, cp, km = lay[j]
            xp = self_blk(blk, xp, cos, sin, nb, g, cp, km)
            if i + 1 in layers:
                out[i + 1] = xp[inv]
        return xp[inv], out

    def encode(self, B):
        """Per-token encoder representation with ALL tokens visible (no masking).
        This is the frozen feature the deconvolution probe reads."""
        at, aw = self._angles(B)
        c = self._cond(B) if self.cond == "adaln" else None
        return self._encode(B, None, at, aw, c)[0]

    def encode_layers(self, B, layers):
        """Per-token features after each requested encoder block (1-based). {k: (N,d)}."""
        at, aw = self._angles(B)
        c = self._cond(B) if self.cond == "adaln" else None
        return self._encode(B, None, at, aw, c, layers)[1]

    def forward_feat(self, B, tok_mask, masked_only=False):
        """(N, d) decoded features; ``masked_only`` -> (masked rows, their
        indices), for an objective that reads nothing else."""
        at, aw = self._angles(B)
        vis = ~tok_mask
        vis_idx, mask_idx = vis.nonzero(as_tuple=True)[0], tok_mask.nonzero(as_tuple=True)[0]
        c = self._cond(B) if self.cond == "adaln" else None
        atv, awv = at[vis], aw[vis]
        xv, _ = self._encode(B, vis_idx, atv, awv, None if c is None else c[vis])

        # CrossMAE decoder (grouped, drift-time): masked queries x-attend visible
        qm = self.mask_tok.expand(mask_idx.numel(), self.d)
        if c is None:                           # with AdaLN, identity enters there instead
            if self.film is not None:
                g_, b_ = self.film(B["band_id"][tok_mask], B["plane_id"][tok_mask], B["wirefeat"][tok_mask]); qm = g_ * qm + b_
            qm = qm + self.band_emb(B["band_id"][tok_mask]) + self.plane_emb(B["plane_id"][tok_mask])
        qm = qm.to(xv.dtype)
        Tq, Tk = qm.shape[0], xv.shape[0]
        oq = torch.argsort(B["t_phys"][tok_mask].double()); okv = torch.argsort(B["t_phys"][vis].double())
        nb = (max(Tq, Tk) + self.gd - 1) // self.gd
        gq, gk = (Tq + nb - 1) // nb, (Tk + nb - 1) // nb
        iq, ik = _pad_idx(oq, Tq, nb * gq), _pad_idx(okv, Tk, nb * gk)
        # Decoder ALWAYS keeps axial (wire) RoPE: it needs the wire address to know which wire it
        # reconstructs -- dropping it froze recon (var_expl ~2%, same as wire_rope=0). rope_split
        # only affects the ENCODER cross-plane (drift-order) layers, via `dw` in _layouts.
        qcos, qsin = rope_tables(at[tok_mask][iq], aw[tok_mask][iq], qm.dtype)
        kcos, ksin = rope_tables(atv[ik], awv[ik], xv.dtype)
        cm = None if c is None else c[tok_mask][iq]
        km = ((torch.arange(nb * gk, device=xv.device) < Tk).view(nb, 1, 1, gk)
              if self.pad_mask else None)
        cross_blk = _ckpt(_blocks(self)[1], self)
        qp, kvp = qm[iq], xv[ik]
        for blk in self.dec:
            qp = cross_blk(blk, qp, kvp, qcos, qsin, kcos, ksin, nb, gq, gk, cm, km)
        qm = qp[_inverse(oq)]

        if masked_only:
            return self.dec_norm(qm), mask_idx
        x = torch.zeros(B["inp"].shape[0], self.d, dtype=xv.dtype, device=xv.device)
        x = x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm)
        return self.dec_norm(x)
