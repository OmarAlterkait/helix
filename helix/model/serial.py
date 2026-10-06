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
from helix.model.layers import apply_rope_tables, qk_normed, rope_tables


def _pad_idx(order, T, npad):
    """`order`, then its last token repeated up to `npad`, as one gather index."""
    return order[torch.arange(npad, device=order.device).clamp(max=T - 1)]


def _inverse(order):
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return inv


class _Permute(torch.autograd.Function):
    """``x[idx]`` where ``idx[:T]`` is a permutation of ``x``'s first ``T`` rows and
    ``idx[T:]`` (padding) repeats ``idx[T - 1]`` -- every gather in this model.

    Autograd's backward for a gather is a general scatter-add: it sorts the
    indices and accumulates, 8-11 ms of a 96-118 ms d768 step (7-11 %). A
    permutation's backward is a gather by its inverse, and the padding's
    gradients all belong to one row. Same forward; the backward differs from
    autograd's only in summation order over the padding rows."""

    @staticmethod
    def forward(ctx, x, idx, T):
        inv = torch.empty(T, dtype=idx.dtype, device=idx.device)
        inv[idx[:T]] = torch.arange(T, device=idx.device)
        ctx.save_for_backward(idx, inv)
        ctx.T, ctx.n = T, x.shape[0]
        return x[idx]

    @staticmethod
    def backward(ctx, gy):
        idx, inv = ctx.saved_tensors
        T = ctx.T
        gx = gy.new_zeros((ctx.n,) + tuple(gy.shape[1:]))
        gx[:T] = gy[:T][inv]
        if gy.shape[0] > T:        # index_add_, not gx[idx[T-1]]: a 0-d index is an .item() sync
            gx.index_add_(0, idx[T - 1:T], gy[T:].sum(0, keepdim=True).to(gx.dtype))  # sum is fp32 under autocast
        return gx, None, None


def _permute(x, idx, T):
    return _Permute.apply(x, idx, T) if torch.is_grad_enabled() and x.requires_grad else x[idx]


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
    q, k = qk_normed(blk, q.view(P, blk.h, blk.hd), k.view(P, blk.h, blk.hd))
    q = apply_rope_tables(q, cos, sin)
    k = apply_rope_tables(k, cos, sin)
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
    k, v = blk.kv(blk.nk(kv)).chunk(2, -1)
    qh, kh = qk_normed(blk, blk.q(hq).view(Pq, blk.h, blk.hd), k.view(Pk, blk.h, blk.hd))
    qh = apply_rope_tables(qh, qcos, qsin)
    kh = apply_rope_tables(kh, kcos, ksin)
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


# --- varlen: the same groups, without padding ------------------------------
# With ``varlen=True`` a layout's groups are its consecutive g-token chunks with
# the LAST ONE SHORTER, run through flash attention with cumulative group
# boundaries (``cu``) instead of being padded to a multiple of g with copies of
# the last token. Measured at d1536 / 18.5k tokens: the padded layouts made the
# encoder run 5,120 or 6,144 rows for 4,625 tokens -- 9% of all FLOPs -- and the
# pads were attended. The decoder's time groups are split evenly. The stream is
# still carried in layout order; permutes become pure permutations.

def _vl_attn(q, k, v, cuq, cuk, mq, mk):
    """Flash varlen attention over (T, H, hd) groups; on CPU / fp32 (tests, eval
    without autocast) the same groups one SDPA at a time."""
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        from torch.nn.attention.varlen import varlen_attn
        return varlen_attn(q, k, v, cuq, cuk, mq, mk)
    out = torch.empty_like(q)
    bq, bk = cuq.tolist(), cuk.tolist()
    for i in range(len(bq) - 1):
        sq, sk = slice(bq[i], bq[i + 1]), slice(bk[i], bk[i + 1])
        out[sq] = F.scaled_dot_product_attention(
            q[sq].transpose(0, 1), k[sk].transpose(0, 1), v[sk].transpose(0, 1)).transpose(0, 1)
    return out


def _qk_rope(blk, X, cos, sin, mode, fused):
    """q, k (and v) from a fused projection: QK-norm, then RoPE. ``mode`` as
    helix.model.kernels.normrope; the unfused path is the reference, and the one
    taken off-GPU (the kernel is Triton; a fused_qk checkpoint must still run on
    CPU)."""
    if fused and X.is_cuda:
        from helix.model.kernels import normrope
        return normrope(X, cos, sin, blk.qn.weight if mode != "kv" else blk.kn.weight,
                        blk.kn.weight if mode == "qkv" else None, mode)
    T = X.shape[0]
    view = lambda t: t.reshape(T, blk.h, blk.hd)
    if mode == "qkv":
        q, k, v = X.chunk(3, -1)
        q, k = qk_normed(blk, view(q), view(k))
        return apply_rope_tables(q, cos, sin), apply_rope_tables(k, cos, sin), view(v)
    if mode == "kv":
        k, v = X.chunk(2, -1)
        k = blk.kn(view(k)) if blk.kn is not None else view(k)
        return apply_rope_tables(k, cos, sin), view(v)
    q = blk.qn(view(X)) if blk.qn is not None else view(X)
    return (apply_rope_tables(q, cos, sin),)


def self_block_varlen(blk, x, cos, sin, cu, maxlen, c=None, fused=False):
    """self_block over a stream in layout order whose groups are ``cu``."""
    T, d = x.shape
    if blk.adaln:
        sa, ba, ga, sm, bm, gm = blk.ada(c).chunk(6, -1)
        hh = blk.n1(x) * (1 + sa) + ba
    else:
        hh = blk.n1(x)
    q, k, v = _qk_rope(blk, blk.qkv(hh), cos, sin, "qkv", fused)
    ao = blk.proj(_vl_attn(q, k, v, cu, cu, maxlen, maxlen).reshape(T, d))
    x = x + (ga * ao if blk.adaln else ao)
    if blk.adaln:
        return x + gm * blk.mlp(blk.n2(x) * (1 + sm) + bm)
    return x + blk.mlp(blk.n2(x))


def cross_block_varlen(blk, q, kv, qcos, qsin, kcos, ksin, cuq, cuk, mq, mk, c=None, fused=False):
    """cross_block with query groups ``cuq`` attending key groups ``cuk``."""
    Tq = q.shape[0]
    if blk.adaln:
        sa, ba, ga, sm, bm, gm = blk.ada(c).chunk(6, -1)
        hq = blk.nq(q) * (1 + sa) + ba
    else:
        hq = blk.nq(q)
    (qh,) = _qk_rope(blk, blk.q(hq), qcos, qsin, "q", fused)
    kh, vh = _qk_rope(blk, blk.kv(blk.nk(kv)), kcos, ksin, "kv", fused)
    ao = blk.proj(_vl_attn(qh, kh, vh, cuq, cuk, mq, mk).reshape(Tq, blk.h * blk.hd))
    q = q + (ga * ao if blk.adaln else ao)
    if blk.adaln:
        return q + gm * blk.mlp(blk.n2(q) * (1 + sm) + bm)
    return q + blk.mlp(blk.n2(q))


def _cu_chunks(T, g, device):
    """[0, g, 2g, ..., T] built on device (a host-built tensor is a copy + sync)."""
    return torch.arange(0, T + g, g, device=device).clamp_(max=T).to(torch.int32)


def _cu_even(T, nb, device):
    """nb near-equal groups over T rows: no group is empty while T >= nb."""
    return ((torch.arange(nb + 1, device=device) * T) // nb).to(torch.int32)


def _locations(plane, t, wire, band, cell, sub):
    """Group token rows into LOCATIONS: (plane, ``cell[0]``-wire block,
    ``cell[1]``-tick drift window), every band together (helix.model.mask's
    ``location`` unit). -> (row -> location index, row -> typed sub-slot,
    location plane, centre time, wire-block start). The sub-slot is the band's
    offset plus which of its ``sub[band]`` sub-windows the row's time falls in --
    A4 | D4 | D3 x2 | D2 x4 for a 128-tick window -- so pooling keeps the scale
    and the time order of what it merges. One host sync (the unique)."""
    cw, ct = cell
    wb = torch.div(wire, cw, rounding_mode="floor").long()
    u = t / ct
    tw = torch.floor(u).long()
    t0 = tw.min()
    key = (plane.long() << 40) | (wb << 20) | (tw - t0)
    uk, loc = torch.unique(key, return_inverse=True)
    lp, lwb, ltw = uk >> 40, (uk >> 20) & 0xFFFFF, (uk & 0xFFFFF) + t0
    nsub = torch.as_tensor(sub, device=t.device)
    off = torch.cumsum(nsub, 0) - nsub
    ns = nsub[band]
    slot = off[band] + torch.minimum(((u - tw) * ns).long(), ns - 1)
    return loc, slot, lp, (ltw.float() + 0.5) * ct, (lwb * cw).float()


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
    sb, cb = (self_block_varlen, cross_block_varlen) if model.varlen else (self_block, cross_block)
    if not model.compile_blocks:
        return sb, cb
    key = ("self_vl", "cross_vl") if model.varlen else ("self", "cross")
    if key[0] not in _COMPILED:
        if not _COMPILED:                                  # process-wide setup, once
            import torch._dynamo as _dyn
            _dyn.config.optimize_ddp = False
            _detach_dynamo_finalizers_at_exit()
            for knob in ("recompile_limit", "cache_size_limit"):
                if hasattr(_dyn.config, knob):
                    setattr(_dyn.config, knob, max(64, getattr(_dyn.config, knob)))
        _COMPILED[key[0]] = torch.compile(sb, dynamic=True)
        _COMPILED[key[1]] = torch.compile(cb, dynamic=True)
    return _COMPILED[key[0]], _COMPILED[key[1]]


class SerialFMModel(FMModel):
    """``band_pool=k`` runs the first ``k`` encoder blocks over band tokens, then
    POOLS every band of a location (``mask_cell``: plane x wire block x drift
    window) into one token for the remaining blocks -- 0.37x the tokens at pw16,
    0.46x of pw8's at pw8 -- with no spatial coarsening. The pool is typed: the
    tokens of a location are laid into ``sum(pool_sub)`` sub-slots by band and
    sub-window, concatenated, and projected (Swin's patch merging, over scale
    instead of space). ``pool_skip=True`` returns per-band-token features
    ``stage-1 + up(trunk[location])`` -- what the decoder attends and the probe
    reads; False returns the trunk alone (keys are the locations; a band token's
    feature is its location's). Pooling only shrinks the encoder if whole
    locations are visible or masked: train it with ``mask_mode="location"``."""

    def __init__(self, *args, rope_split=True, gp=1024, gd=2048, varlen=False,
                 band_pool=None, pool_skip=True, pool_sub=(1, 1, 2, 4), **kw):
        super().__init__(*args, **kw)
        self.rope_split, self.gp, self.gd, self.varlen = rope_split, gp, gd, bool(varlen)
        if self.varlen and kw.get("n_sink", 0):
            raise ValueError("varlen=True does not support attention sinks (n_sink>0): "
                             "flash varlen has no per-group extra keys")
        self.band_pool, self.pool_skip, self.pool_sub = band_pool, bool(pool_skip), tuple(pool_sub)
        if band_pool is not None:
            if not (self.varlen and self.cond != "adaln" and 0 <= band_pool < len(self.enc)):
                raise ValueError(f"band_pool={band_pool} needs varlen=True, cond!='adaln' and "
                                 f"0 <= band_pool < blocks={len(self.enc)}")
            if len(self.pool_sub) != len(self.band_emb.weight):
                raise ValueError(f"pool_sub {self.pool_sub} needs one entry per band ({len(self.band_emb.weight)})")
            d, S = self.d, sum(self.pool_sub)
            self.pool_norm = torch.nn.LayerNorm(d)
            self.pool_proj = torch.nn.Linear(S * d, d)
            self.pool_up = torch.nn.Linear(d, d)
            torch.nn.init.zeros_(self.pool_up.weight); torch.nn.init.zeros_(self.pool_up.bias)
            if self.mup:                                   # hidden: var / m, as _mup_init does
                with torch.no_grad():
                    self.pool_proj.weight.mul_(self.m ** -0.5)

    def _fused(self):
        """``fused_qk`` (a train option) after checking it can apply: the kernel
        is QK-norm + RoPE at head dim 64, on the varlen path."""
        if not self.fused_qk:
            return False
        hd = self.d // self.heads
        if not (self.varlen and self.enc[0].qn is not None and hd == 64):
            raise ValueError(f"fused_qk needs varlen=True, qk_norm=True and head dim 64 "
                             f"(got varlen={self.varlen}, qk_norm={self.enc[0].qn is not None}, hd={hd})")
        return True

    def _layouts(self, plane, t, wire, depth=None):
        """The encoder's distinct (order, group size, wire RoPE on) layouts;
        block i uses ``layouts[i % 4]``."""
        bp = plane.double()
        o_pt = torch.argsort(bp * 1e9 + t.double())
        o_pw = torch.argsort(bp * 1e13 + wire.double() * 1e6 + t.double())
        o_t = torch.argsort(t.double()); o_ts = torch.roll(o_t, self.gd // 2)
        dw = not self.rope_split                                   # drift-layer wire RoPE: off if split
        cell = [(o_pt, self.gp, True), (o_t, self.gd, dw), (o_pw, self.gp, True), (o_ts, self.gd, dw)]
        return cell[:len(self.enc) if depth is None else depth]

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
        if self.varlen:
            return self._encode_varlen(B, sel, x, at, aw, c, layers)
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
        xp, out = _permute(x, lay[0][0], T), {}
        for i, blk in enumerate(self.enc):
            j = i % len(lay)
            if i:
                xp = _permute(xp, step[j], T)
            _, inv, nb, g, cos, sin, cp, km = lay[j]
            xp = self_blk(blk, xp, cos, sin, nb, g, cp, km)
            if i + 1 in layers:
                out[i + 1] = _permute(xp, inv, T)
        return _permute(xp, inv, T), out

    def _encode_varlen(self, B, sel, x, at, aw, c, layers):
        """_encode with unpadded groups. The stream takes the weights' dtype, so
        with bf16 weights (``bf16_params``) the residual stream is bf16 too --
        unless ``fp32_stream``."""
        x = x.to(torch.float32 if self.fp32_stream else self.embed.weight.dtype)
        if self.band_pool is not None:
            return self._encode_pooled(B, sel, x, at, aw, layers)[:2]
        return self._stack_varlen(self.enc, x, B["plane_id"][sel], B["t_phys"][sel],
                                  B["wire_pos"][sel], at, aw, c, layers)

    def _stack_varlen(self, blocks, x, plane, t, wire, at, aw, c=None, layers=(), base=0):
        """``blocks`` over rows at (plane, t, wire) -> (x, {base + i: x}), both in
        the rows' order."""
        T, dev = x.shape[0], x.device
        lay = []
        for o, g, uw in self._layouts(plane, t, wire, len(blocks)):
            cos, sin = rope_tables(at[o], aw[o] if uw else None, x.dtype)
            lay.append((o, _inverse(o), _cu_chunks(T, g, dev), min(g, T), cos, sin,
                        None if c is None else c[o]))
        if not lay:
            return x, {}
        step = [lay[j - 1][1][lay[j][0]] for j in range(len(lay))]
        self_blk, fused = _ckpt(_blocks(self)[0], self), self._fused()
        xp, out = _permute(x, lay[0][0], T), {}
        for i, blk in enumerate(blocks):
            j = i % len(lay)
            if i:
                xp = _permute(xp, step[j], T)
            _, inv, cu, mx, cos, sin, cp = lay[j]
            xp = self_blk(blk, xp, cos, sin, cu, mx, cp, fused)
            if base + i + 1 in layers:
                out[base + i + 1] = _permute(xp, inv, T)
        return _permute(xp, inv, T), out

    def _encode_pooled(self, B, sel, x, at, aw, layers):
        """Stage 1 over band tokens, typed pool to locations, trunk over locations.
        -> (per-token features, {layer: per-token}, trunk, location of each row,
        (t, ang_t, ang_w) of each location)."""
        k = self.band_pool
        plane, t, wire, band = B["plane_id"][sel], B["t_phys"][sel], B["wire_pos"][sel], B["band_id"][sel]
        x, out = self._stack_varlen(self.enc[:k], x, plane, t, wire, at, aw, layers=layers)
        loc, slot, lp, lt, lw = _locations(plane, t, wire, band, tuple(self.mask_cell), self.pool_sub)
        S, d, n = sum(self.pool_sub), self.d, lp.numel()
        Z = x.new_zeros(n * S, d).index_add_(0, loc * S + slot, self.pool_norm(x).to(x.dtype))
        xl = self.pool_proj(Z.view(n, S * d)).to(x.dtype)
        hd = self.d // self.heads
        lat, law = rope_angles(lt, hd, *self.lam_t), rope_angles(lw, hd, *self.lam_w)
        xl, lo = self._stack_varlen(self.enc[k:], xl, lp, lt, lw, lat, law, layers=layers, base=k)
        tok = (lambda z: x + self.pool_up(z)[loc].to(x.dtype)) if self.pool_skip else (lambda z: z[loc])
        out.update({i: tok(z) for i, z in lo.items()})
        return tok(xl), out, xl, loc, (lt, lat, law)

    def _decode_varlen(self, qm, xv, qpos, kpos, cm):
        """The grouped-cross decoder with even, unpadded time groups. ``qpos`` /
        ``kpos`` are (t_phys, ang_t, ang_w) of the queries / keys; ``cm`` the
        queries' AdaLN conditioning or None."""
        (qt, qat, qaw), (kt, kat, kaw) = qpos, kpos
        Tq, Tk, dev = qm.shape[0], xv.shape[0], xv.device
        oq = torch.argsort(qt.double()); okv = torch.argsort(kt.double())
        nb = max(1, min((max(Tq, Tk) + self.gd - 1) // self.gd, Tq, Tk))
        cuq, cuk = _cu_even(Tq, nb, dev), _cu_even(Tk, nb, dev)
        mq, mk = (Tq + nb - 1) // nb, (Tk + nb - 1) // nb
        qcos, qsin = rope_tables(qat[oq], qaw[oq], qm.dtype)
        kcos, ksin = rope_tables(kat[okv], kaw[okv], xv.dtype)
        cm = None if cm is None else cm[oq]
        cross_blk, fused = _ckpt(_blocks(self)[1], self), self._fused()
        qp, kvp = _permute(qm, oq, Tq), _permute(xv, okv, Tk)
        for blk in self.dec:
            qp = cross_blk(blk, qp, kvp, qcos, qsin, kcos, ksin, cuq, cuk, mq, mk, cm, fused)
        return _permute(qp, _inverse(oq), Tq)

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
        # ONE host sync (the visible count) instead of one per boolean-mask index
        # below: a stable argsort of the mask lists visible rows then masked rows,
        # each ascending -- the same rows, in the same order, as nonzero().
        n_vis = int((~tok_mask).sum())
        order = torch.argsort(tok_mask.to(torch.uint8), stable=True)
        vis_idx, mask_idx = order[:n_vis], order[n_vis:]
        if masked_only and self.training and isinstance(self.dec_frac, (tuple, list)):
            # Per-band partial reconstruction: each masked row kept with its band's
            # probability (FMModel.dec_weights supplies the 1/p loss weights). The
            # count is random, so this costs one host sync; always at least one row.
            p = torch.as_tensor(self.dec_frac, dtype=torch.float32, device=order.device)
            keep = torch.rand(mask_idx.numel(), device=order.device) < p[B["band_id"][mask_idx]]
            keep[0] |= ~keep.any()
            mask_idx = mask_idx[keep]
        elif masked_only and self.training and self.dec_frac < 1.0:
            # Partial reconstruction (CrossMAE): decode a random subset of the
            # masked tokens. The encoder still sees only the visible ones, so its
            # task is unchanged; the decoder -- a third of the step -- shrinks.
            n_m = order.numel() - n_vis                          # host ints: no sync
            keep = torch.randperm(n_m, device=order.device)[:max(1, round(n_m * self.dec_frac))]
            mask_idx = mask_idx[keep.sort().values]
        if masked_only and self.training and self.vis_frac > 0:
            # Visible tokens also decoded (FMModel vis_frac): a random subset joins
            # the queries after the masked rows. n_vis is a host int: no sync.
            k = max(1, round(n_vis * self.vis_frac))
            sel = vis_idx[torch.randperm(n_vis, device=order.device)[:k].sort().values]
            mask_idx = torch.cat([mask_idx, sel])
        c = self._cond(B) if self.cond == "adaln" else None
        atv, awv = at[vis_idx], aw[vis_idx]
        kpos = (B["t_phys"][vis_idx], atv, awv)
        vis_feat = None                                  # per visible token, when keys are not
        if self.band_pool is not None:
            x0 = self._emb(B, vis_idx).to(torch.float32 if self.fp32_stream else self.embed.weight.dtype)
            xv, _, xl, vloc, lpos = self._encode_pooled(B, vis_idx, x0, atv, awv, ())
            if not self.pool_skip:                       # keys: the locations themselves
                xv, kpos, vis_feat = xl, lpos, vloc
        else:
            xv, _ = self._encode(B, vis_idx, atv, awv, None if c is None else c[vis_idx])

        # CrossMAE decoder (grouped, drift-time): masked queries x-attend visible
        qm = self.mask_tok.expand(mask_idx.numel(), self.d)
        if c is None:                           # with AdaLN, identity enters there instead
            if self.film is not None:
                g_, b_ = self.film(B["band_id"][mask_idx], B["plane_id"][mask_idx], B["wirefeat"][mask_idx]); qm = g_ * qm + b_
            qm = qm + self.band_emb(B["band_id"][mask_idx]) + self.plane_emb(B["plane_id"][mask_idx])
        qm = qm.to(xv.dtype)
        if self.dec_embed is not None:          # a narrower decoder reads the encoder through one map
            qm, xv = self.dec_embed(qm), self.dec_embed(xv)
        if vis_feat is not None:                # pool_skip=False: a visible token's feature is its location's
            vis_feat = xv[vis_feat]
        if self.varlen:
            qm = self._decode_varlen(qm, xv, (B["t_phys"][mask_idx], at[mask_idx], aw[mask_idx]), kpos,
                                     None if c is None else c[mask_idx])
            if masked_only:
                return self.dec_norm(qm), mask_idx
            x = torch.zeros(B["inp"].shape[0], self.d_dec, dtype=xv.dtype, device=xv.device)
            xv_rows = xv if vis_feat is None else vis_feat
            return self.dec_norm(x.index_copy(0, vis_idx, xv_rows).index_copy(0, mask_idx, qm))
        Tq, Tk = qm.shape[0], xv.shape[0]
        oq = torch.argsort(B["t_phys"][mask_idx].double()); okv = torch.argsort(B["t_phys"][vis_idx].double())
        nb = (max(Tq, Tk) + self.gd - 1) // self.gd
        gq, gk = (Tq + nb - 1) // nb, (Tk + nb - 1) // nb
        iq, ik = _pad_idx(oq, Tq, nb * gq), _pad_idx(okv, Tk, nb * gk)
        # Decoder ALWAYS keeps axial (wire) RoPE: it needs the wire address to know which wire it
        # reconstructs -- dropping it froze recon (var_expl ~2%, same as wire_rope=0). rope_split
        # only affects the ENCODER cross-plane (drift-order) layers, via `dw` in _layouts.
        qcos, qsin = rope_tables(at[mask_idx][iq], aw[mask_idx][iq], qm.dtype)
        kcos, ksin = rope_tables(atv[ik], awv[ik], xv.dtype)
        cm = None if c is None else c[mask_idx][iq]
        km = ((torch.arange(nb * gk, device=xv.device) < Tk).view(nb, 1, 1, gk)
              if self.pad_mask else None)
        cross_blk = _ckpt(_blocks(self)[1], self)
        qp, kvp = _permute(qm, iq, Tq), _permute(xv, ik, Tk)
        for blk in self.dec:
            qp = cross_blk(blk, qp, kvp, qcos, qsin, kcos, ksin, nb, gq, gk, cm, km)
        qm = _permute(qp, _inverse(oq), Tq)

        if masked_only:
            return self.dec_norm(qm), mask_idx
        x = torch.zeros(B["inp"].shape[0], self.d_dec, dtype=xv.dtype, device=xv.device)
        x = x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm)
        return self.dec_norm(x)
