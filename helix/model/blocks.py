"""Building blocks of the coefficient FM: their parameters and their forward passes.

Axial RoPE on (physical_time, wire), a FiLM conditioner on (band, plane, wire), and the
parameters of the self / cross attention blocks; then the grouped attention that runs them.

GROUPED ATTENTION
=================
Each encoder block attends within groups of ``g`` consecutive tokens taken along
one of four orders (plane x time, time, plane x wire x time, time shifted by
half a group); the decoder's masked queries cross-attend visible tokens in
drift-time groups. A layout's groups are its consecutive g-token chunks with the
LAST ONE SHORTER, run through flash varlen attention with cumulative group
boundaries (``cu``) -- no padding, so no token is attended twice. The decoder's
time groups are split evenly.

The residual stream is CARRIED in layout order. LayerNorm, qkv, the projection,
the MLP and both residual adds are row-wise, so only attention cares about
order, and a block costs one permutation composing the previous layout with this
one instead of gathering q, k, v and scattering the output back. The decoder's
query and key orders are the same in every block, so it is permuted once before
the stack and back once after (tests/test_attention.py keeps the per-block
formulation as the reference).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_angles(pos, dim, lam_min=2.0, lam_max=10000.0):
    """Axial RoPE angles with geometric wavelengths in [lam_min, lam_max]
    (n = dim//4 frequencies). Set lam_min ~ 2x the finest position spacing and
    lam_max ~ the full coordinate span, PER AXIS: a single base=10000 would
    spread wavelengths over [6.3, 47000] on both axes and waste ~55% of the WIRE
    dims (span 1904, spacing 16 -> useful band [32,1904]) and ~28% of the TIME
    dims (span 4336)."""
    n = dim // 4
    k = torch.arange(n, device=pos.device).float() / max(n - 1, 1)
    inv = (2 * math.pi) / (lam_min * (lam_max / lam_min) ** k)
    return pos[:, None].float() * inv[None, :]                 # (T, half/2)


def rope_tables(ang_t, ang_w, dtype=None):
    """-> (cos, sin), each ``(T, 1, hd/2)``: the time half, then the wire half.

    Built once per token layout rather than inside every block. ``ang_w=None``
    leaves the wire half unrotated (cos 1, sin 0) rather than reallocating it to
    time -- a known defect (REVIEW_FIELD §1.3 at git tag archive-2026-11; TODO.md), kept because changing it
    changes the model."""
    if ang_w is None:
        c = torch.cat([torch.cos(ang_t), torch.ones_like(ang_t)], -1)
        s = torch.cat([torch.sin(ang_t), torch.zeros_like(ang_t)], -1)
    else:
        c = torch.cat([torch.cos(ang_t), torch.cos(ang_w)], -1)
        s = torch.cat([torch.sin(ang_t), torch.sin(ang_w)], -1)
    if dtype is not None:
        c, s = c.to(dtype), s.to(dtype)
    return c[:, None, :].contiguous(), s[:, None, :].contiguous()


def apply_rope_tables(x, cos, sin):
    """x: (T, H, hd) rotated pairwise, (2i, 2i+1) by angle i, from precomputed
    tables. A (T, H, hd/2, 2) view makes this a few contiguous kernels."""
    T, h, hd = x.shape
    v = x.view(T, h, hd // 2, 2)
    a, b = v[..., 0], v[..., 1]
    return torch.stack([a * cos - b * sin, b * cos + a * sin], -1).view(T, h, hd)


# ---- response conditioning -------------------------------------------------
class ResponseFiLM(nn.Module):
    """gamma,beta from [band_emb, plane_emb, MLP(wire feature)]. Conditions the
    token on the response chain (scale, plane-type bipolar/unipolar, wire geom)."""

    def __init__(self, d, n_band, n_plane, de=32):
        super().__init__()
        self.band = nn.Embedding(n_band, de)
        self.plane = nn.Embedding(n_plane, de)
        self.wire = nn.Sequential(nn.Linear(1, de), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(3 * de, 2 * d))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)  # start = identity

    def forward(self, band, plane, wirefeat):
        parts = [self.band(band), self.plane(plane), self.wire(wirefeat)]
        g, b = self.mlp(torch.cat(parts, -1)).chunk(2, -1)
        return 1 + g, b


# ---- transformer blocks ----------------------------------------------------
def _qk_norms(hd, on):
    """QK-norm: an RMSNorm over each head's query and key, before RoPE (as OLMo 2
    and Qwen3 place it). It bounds the attention logits, which otherwise grow with
    the q/k projections: the qkv biases of the late encoder blocks grew
    monotonically over a 47k-step d768 run (7 -> 18) and one seed went NaN at
    19.4k. Off by default: it adds parameters and changes the numerics."""
    if not on:
        return None, None
    return nn.RMSNorm(hd, eps=1e-6), nn.RMSNorm(hd, eps=1e-6)


def qk_normed(blk, q, k):
    """(q, k) through the block's QK-norm, or unchanged when it has none."""
    if blk.qn is None:
        return q, k
    return blk.qn(q), blk.kn(k)


class Block(nn.Module):
    """Pre-LN self-attention block: the parameters only. The forward is
    :func:`self_block`, which runs it over grouped attention."""

    def __init__(self, d, heads, ffn_mult=4, qk_norm=False):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qn, self.kn = _qk_norms(self.hd, qk_norm)
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))


# ---- CrossMAE decoder block ------------------------------------------------
class CrossBlock(nn.Module):
    """CrossMAE decoder block: masked-token queries cross-attend the VISIBLE
    encoded tokens (no mask-mask self-attention), with RoPE on both the query
    and the key positions. The parameters only; the forward is
    :func:`cross_block`."""

    def __init__(self, d, heads, ffn_mult=4, qk_norm=False):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qn, self.kn = _qk_norms(self.hd, qk_norm)
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))


# ---- grouped attention: the blocks' forward passes -------------------------------------------

def _inverse(order):
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return inv


class _Permute(torch.autograd.Function):
    """``x[idx]`` for a permutation ``idx`` -- every reorder in this model.

    Autograd's backward for a gather is a general scatter-add: it sorts the
    indices and accumulates, 8-11 ms of a 96-118 ms d768 step (7-11 %). A
    permutation's backward is a gather by its inverse."""

    @staticmethod
    def forward(ctx, x, idx):
        ctx.save_for_backward(_inverse(idx))
        return x[idx]

    @staticmethod
    def backward(ctx, gy):
        (inv,) = ctx.saved_tensors
        return gy[inv], None


def _permute(x, idx):
    return _Permute.apply(x, idx) if torch.is_grad_enabled() and x.requires_grad else x[idx]


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


def self_block(blk, x, cos, sin, cu, maxlen, fused=False):
    """Block over a stream in layout order whose groups are ``cu``."""
    T, d = x.shape
    q, k, v = _qk_rope(blk, blk.qkv(blk.n1(x)), cos, sin, "qkv", fused)
    x = x + blk.proj(_vl_attn(q, k, v, cu, cu, maxlen, maxlen).reshape(T, d))
    return x + blk.mlp(blk.n2(x))


def cross_block(blk, q, kv, qcos, qsin, kcos, ksin, cuq, cuk, mq, mk, fused=False):
    """CrossBlock with query groups ``cuq`` attending key groups ``cuk``."""
    Tq = q.shape[0]
    (qh,) = _qk_rope(blk, blk.q(blk.nq(q)), qcos, qsin, "q", fused)
    kh, vh = _qk_rope(blk, blk.kv(blk.nk(kv)), kcos, ksin, "kv", fused)
    q = q + blk.proj(_vl_attn(qh, kh, vh, cuq, cuk, mq, mk).reshape(Tq, blk.h * blk.hd))
    return q + blk.mlp(blk.n2(q))


def _cu_chunks(T, g, device):
    """[0, g, 2g, ..., T] built on device (a host-built tensor is a copy + sync)."""
    return torch.arange(0, T + g, g, device=device).clamp_(max=T).to(torch.int32)


def _cu_even(T, nb, device):
    """nb near-equal groups over T rows: no group is empty while T >= nb."""
    return ((torch.arange(nb + 1, device=device) * T) // nb).to(torch.int32)


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
    if not _COMPILED:                                      # process-wide, once
        import torch._dynamo as _dyn
        _dyn.config.optimize_ddp = False
        _detach_dynamo_finalizers_at_exit()
        for knob in ("recompile_limit", "cache_size_limit"):
            if hasattr(_dyn.config, knob):
                setattr(_dyn.config, knob, max(64, getattr(_dyn.config, knob)))
        _COMPILED["self"] = torch.compile(self_block, dynamic=True)
        _COMPILED["cross"] = torch.compile(cross_block, dynamic=True)
    return _COMPILED["self"], _COMPILED["cross"]
