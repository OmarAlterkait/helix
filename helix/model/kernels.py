"""Triton kernels for the serial model's hot path. Imported lazily (only when a
model is built with ``fused_qk``), so neither CPU tests nor the login node need
Triton.

``normrope``: QK RMSNorm + helix's interleaved RoPE ((2i, 2i+1) rotated by angle
i, as ``apply_rope_tables``) in one kernel per normalised part, forward and
backward. It reads straight out of a fused projection output X (T, P*d):

  mode "qkv": parts (q, k, v) -> q, k normalised+rotated, v passed through
  mode "kv":  parts (k, v)    -> k normalised+rotated, v passed through
  mode "q":   part  (q,)      -> q normalised+rotated

Outputs are (T, H, 64). The backward recomputes the norm from X (alive anyway for
v), so nothing extra is saved, and writes all of dX in one buffer. Replaces the
four-to-six inductor kernels per block the unfused path compiles to; matches it
to ~1e-5 relative (tests/test_serial.py).
"""
import torch, triton, triton.language as tl

HD = 64


@triton.jit
def _fwd(X, COS, SIN, W, O, R, H, LDX, COL0, eps, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR); rm = rows < R
    t = rows // H; h = rows % H; j64 = tl.arange(0, 64); j32 = tl.arange(0, 32)
    x = tl.load(X + (t * LDX + COL0 + h * 64)[:, None] + j64[None, :], mask=rm[:, None], other=0.).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 1) / 64 + eps)[:, None]
    n = x * r * tl.load(W + j64).to(tl.float32)[None, :]
    a, b = tl.split(tl.reshape(n, (BR, 32, 2)))
    c = tl.load(COS + t[:, None] * 32 + j32[None, :], mask=rm[:, None], other=0.).to(tl.float32)
    s = tl.load(SIN + t[:, None] * 32 + j32[None, :], mask=rm[:, None], other=0.).to(tl.float32)
    o = tl.reshape(tl.join(a * c - b * s, b * c + a * s), (BR, 64))
    tl.store(O + rows[:, None] * 64 + j64[None, :], o.to(O.dtype.element_ty), mask=rm[:, None])


@triton.jit
def _bwd(X, COS, SIN, W, GO, DX, DWP, R, H, LDX, COL0, eps, BR: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR); rm = rows < R
    t = rows // H; h = rows % H; j64 = tl.arange(0, 64); j32 = tl.arange(0, 32)
    off = (t * LDX + COL0 + h * 64)[:, None] + j64[None, :]
    x = tl.load(X + off, mask=rm[:, None], other=0.).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 1) / 64 + eps)[:, None]
    ga, gb = tl.split(tl.reshape(tl.load(GO + rows[:, None] * 64 + j64[None, :], mask=rm[:, None], other=0.).to(tl.float32), (BR, 32, 2)))
    c = tl.load(COS + t[:, None] * 32 + j32[None, :], mask=rm[:, None], other=0.).to(tl.float32)
    s = tl.load(SIN + t[:, None] * 32 + j32[None, :], mask=rm[:, None], other=0.).to(tl.float32)
    gn = tl.reshape(tl.join(ga * c + gb * s, gb * c - ga * s), (BR, 64))      # rotation transposed
    w = tl.load(W + j64).to(tl.float32)[None, :]
    tl.store(DWP + pid * 64 + j64, tl.sum(gn * x * r, 0))
    g = gn * w
    m = (tl.sum(g * x, 1) / 64)[:, None]
    tl.store(DX + off, (r * (g - x * r * r * m)).to(DX.dtype.element_ty), mask=rm[:, None])


BR = 32
_PARTS = {"qkv": ((0, 1), (2,)), "kv": ((0,), (1,)), "q": ((0,), ())}


class NormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, cos, sin, wa, wb, mode, eps):
        T, C = X.shape; normed, passed = _PARTS[mode]; P = len(normed) + len(passed); d = C // P; H = d // HD; R = T * H
        X = X.contiguous(); cos = cos.reshape(T, 32).contiguous(); sin = sin.reshape(T, 32).contiguous()
        outs = []
        for i, p in enumerate(normed):
            o = torch.empty(T, H, HD, device=X.device, dtype=X.dtype)
            _fwd[(triton.cdiv(R, BR),)](X, cos, sin, (wa, wb)[i], o, R, H, C, p * d, eps, BR=BR)
            outs.append(o)
        for p in passed:
            outs.append(X[:, p * d:(p + 1) * d].view(T, H, HD))
        ctx.save_for_backward(X, cos, sin, wa, wb); ctx.mode, ctx.eps = mode, eps
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grads):
        X, cos, sin, wa, wb = ctx.saved_tensors; mode, eps = ctx.mode, ctx.eps
        T, C = X.shape; normed, passed = _PARTS[mode]; P = len(normed) + len(passed); d = C // P; H = d // HD; R = T * H
        n = triton.cdiv(R, BR); dX = torch.empty_like(X); dw = []
        for i, p in enumerate(normed):
            g = grads[i]
            g = torch.zeros(T, H, HD, device=X.device, dtype=X.dtype) if g is None else g.contiguous()
            dwp = torch.empty(n, HD, device=X.device, dtype=torch.float32)
            _bwd[(n,)](X, cos, sin, (wa, wb)[i], g, dX, dwp, R, H, C, p * d, eps, BR=BR)
            dw.append(dwp.sum(0).to((wa, wb)[i].dtype))
        for k, p in enumerate(passed):
            g = grads[len(normed) + k]
            if g is None: dX[:, p * d:(p + 1) * d].zero_()
            else: dX[:, p * d:(p + 1) * d].copy_(g.reshape(T, d))
        dwa = dw[0] if len(dw) > 0 else None; dwb = dw[1] if len(dw) > 1 else None
        return dX, None, None, dwa, dwb, None, None


def normrope(X, cos, sin, wa, wb=None, mode="qkv", eps=1e-6):
    """See the module docstring. ``wb`` defaults to ``wa`` (single-part modes)."""
    return NormRope.apply(X, cos, sin, wa, wb if wb is not None else wa, mode, eps)


