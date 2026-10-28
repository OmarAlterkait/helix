"""Building blocks of the coefficient FM.

Axial RoPE on (physical_time, wire), a FiLM conditioner on (band, plane, wire),
and the parameters of the self / cross attention blocks (their forward passes
are in :mod:`helix.model.serial`).
"""
import math
import torch
import torch.nn as nn


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
    time -- a known defect (docs/archive/REVIEW_FIELD.md §1.3), kept because changing it
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
    ``helix.model.serial.self_block``, which runs it over grouped attention."""

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
    ``helix.model.serial.cross_block``."""

    def __init__(self, d, heads, ffn_mult=4, qk_norm=False):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qn, self.kn = _qk_norms(self.hd, qk_norm)
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))
