"""Attention primitives for the coefficient FM — extracted verbatim from the
research tree (``coeff_foundation_model/fm/model.py`` lines 21-137).

Axial RoPE on (physical_time, wire), a FiLM conditioner on (band, plane, wire),
and the self / cross attention blocks. No behavioural change from research.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_angles(pos, dim, lam_min=2.0, lam_max=10000.0):
    """Axial RoPE angles with geometric wavelengths in [lam_min, lam_max]
    (n = dim//4 frequencies). Set lam_min ~ 2x the finest position spacing and
    lam_max ~ the full coordinate span, PER AXIS. The old base=10000 spread
    wavelengths over [6.3, 47000] on both axes -> wasted ~55% of the WIRE dims
    (span 1904, spacing 16 -> useful band [32,1904]) and ~28% of the TIME dims
    (span 4336)."""
    n = dim // 4
    k = torch.arange(n, device=pos.device).float() / max(n - 1, 1)
    inv = (2 * math.pi) / (lam_min * (lam_max / lam_min) ** k)
    return pos[:, None].float() * inv[None, :]                 # (T, half/2)


def rope_tables(ang_t, ang_w, dtype=None):
    """-> (cos, sin), each ``(T, 1, hd/2)``: the time half, then the wire half.

    Built once per token layout rather than inside every block. ``ang_w=None``
    leaves the wire half unrotated (cos 1, sin 0) rather than reallocating it to
    time -- a known defect (docs/REVIEW_FIELD.md §1.3), kept because changing it
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


def apply_rope(x, ang_t, ang_w):
    """x: (T, H, hd). First half of hd rotated by time, second half by wire."""
    return apply_rope_tables(x, *rope_tables(ang_t, ang_w))


# ---- response conditioning -------------------------------------------------
class ResponseFiLM(nn.Module):
    """gamma,beta from [band_emb, plane_emb, MLP(wire features)]. Conditions the
    token on the response chain (scale, plane-type bipolar/unipolar, wire geom)."""

    def __init__(self, d, n_band, n_plane, n_wirefeat, de=32, use=("band", "plane", "wire")):
        super().__init__()
        self.use = set(use)
        self.band = nn.Embedding(n_band, de)
        self.plane = nn.Embedding(n_plane, de)
        self.wire = nn.Sequential(nn.Linear(n_wirefeat, de), nn.GELU())
        cin = de * len(self.use)
        self.mlp = nn.Sequential(nn.Linear(cin, 2 * d))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)  # start = identity

    def forward(self, band, plane, wirefeat):
        parts = []
        if "band" in self.use:  parts.append(self.band(band))
        if "plane" in self.use: parts.append(self.plane(plane))
        if "wire" in self.use:  parts.append(self.wire(wirefeat))
        g, b = self.mlp(torch.cat(parts, -1)).chunk(2, -1)
        return 1 + g, b


# ---- transformer block (pre-LN, SDPA full attention with RoPE) -------------
class Block(nn.Module):
    def __init__(self, d, heads, ffn_mult=4, adaln=False, attn_scale=None):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.attn_scale = attn_scale          # muP: 1/head_dim (else None => SDPA default 1/sqrt(hd))
        self.adaln = adaln
        self.n1 = nn.LayerNorm(d, elementwise_affine=not adaln)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=not adaln)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))
        if adaln:                              # AdaLN-Zero: per-token scale/shift/gate, zero-init = identity
            self.ada = nn.Linear(d, 6 * d)
            nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, ang_t, ang_w, c=None):
        T, d = x.shape
        if self.adaln:
            sa, ba, ga, sm, bm, gm = self.ada(c).chunk(6, -1)    # each (T, d)
            h = self.n1(x) * (1 + sa) + ba
        else:
            h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        q = apply_rope(q.view(T, self.h, self.hd), ang_t, ang_w)
        k = apply_rope(k.view(T, self.h, self.hd), ang_t, ang_w)
        v = v.view(T, self.h, self.hd)
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                           v.transpose(0, 1)[None], scale=self.attn_scale)[0].transpose(0, 1)
        ao = self.proj(o.reshape(T, d))
        x = x + (ga * ao if self.adaln else ao)
        if self.adaln:
            return x + gm * self.mlp(self.n2(x) * (1 + sm) + bm)
        return x + self.mlp(self.n2(x))


# ---- CrossMAE decoder block (cheaper decoder) ------------------------------
class CrossBlock(nn.Module):
    """CrossMAE decoder: masked-token queries cross-attend the VISIBLE encoded
    tokens (no mask-mask self-attention) -> O(N_mask * N_vis), far cheaper than the
    full-self-attention decoder over all N. Axial RoPE on BOTH query (masked) and
    key (visible) positions, so it keeps the relative-position addressing.
    NOTE: queries see the FULL visible set (not a latent summary) — unlike the
    Perceiver bottleneck that capped masked prediction at ~33%."""

    def __init__(self, d, heads, ffn_mult=4, attn_scale=None, adaln=False):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.attn_scale = attn_scale          # muP: 1/head_dim (else None => SDPA default)
        self.adaln = adaln
        # AdaLN modulates the QUERY stream only. Keys/values are encoder output, already
        # conditioned encoder-side; re-modulating them here would double-apply.
        self.nq = nn.LayerNorm(d, elementwise_affine=not adaln); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=not adaln)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))
        if adaln:                              # AdaLN-Zero, mirroring Block: zero-init = identity
            self.ada = nn.Linear(d, 6 * d)
            nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, q, kv, qa_t, qa_w, ka_t, ka_w, c=None):
        Tq, Tk = q.shape[0], kv.shape[0]
        if self.adaln:
            sa, ba, ga, sm, bm, gm = self.ada(c).chunk(6, -1)     # each (Tq, d), query-side
            hq = self.nq(q) * (1 + sa) + ba
        else:
            hq = self.nq(q)
        qh = apply_rope(self.q(hq).view(Tq, self.h, self.hd), qa_t, qa_w)
        k, v = self.kv(self.nk(kv)).chunk(2, -1)
        kh = apply_rope(k.view(Tk, self.h, self.hd), ka_t, ka_w)
        vh = v.view(Tk, self.h, self.hd)
        o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kh.transpose(0, 1)[None],
                                           vh.transpose(0, 1)[None], scale=self.attn_scale)[0].transpose(0, 1)
        ao = self.proj(o.reshape(Tq, self.h * self.hd))
        q = q + (ga * ao if self.adaln else ao)
        if self.adaln:
            return q + gm * self.mlp(self.n2(q) * (1 + sm) + bm)
        return q + self.mlp(self.n2(q))
