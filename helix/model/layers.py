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


def apply_rope(x, ang_t, ang_w):
    """x: (T, H, hd). First half of hd rotated by time, second half by wire."""
    h2 = x.shape[-1] // 2

    def rot(v, ang):
        c = torch.cos(ang)[:, None, :].repeat_interleave(2, -1)
        s = torch.sin(ang)[:, None, :].repeat_interleave(2, -1)
        v2 = torch.stack([-v[..., 1::2], v[..., 0::2]], -1).reshape_as(v)
        return v * c + v2 * s
    xt = rot(x[..., :h2], ang_t)
    xw = rot(x[..., h2:], ang_w) if ang_w is not None else x[..., h2:]
    return torch.cat([xt, xw], -1)


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

    def __init__(self, d, heads, ffn_mult=4, attn_scale=None):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.attn_scale = attn_scale          # muP: 1/head_dim (else None => SDPA default)
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, q, kv, qa_t, qa_w, ka_t, ka_w):
        Tq, Tk = q.shape[0], kv.shape[0]
        qh = apply_rope(self.q(self.nq(q)).view(Tq, self.h, self.hd), qa_t, qa_w)
        k, v = self.kv(self.nk(kv)).chunk(2, -1)
        kh = apply_rope(k.view(Tk, self.h, self.hd), ka_t, ka_w)
        vh = v.view(Tk, self.h, self.hd)
        o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kh.transpose(0, 1)[None],
                                           vh.transpose(0, 1)[None], scale=self.attn_scale)[0].transpose(0, 1)
        q = q + self.proj(o.reshape(Tq, self.h * self.hd))
        return q + self.mlp(self.n2(q))
