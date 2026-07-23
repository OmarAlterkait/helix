"""Perceiver-IO / Senseiver-style model for sparse-token -> dense regression.

cross-in (M latents read the N input tokens) -> DEEP self-attn stack on the M
latents (depth = the lever, cost independent of N) -> output queries (one per
target cell, position+band only) cross-attend latents -> per-cell feature z.
forward_feat(B) returns (n_cells, d), drop-in for the flow/value head.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


# ---- axial RoPE (ported from model.py) — for the decode cross-attention test ----
def rope_angles(pos, dim, lam_min=2.0, lam_max=10000.0):
    n = dim // 4
    k = torch.arange(n, device=pos.device).float() / max(n - 1, 1)
    inv = (2 * math.pi) / (lam_min * (lam_max / lam_min) ** k)
    return pos[:, None].float() * inv[None, :]        # (T, dim//4)


def apply_rope(x, ang_t, ang_w):                      # x: (T, H, hd); rotate first half by time, second by wire
    h2 = x.shape[-1] // 2

    def rot(v, ang):
        c = torch.cos(ang)[:, None, :].repeat_interleave(2, -1)
        s = torch.sin(ang)[:, None, :].repeat_interleave(2, -1)
        v2 = torch.stack([-v[..., 1::2], v[..., 0::2]], -1).reshape_as(v)
        return v * c + v2 * s
    return torch.cat([rot(x[..., :h2], ang_t), rot(x[..., h2:], ang_w)], -1)


def _mha(q, k, v, h, qrope=None, krope=None):
    """Multi-head attention; ANY leading batch dims. MUST be 4-D for flash (a 3-D
    single-event input falls back to MATH -> OOM), so add a batch dim for 3-D.
    Optional axial RoPE: qrope=(ang_t,ang_w) for query positions, krope for keys
    (used by the decode cross-attention to test RELATIVE position addressing)."""
    qh = q.unflatten(-1, (h, -1)); kh = k.unflatten(-1, (h, -1)); vh = v.unflatten(-1, (h, -1))
    if qrope is not None:
        qh = apply_rope(qh, *qrope)                   # (Tq, h, hd) rotated by query positions
    if krope is not None:
        kh = apply_rope(kh, *krope)
    qr = qh.transpose(-3, -2); kr = kh.transpose(-3, -2); vr = vh.transpose(-3, -2)   # (..., h, T, hd)
    three = qr.dim() == 3
    if three:
        qr, kr, vr = qr[None], kr[None], vr[None]     # -> (1, h, Tq, hd) for flash
    o = F.scaled_dot_product_attention(qr, kr, vr)
    if three:
        o = o[0]
    return o.transpose(-3, -2).flatten(-2)            # (..., Tq, D)


class Self(nn.Module):
    def __init__(self, d, h, mult=4):
        super().__init__(); self.h = h; self.n = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3*d)
        self.p = nn.Linear(d, d); self.n2 = nn.LayerNorm(d)
        self.m = nn.Sequential(nn.Linear(d, mult*d), nn.GELU(), nn.Linear(mult*d, d))

    def forward(self, x):
        q, k, v = self.qkv(self.n(x)).chunk(3, -1)
        x = x + self.p(_mha(q, k, v, self.h)); return x + self.m(self.n2(x))


class Cross(nn.Module):
    def __init__(self, d, h, mult=4):
        super().__init__(); self.h = h; self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2*d); self.p = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.m = nn.Sequential(nn.Linear(d, mult*d), nn.GELU(), nn.Linear(mult*d, d))

    def forward(self, q, kv, qrope=None, krope=None):
        k, v = self.kv(self.nk(kv)).chunk(2, -1)
        q = q + self.p(_mha(self.q(self.nq(q)), k, v, self.h, qrope=qrope, krope=krope))
        return q + self.m(self.n2(q))


class PosEnc(nn.Module):
    def __init__(self, d, tmax=4336., wmax=2048.):
        super().__init__(); nf = d // 4
        self.register_buffer("tf", torch.exp(torch.linspace(math.log(2), math.log(tmax), nf)))
        self.register_buffer("wf", torch.exp(torch.linspace(math.log(8), math.log(wmax), nf)))
        self.proj = nn.Linear(4 * nf, d)

    def forward(self, t, w):
        at = t[:, None] / self.tf; aw = w[:, None] / self.wf
        return self.proj(torch.cat([at.sin(), at.cos(), aw.sin(), aw.cos()], -1))


class PerceiverDeconv(nn.Module):
    def __init__(self, n_slot, n_band, n_plane, d=512, M=2048, depth=24, heads=8, n_wirefeat=1, nll=False):
        super().__init__(); self.n_slot = n_slot; self.M = M; self.d = d; self.nll = nll
        self.embed = nn.Linear(2 * n_slot, d)
        self.pos = PosEnc(d); self.band = nn.Embedding(n_band, d); self.plane = nn.Embedding(n_plane, d)
        self.latents = nn.Parameter(torch.randn(M, d) * 0.02)
        self.cin = Cross(d, heads); self.deep = nn.Sequential(*(Self(d, heads) for _ in range(depth))); self.cout = Cross(d, heads)
        self.norm = nn.LayerNorm(d)

    def _cond(self, B):
        return self.pos(B["t_phys"], B["wire_pos"]) + self.band(B["band_id"]) + self.plane(B["plane_id"])

    def _xtok(self, B):                                # input token: noisy value + position/band/plane
        return self.embed(torch.cat([B["inp"], B["occ"]], -1)) + self._cond(B)

    def forward_feat(self, B, mask=None):              # mask unused (all-visible); kept for API parity
        x = self._xtok(B)                                              # input tokens carry values
        lat = self.deep(self.cin(self.latents, x))                     # M latents read N inputs, deep stack
        z = self.cout(x, lat)                                          # decode query = input token (sees its own value) + latent context
        return self.norm(z)                                            # (n_cells, d)

    def _crossin(self, B):
        return self.cin(self.latents, self._xtok(B))

    def _decode(self, B, lat_i):
        return self.norm(self.cout(self._xtok(B), lat_i))            # query carries the cell's noisy input value

    def forward_batch(self, Bs, ckpt=True):            # Bs: list of B event dicts (variable N each)
        from torch.utils.checkpoint import checkpoint
        ci = (lambda B: checkpoint(self._crossin, B, use_reentrant=False)) if ckpt else self._crossin
        dc = (lambda B, l: checkpoint(self._decode, B, l, use_reentrant=False)) if ckpt else self._decode
        lat = torch.stack([ci(B) for B in Bs])                       # (B, M, d); cross-in checkpointed
        lat = self.deep(lat)                                         # BATCHED deep stack over (B, M, d)
        return [dc(B, lat[i]) for i, B in enumerate(Bs)]             # decode checkpointed, per event
