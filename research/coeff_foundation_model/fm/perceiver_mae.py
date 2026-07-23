"""Perceiver-IO masked autoencoder over wavelet-coeff tokens.

Encode the VISIBLE (1-mask) tokens -> M latents -> deep stack; decode ALL token
positions (position/band query + a learned mask token on masked slots) -> recon.
Self-supervised: reconstruct the MASKED tokens' coefficients. Tests whether the
latent-bottleneck architecture learns representations (cf. FMModel MAE 77-85%
var-explained) and whether a flow recon head keeps var-ratio (the deconv decode
fed the input value, collapsing the flow; the MAE decode is position-only, so the
conditional is genuinely stochastic).

Reuses Cross/Self/PosEnc from perceiver_model. forward_batch mirrors the deconv
trainer: encode/decode looped per event (variable N), deep stack batched (B,M,d).
"""
import torch, torch.nn as nn
from perceiver_model import Cross, Self, PosEnc, rope_angles


def make_mask(n, ratio, device, gen=None):
    """Bool (n,), True = masked (hidden from the encoder)."""
    k = int(n * ratio)
    perm = torch.randperm(n, generator=gen, device=device)
    m = torch.zeros(n, dtype=torch.bool, device=device); m[perm[:k]] = True
    return m


class PerceiverMAE(nn.Module):
    def __init__(self, n_slot, n_band, n_plane, d=512, M=2048, depth=24, heads=8,
                 parallel_decode=False, rope_cout2=False, lam_t=(8.0, 4336.0), lam_w=(32.0, 2048.0)):
        super().__init__(); self.n_slot = n_slot; self.M = M; self.d = d; self.heads = heads
        self.parallel_decode = parallel_decode    # query cout/cout2 with the CLEAN address (vs sequential)
        self.rope_cout2 = rope_cout2              # axial RoPE (RELATIVE pos) in the visible-token cross-attn
        self.lam_t, self.lam_w = lam_t, lam_w
        self.embed = nn.Linear(2 * n_slot, d)
        self.pos = PosEnc(d); self.band = nn.Embedding(n_band, d); self.plane = nn.Embedding(n_plane, d)
        self.latents = nn.Parameter(torch.randn(M, d) * 0.02)
        self.mask_tok = nn.Parameter(torch.zeros(d))                  # added to masked query positions
        self.cin = Cross(d, heads); self.deep = nn.Sequential(*(Self(d, heads) for _ in range(depth)))
        self.cout = Cross(d, heads)                                   # decode: global context from latents
        self.cout2 = Cross(d, heads)                                  # CrossMAE: position-resolved from VISIBLE tokens
        self.norm = nn.LayerNorm(d)

    def _cond(self, B):
        return self.pos(B["t_phys"], B["wire_pos"]) + self.band(B["band_id"]) + self.plane(B["plane_id"])

    def _vis(self, B, mask):                                          # position-tagged VISIBLE token features
        x = self.embed(torch.cat([B["inp"], B["occ"]], -1)) + self._cond(B)
        return x[~mask]

    def _dec(self, B, mask, lat, vis):                               # CrossMAE: latents (global) + visible (position-resolved)
        addr = self._cond(B) + mask.unsqueeze(-1).to(lat.dtype) * self.mask_tok   # CLEAN position address
        qr = kr = None
        if self.rope_cout2:                                          # RELATIVE-position addressing in cout2
            hd = self.d // self.heads; vism = ~mask
            qr = (rope_angles(B["t_phys"], hd, *self.lam_t), rope_angles(B["wire_pos"], hd, *self.lam_w))
            kr = (rope_angles(B["t_phys"][vism], hd, *self.lam_t), rope_angles(B["wire_pos"][vism], hd, *self.lam_w))
        if self.parallel_decode:                                     # query BOTH paths with the clean address, then fuse
            return self.norm(self.cout(addr, lat) + self.cout2(addr, vis, qrope=qr, krope=kr))
        q = self.cout(addr, lat)                                     # (baseline) sequential: cout2 sees the cout output
        q = self.cout2(q, vis, qrope=qr, krope=kr)
        return self.norm(q)

    def forward_feat(self, B, mask):
        vis = self._vis(B, mask)
        lat = self.deep(self.cin(self.latents, vis))
        return self._dec(B, mask, lat, vis)

    def forward_batch(self, Bs, masks, ckpt=True):
        from torch.utils.checkpoint import checkpoint
        viss = [self._vis(B, m) for B, m in zip(Bs, masks)]           # visible features (shared by encode + decode)
        enc = (lambda v: checkpoint(self.cin, self.latents, v, use_reentrant=False)) if ckpt else (lambda v: self.cin(self.latents, v))
        dc = (lambda B, m, l, v: checkpoint(self._dec, B, m, l, v, use_reentrant=False)) if ckpt else self._dec
        lat = torch.stack([enc(v) for v in viss])                    # (B, M, d)
        lat = self.deep(lat)                                         # BATCHED deep stack
        return [dc(B, m, lat[i], viss[i]) for i, (B, m) in enumerate(zip(Bs, masks))]
