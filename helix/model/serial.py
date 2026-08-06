"""SerialFMModel — the production variant (grouped/serialised attention).

Extracted verbatim from ``fm/model_serial.py``; this is the class the live runs
used, so it is what ``build_fm`` returns by default.
"""
"""SerialFMModel: FMModel with the grouped 3-order serial encoder + grouped-cross decoder.
Reuses ALL FMModel weights (same Block/CrossBlock params) -> warm-starts from a full-attention ckpt.
rope_split=True  -> axial (t,wire) RoPE on plane-order layers, TIME-ONLY on drift-order + decoder (the fix).
rope_split=False -> global (t,wire) RoPE everywhere (matches the full-attn model's behavior).
"""
import torch, torch.nn.functional as F
from helix.model.fm import FMModel, apply_rope, rope_angles


def uniform_attn(q, k, v, order, g):
    T, h, hd = q.shape; npad = ((T + g - 1) // g) * g; nb = npad // g
    def grp(x):
        b = x.new_empty(npad, h, hd); b[:T] = x[order]
        if npad > T: b[T:] = x[order[-1]]
        return b.view(nb, g, h, hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(grp(q), grp(k.to(q.dtype)), grp(v.to(q.dtype)))
    o = o.permute(0, 2, 1, 3).reshape(npad, h, hd)[:T]
    out = o.new_empty(T, h, hd); out[order] = o; return out


def grouped_cross(q, k, v, oq, ok, g):
    Tq, h, hd = q.shape; Tk = k.shape[0]; nb = (max(Tq, Tk) + g - 1) // g
    def grp(x, order, T):
        gg = (T + nb - 1) // nb; npad = nb * gg
        b = x.new_empty(npad, h, hd); b[:T] = x[order]
        if npad > T: b[T:] = x[order[-1]]
        return b.view(nb, gg, h, hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(grp(q, oq, Tq), grp(k.to(q.dtype), ok, Tk), grp(v.to(q.dtype), ok, Tk))
    gq = ((Tq + nb - 1) // nb); o = o.permute(0, 2, 1, 3).reshape(nb * gq, h, hd)[:Tq]
    out = o.new_empty(Tq, h, hd); out[oq] = o; return out


def _self(blk, x, at, aw, order, g):
    T, d = x.shape; hh = blk.n1(x)
    q, k, v = blk.qkv(hh).chunk(3, -1)
    q = apply_rope(q.view(T, blk.h, blk.hd), at, aw)
    k = apply_rope(k.view(T, blk.h, blk.hd), at, aw)
    o = uniform_attn(q, k, v.view(T, blk.h, blk.hd), order, g)
    x = x + blk.proj(o.reshape(T, d)); return x + blk.mlp(blk.n2(x))


def _cross(blk, q, kv, qat, qaw, kat, kaw, oq, okv, g):
    Tq, Tk = q.shape[0], kv.shape[0]
    qh = apply_rope(blk.q(blk.nq(q)).view(Tq, blk.h, blk.hd), qat, qaw)
    k, v = blk.kv(blk.nk(kv)).chunk(2, -1)
    kh = apply_rope(k.view(Tk, blk.h, blk.hd), kat, kaw)
    o = grouped_cross(qh, kh, v.view(Tk, blk.h, blk.hd), oq, okv, g)
    q = q + blk.proj(o.reshape(Tq, blk.h * blk.hd)); return q + blk.mlp(blk.n2(q))


class SerialFMModel(FMModel):
    def __init__(self, *args, rope_split=True, gp=1024, gd=2048, **kw):
        super().__init__(*args, **kw)
        self.rope_split, self.gp, self.gd = rope_split, gp, gd

    def _sched(self, plane, t, wire):
        bp = plane.double()
        o_pt = torch.argsort(bp * 1e9 + t.double())
        o_pw = torch.argsort(bp * 1e13 + wire.double() * 1e6 + t.double())
        o_t = torch.argsort(t.double()); o_ts = torch.roll(o_t, self.gd // 2)
        dw = not self.rope_split                                   # drift-layer wire RoPE: off if split
        cell = [(o_pt, self.gp, True), (o_t, self.gd, dw), (o_pw, self.gp, True), (o_ts, self.gd, dw)]
        return (cell * ((len(self.enc) + 3) // 4))[:len(self.enc)]

    def _emb(self, B, idx=None):
        band, plane = B["band_id"], B["plane_id"]
        sel = slice(None) if idx is None else idx
        x = self.embed(torch.cat([B["inp"][sel], B["occ"][sel]], -1))
        if self.film is not None:
            g, b = self.film(band[sel], plane[sel], B["wirefeat"][sel]); x = g * x + b
        return x + self.band_emb(band[sel]) + self.plane_emb(plane[sel])

    def encode(self, B):
        at = rope_angles(B["t_phys"], self.d // self.heads, *self.lam_t)
        aw = rope_angles(B["wire_pos"], self.d // self.heads, *self.lam_w)
        x = self._emb(B); sched = self._sched(B["plane_id"], B["t_phys"], B["wire_pos"])
        for blk, (o, g, uw) in zip(self.enc, sched):
            x = _self(blk, x, at, aw if uw else None, o, g)
        return x

    def encode_layers(self, B, layers):
        at = rope_angles(B["t_phys"], self.d // self.heads, *self.lam_t)
        aw = rope_angles(B["wire_pos"], self.d // self.heads, *self.lam_w)
        x = self._emb(B); sched = self._sched(B["plane_id"], B["t_phys"], B["wire_pos"]); out = {}
        for i, (blk, (o, g, uw)) in enumerate(zip(self.enc, sched), 1):
            x = _self(blk, x, at, aw if uw else None, o, g)
            if i in layers: out[i] = x
        return out

    def forward_feat(self, B, tok_mask, return_ctx=False):
        N = B["inp"].shape[0]
        at = rope_angles(B["t_phys"], self.d // self.heads, *self.lam_t)
        aw = rope_angles(B["wire_pos"], self.d // self.heads, *self.lam_w)
        vis = ~tok_mask; vis_idx = vis.nonzero(as_tuple=True)[0]; mask_idx = tok_mask.nonzero(as_tuple=True)[0]
        xv = self._emb(B, vis_idx); atv, awv = at[vis], aw[vis]
        sched = self._sched(B["plane_id"][vis], B["t_phys"][vis], B["wire_pos"][vis])
        for blk, (o, g, uw) in zip(self.enc, sched):
            xv = _self(blk, xv, atv, awv if uw else None, o, g)
        # CrossMAE decoder (grouped, drift-time): masked queries x-attend visible
        qm = self.mask_tok.expand(mask_idx.numel(), self.d)
        if self.film is not None:
            g_, b_ = self.film(B["band_id"][tok_mask], B["plane_id"][tok_mask], B["wirefeat"][tok_mask]); qm = g_ * qm + b_
        qm = (qm + self.band_emb(B["band_id"][tok_mask]) + self.plane_emb(B["plane_id"][tok_mask])).to(xv.dtype)
        atm, awm = at[tok_mask], aw[tok_mask]
        oq = torch.argsort(B["t_phys"][tok_mask].double()); okv = torch.argsort(B["t_phys"][vis].double())
        # Decoder ALWAYS keeps axial (wire) RoPE: it needs the wire address to know which wire it
        # reconstructs -- dropping it froze recon (var_expl ~2%, same as wire_rope=0). rope_split
        # only affects the ENCODER cross-plane (drift-order) layers, via `dw` in _sched.
        for blk in self.dec:
            qm = _cross(blk, qm, xv, atm, awm, atv, awv, oq, okv, self.gd)
        x = torch.zeros(N, self.d, dtype=xv.dtype, device=xv.device)
        x = x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm)
        return (self.dec_norm(x), xv) if return_ctx else self.dec_norm(x)
