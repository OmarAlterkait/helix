"""Supervised denoising: noisy coefficient tokens -> pre-response charge per cell.

The target is the simulation's ``hits`` (charge arriving at each wire before the
field response) summed onto the fine grid of :mod:`helix.probe.resolution`
(``FW`` wires x ``FT`` ticks), predicted as ``log1p(q / Q0)``: the representation
the floor evaluation scores. The input is exactly what the foundation model
sees -- every noisy token of an event, none masked.

``DenoiseModel`` = the FM encoder (``SerialFMModel.encode``) + ``CellHead``.
With ``presence`` the head also returns a logit for "this cell carries charge"
(a hurdle model): the floor is a detection question, and a regression to
``log1p(q/Q0)`` pulls an ambiguous blip to an intermediate value either way,
which puts faint deposits beside noise in charge units. The
head reads, for each queried cell, the encoder features of the token covering it
in each band plus the cell's offset inside those tokens
(:func:`helix.probe.resolution.cell_inputs`) -- the frozen probe's input, made
trainable -- so the same encoder can be trained from scratch, fine-tuned from a
pretrained checkpoint, or frozen, and the three differ only in what was trained.

The encoder's decoder, heads and mask token are frozen (``requires_grad=False``):
``encode`` never reaches them, and a parameter DDP never sees a gradient for would
otherwise stall the all-reduce.
"""
from __future__ import annotations

import torch
import torch.nn as nn

#: Encoder parameters ``encode`` reads; everything else in the FM is frozen.
ENCODER_PREFIXES = ("embed.", "film.", "band_emb.", "plane_emb.", "cond_wire.", "enc.", "pool_")


class CellHead(nn.Module):
    """Per-cell readout over the covering tokens of every band."""

    def __init__(self, d, n_bands=4, proj=256, hidden=1024, n_aux=None, presence=False):
        super().__init__()
        self.n_bands, self.proj_dim, self.presence = n_bands, proj, bool(presence)
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, proj)
        n_aux = 13 * n_bands if n_aux is None else n_aux
        self.mlp = nn.Sequential(nn.Linear(n_bands * proj + n_aux, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden // 2), nn.GELU(),
                                 nn.Linear(hidden // 2, 2 if presence else 1))

    def forward(self, feats, idx, aux):
        """feats (N_tok, d); idx (n, n_bands) covering token or -1; aux (n, n_aux).
        -> charge ``y`` (n,), or ``(y, presence_logit)`` with ``presence``."""
        z = self.proj(self.norm(feats))
        g = z[idx.clamp(min=0)] * (idx >= 0).unsqueeze(-1).to(z.dtype)      # absent band -> zeros
        out = self.mlp(torch.cat([g.flatten(1), aux.to(g.dtype)], 1))
        return (out[:, 0], out[:, 1]) if self.presence else out.squeeze(-1)


class CellDecoder(nn.Module):
    """Per-cell readout at cell resolution: cross-attention over the token
    neighbourhood, plus the covering tokens' raw coefficients.

    ``CellHead`` predicts a 2-wire x 16-tick cell from ONE vector per band (the
    token covering it, up to 16 wires x 128 ticks) and the cell's offset: it
    cannot compare the cell with the patch beside it, so near activity it spreads
    probability over the whole patch -- the blocky haze -- and localises coarsely.
    Here each cell's query (its offsets and the covering tokens' 128 input slots
    per band, i.e. the coefficients themselves) attends to the encoder features of
    the (2r+1)^2 patches around it in every band (:func:`helix.probe.resolution.
    cell_neighbors`), with a learned embedding per (band, neighbour slot) and a
    learned null key so a cell with no token anywhere near still has one.
    """

    def __init__(self, d, n_bands=4, n_nbr=9, n_slot=128, proj=128, heads=4, layers=2, hidden=512,
                 n_aux=None, presence=False):
        super().__init__()
        self.n_bands, self.n_nbr, self.heads, self.presence = n_bands, n_nbr, heads, bool(presence)
        n_aux = 13 * n_bands if n_aux is None else n_aux
        self.norm = nn.LayerNorm(d)
        self.kv = nn.Linear(d, proj)
        self.pos = nn.Parameter(torch.zeros(n_bands * n_nbr + 1, proj))       # + null key
        self.null = nn.Parameter(torch.zeros(proj))
        self.q0 = nn.Sequential(nn.Linear(n_aux + n_bands * 2 * n_slot, hidden), nn.GELU(), nn.Linear(hidden, proj))
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict(dict(
                nq=nn.LayerNorm(proj), nk=nn.LayerNorm(proj), q=nn.Linear(proj, proj), k=nn.Linear(proj, proj),
                v=nn.Linear(proj, proj), o=nn.Linear(proj, proj),
                nm=nn.LayerNorm(proj), mlp=nn.Sequential(nn.Linear(proj, 4 * proj), nn.GELU(), nn.Linear(4 * proj, proj)))))
        self.out = nn.Sequential(nn.LayerNorm(proj), nn.Linear(proj, 2 if presence else 1))
        nn.init.normal_(self.pos, std=0.02); nn.init.normal_(self.null, std=0.02)

    def forward(self, feats, idx, aux, nbr=None, B=None):
        import torch.nn.functional as F
        n = idx.shape[0]
        z = self.kv(self.norm(feats))                                           # (N_tok, P)
        flat = nbr.reshape(n, -1)                                               # (n, nb*K)
        kv = z[flat.clamp(min=0)]                                               # (n, nb*K, P)
        kv = torch.cat([kv, self.null.to(kv.dtype).expand(n, 1, -1)], 1) + self.pos.to(kv.dtype)
        keep = torch.cat([flat >= 0, torch.ones(n, 1, dtype=torch.bool, device=flat.device)], 1)
        raw = torch.cat([B["inp"], B["occ"].to(B["inp"].dtype)], -1)            # (N_tok, 2*n_slot)
        r = raw[idx.clamp(min=0)] * (idx >= 0).unsqueeze(-1).to(raw.dtype)      # (n, nb, 2*n_slot)
        q = self.q0(torch.cat([aux.to(r.dtype), r.flatten(1)], 1).to(kv.dtype))
        h, P = self.heads, q.shape[-1]
        mask = keep[:, None, None, :]                                           # (n, 1, 1, L)
        for blk in self.blocks:
            kn = blk["nk"](kv)
            Q = blk["q"](blk["nq"](q)).view(n, 1, h, P // h).transpose(1, 2)
            K = blk["k"](kn).view(n, -1, h, P // h).transpose(1, 2)
            V = blk["v"](kn).view(n, -1, h, P // h).transpose(1, 2)
            a = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)        # (n, h, 1, P/h)
            q = q + blk["o"](a.transpose(1, 2).reshape(n, P))
            q = q + blk["mlp"](blk["nm"](q))
        out = self.out(q)
        return (out[:, 0], out[:, 1]) if self.presence else out.squeeze(-1)


class DenoiseModel(nn.Module):
    def __init__(self, fm, head):
        super().__init__()
        self.fm, self.head = fm, head
        for n, p in self.fm.named_parameters():
            if not n.startswith(ENCODER_PREFIXES):
                p.requires_grad_(False)

    def forward(self, B, idx, aux, nbr=None):
        feats = self.fm.encode(B)
        if isinstance(self.head, CellDecoder):
            return self.head(feats, idx, aux, nbr=nbr, B=B)
        return self.head(feats, idx, aux)

    def param_groups(self, lr, lr_head, weight_decay):
        """The encoder's muP groups (trainable parameters only) plus the head's."""
        groups = self.fm.param_groups(lr, weight_decay=weight_decay)
        groups.append(dict(params=[p for p in self.head.parameters()], lr=lr_head,
                           weight_decay=weight_decay))
        return groups


def build_denoise(arch, state_dict=None, *, head_kw=None, overrides=None):
    """A DenoiseModel from an FM architecture dict (e.g. an eval artifact's
    ``arch``), optionally with the FM's pretrained weights."""
    from helix.model.fm import build_fm

    a = dict(arch)
    a.update(overrides or {})
    fm = build_fm(a)
    if state_dict is not None:
        missing, unexpected = fm.load_state_dict(state_dict, strict=False)
        enc_missing = [k for k in missing if k.startswith(ENCODER_PREFIXES)]
        if enc_missing or unexpected:
            raise ValueError(f"pretrained weights do not fit the encoder: missing {enc_missing[:5]}, "
                             f"unexpected {list(unexpected)[:5]}")
    kw = dict(head_kw or {})
    kind = kw.pop("kind", "mlp")
    nb = fm.band_emb.weight.shape[0]
    if kind == "decoder":
        kw.setdefault("n_slot", int(a.get("n_slot", 128)))          # the raw slots the query reads
        head = CellDecoder(fm.d, n_bands=nb, **kw)
    else:
        head = CellHead(fm.d, n_bands=nb, **kw)
    return DenoiseModel(fm, head)
