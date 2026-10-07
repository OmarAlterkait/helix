"""Supervised denoising: noisy coefficient tokens -> pre-response charge per cell.

The target is the simulation's ``hits`` (charge arriving at each wire before the
field response) summed onto the fine grid of :mod:`helix.probe.resolution`
(``FW`` wires x ``FT`` ticks), predicted as ``log1p(q / Q0)``: the representation
the floor evaluation scores. The input is exactly what the foundation model
sees -- every noisy token of an event, none masked.

``DenoiseModel`` = the FM encoder (``SerialFMModel.encode``) + ``CellHead``. The
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

    def __init__(self, d, n_bands=4, proj=256, hidden=1024, n_aux=None):
        super().__init__()
        self.n_bands, self.proj_dim = n_bands, proj
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, proj)
        n_aux = 13 * n_bands if n_aux is None else n_aux
        self.mlp = nn.Sequential(nn.Linear(n_bands * proj + n_aux, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden // 2), nn.GELU(),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, feats, idx, aux):
        """feats (N_tok, d); idx (n, n_bands) covering token or -1; aux (n, n_aux)."""
        z = self.proj(self.norm(feats))
        g = z[idx.clamp(min=0)] * (idx >= 0).unsqueeze(-1).to(z.dtype)      # absent band -> zeros
        return self.mlp(torch.cat([g.flatten(1), aux.to(g.dtype)], 1)).squeeze(-1)


class DenoiseModel(nn.Module):
    def __init__(self, fm, head):
        super().__init__()
        self.fm, self.head = fm, head
        for n, p in self.fm.named_parameters():
            if not n.startswith(ENCODER_PREFIXES):
                p.requires_grad_(False)

    def forward(self, B, idx, aux):
        return self.head(self.fm.encode(B), idx, aux)

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
    head = CellHead(fm.d, n_bands=fm.band_emb.weight.shape[0], **(head_kw or {}))
    return DenoiseModel(fm, head)
