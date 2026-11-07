"""Supervised denoising: noisy coefficient tokens -> pre-response charge per fine cell.

A denoiser is a pretrained FM run plus a task head. The target is the simulation's ``hits``
(charge arriving at each wire before the field response) on the fine grid of
:mod:`helix.data.cells` (``FW`` wires x ``FT`` ticks), predicted as ``log1p(q / Q0)``, together
with a presence logit for "this cell carries charge" (a hurdle model: the floor is a detection
question, and a regression alone pulls an ambiguous blip to an intermediate value). The input is
exactly what the FM sees -- every noisy token of an event, none masked. The query cells and their
labels come from :class:`helix.data.denoise.DenoiseRows`.

``DenoiseModel`` = the FM encoder (``FMModel.encode``) + ``CellDecoder``. The encoder's
architecture and tokenizer are those of the FM run ``fm`` (:func:`helix.model.checkpoint.
load_run`); ``pretrained`` loads its weights (False: the same architecture from scratch, the
baseline), ``freeze_encoder`` keeps them fixed. The FM's decoder, heads and mask token never
train (``encode`` does not reach them, and a parameter DDP never sees a gradient for would stall
the all-reduce). ``lin_amp_scale`` > 0 gives the encoder the linear-amplitude input channel
(``FMModel``); a pretrained FM without it loads with those columns at zero
(:func:`pad_input_columns`).

``cond_charge`` selects the product: off (M6d, DENOISE at tag archive-2026-11) regresses charge on every queried
cell -- detection and maps; on (M6dc) regresses it on charged cells only, a charge conditional on
presence -- calorimetry.
"""
from __future__ import annotations

import torch
import torch.nn as nn

#: Encoder parameters ``encode`` reads; everything else in the FM is frozen.
ENCODER_PREFIXES = ("embed.", "film.", "band_emb.", "plane_emb.", "enc.")


class CellDecoder(nn.Module):
    """Per-cell readout at cell resolution: cross-attention over the token neighbourhood, plus
    the covering tokens' raw coefficients; returns ``(charge, presence_logit)`` per cell.

    Each cell's query (its offsets and the covering tokens' ``n_slot`` input slots per band, i.e.
    the coefficients themselves) attends to the encoder features of the (2r+1)^2 patches around
    it in every band (:func:`helix.data.cells.cell_neighbors`), with a learned embedding per
    (band, neighbour slot) and a learned null key so a cell with no token anywhere near still
    has one. (A per-band covering-token MLP predicts a 2-wire x 16-tick cell from one vector per
    16-wire x 128-tick patch, cannot compare the cell with the patch beside it, and spreads
    probability over the whole patch near activity: DENOISE at tag archive-2026-11, "Output model".)
    """

    def __init__(self, d, n_bands=4, n_nbr=9, n_slot=128, proj=128, heads=4, layers=2, hidden=512):
        super().__init__()
        self.n_bands, self.n_nbr, self.heads = n_bands, n_nbr, heads
        n_aux = 13 * n_bands                                                    # helix.data.cells.cell_inputs
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
        # a plain linear readout of the residual stream: a LayerNorm before it bounds the
        # charge logit by the weights (M5d/M5dc capped near 0.8M e- per cell, M4n reaches 2.6M)
        self.out = nn.Linear(proj, 2)
        nn.init.normal_(self.pos, std=0.02); nn.init.normal_(self.null, std=0.02)

    def forward(self, feats, idx, aux, nbr, B):
        """feats (N_tok, d); idx (n, n_bands) covering token or -1; aux (n, 13 n_bands); nbr
        (n, n_bands, n_nbr) neighbourhood tokens or -1; ``B`` the FM batch (its ``inp``/``occ``).
        -> ``(charge, presence_logit)``, each (n,) float32."""
        n = idx.shape[0]
        # keys/values are projected once per TOKEN (~30k rows), then gathered for every
        # (cell, neighbour slot) pair (~n x 37) -- projecting after the gather cost 90x the
        # matmul rows and made a step 5x slower; the slot embedding is projected and added
        # after the gather instead of before.
        # Gather indices must not repeat heavily: a gather's backward is an atomic
        # scatter-add, and absent slots clamped to one row (or the null key gathered
        # by every cell) serialise ~1M adds on that row -- a 6-9 s backward on real
        # events. Absent slots get spread dummy rows (masked, so their gradients are
        # zero); the null key is appended, not gathered.
        z = self.kv(self.norm(feats))                                           # (N_tok, P)
        flat = nbr.reshape(n, -1)                                               # (n, nb*K)
        nt = z.shape[0]
        spread = torch.arange(flat.numel(), device=flat.device).view_as(flat) % nt
        gidx = torch.where(flat >= 0, flat, spread)
        keep = torch.cat([flat >= 0, torch.ones(n, 1, dtype=torch.bool, device=flat.device)], 1)
        raw = torch.cat([B["inp"], B["occ"].to(B["inp"].dtype)], -1)            # (N_tok, 2*n_slot)
        r = raw[idx.clamp(min=0)] * (idx >= 0).unsqueeze(-1).to(raw.dtype)      # (n, nb, 2*n_slot)
        q = self.q0(torch.cat([aux.to(r.dtype), r.flatten(1)], 1)).float()
        h, P = self.heads, q.shape[-1]
        L = flat.shape[1]
        for blk in self.blocks:
            zn, nn_ = blk["nk"](z), blk["nk"](self.null)
            Kt, Vt = blk["k"](zn).float(), blk["v"](zn).float()                # (N_tok, P), gathered in fp32
            kp, vp = blk["k"](self.pos).float(), blk["v"](self.pos).float()    # (L+1, P)
            K = torch.cat([Kt[gidx] + kp[:L], (blk["k"](nn_).float() + kp[L]).expand(n, 1, P)], 1)
            V = torch.cat([Vt[gidx] + vp[:L], (blk["v"](nn_).float() + vp[L]).expand(n, 1, P)], 1)
            Qh = blk["q"](blk["nq"](q)).float().view(n, h, P // h)
            s = torch.einsum("nhd,nlhd->nhl", Qh, K.view(n, L + 1, h, P // h)) * (P // h) ** -0.5
            a = s.masked_fill(~keep[:, None, :], float("-inf")).softmax(-1)
            att = torch.einsum("nhl,nlhd->nhd", a, V.view(n, L + 1, h, P // h)).reshape(n, P)
            q = q + blk["o"](att).float()
            q = q + blk["mlp"](blk["nm"](q)).float()
        out = self.out(q).float()
        return out[:, 0], out[:, 1]


class DenoiseModel(nn.Module):
    """The FM encoder + :class:`CellDecoder`; ``forward(batch) -> {loss, mse, bce}`` (pimm's
    Trainer contract) on the query cells :class:`helix.data.denoise.DenoiseRows` put in the
    batch (``q_idx``, ``q_aux``, ``q_nbr``, ``q_y``, ``q_present``)."""

    def __init__(self, fm, head, *, tokenizer=None, freeze_encoder=False, cond_charge=False, lr_head=1e-2):
        super().__init__()
        self.fm, self.head = fm, head
        self.tokenizer = tokenizer                 # the FM run's PatchConfig (FMTrainer gives it to the data)
        self.freeze_encoder, self.cond_charge, self.lr_head = bool(freeze_encoder), bool(cond_charge), float(lr_head)
        for n, p in self.fm.named_parameters():
            if self.freeze_encoder or not n.startswith(ENCODER_PREFIXES):
                p.requires_grad_(False)

    def predict(self, B, idx, aux, nbr):
        """``(charge, presence_logit)`` per queried cell, float32."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            feats = self.fm.encode(B)
        return self.head(feats, idx, aux, nbr, B)

    def forward(self, batch):
        B = dict(batch)
        B.setdefault("n_cells", B["plane_id"].shape[0])
        y_hat, logit = self.predict(B, B["q_idx"], B["q_aux"], B["q_nbr"])
        y, present = B["q_y"].float(), B["q_present"].float()
        bce = nn.functional.binary_cross_entropy_with_logits(logit, present)
        if self.cond_charge:                       # charge only where there is charge
            mse = ((y_hat - y) ** 2 * present).sum() / present.sum().clamp(min=1)
        else:
            mse = nn.functional.mse_loss(y_hat, y)
        return {"loss": mse + bce, "mse": mse.detach(), "bce": bce.detach()}

    def param_groups(self, base_lr, weight_decay=None):
        """The encoder's muP groups at ``base_lr`` (trainable parameters only: none when frozen)
        plus the head's at ``lr_head`` -- ``FMTrainer.build_optimizer``'s contract."""
        groups = self.fm.param_groups(base_lr, weight_decay=weight_decay)
        g = dict(params=list(self.head.parameters()), lr=self.lr_head)
        if weight_decay is not None:
            g["weight_decay"] = weight_decay
        return groups + [g]


def pad_input_columns(module, state_dict):
    """``state_dict`` with every matrix that has FEWER input columns than ``module``'s own (an
    encoder option such as ``lin_amp_scale`` added input features) zero-padded to fit: the old
    columns copied, the new ones zero, so the loaded model computes exactly what the checkpoint
    did until training moves the new weights."""
    own = module.state_dict()
    out = dict(state_dict)
    for k, v in state_dict.items():
        o = own.get(k)
        if o is not None and o.shape != v.shape and o.dim() == 2 and v.dim() == 2 \
                and o.shape[0] == v.shape[0] and o.shape[1] > v.shape[1]:
            w = o.new_zeros(o.shape)
            w[:, :v.shape[1]] = v
            out[k] = w
    return out


def build_denoise(fm, *, pretrained=True, freeze_encoder=False, cond_charge=False, lr_head=1e-2,
                  compile_blocks=True, lin_amp_scale=0.0):
    """A :class:`DenoiseModel` on the FM run ``fm`` (a run directory, or one checkpoint of it).

    The encoder takes the run's architecture and tokenizer and, with ``pretrained``, its weights
    (``load_run``: the run's EMA, or the checkpoint named); otherwise it is freshly initialised.
    ``lin_amp_scale`` > 0 adds the linear-amplitude input channel to that architecture (0 keeps
    the run's own); the run's weights load into it with the new columns at zero
    (:func:`pad_input_columns`). ``compile_blocks`` is how the encoder executes (``torch.compile``
    per block), not what it is.
    """
    import json
    import os

    from helix.model.checkpoint import load_run, model_config, run_dir, tokenizer_config
    from helix.model.fm import build_fm

    config = json.load(open(os.path.join(run_dir(fm), "resolved_config.json")))
    if (config.get("model") or {}).get("type") != "Coeff-FM":
        raise ValueError(f"{fm}: not an FM run (model type {(config.get('model') or {}).get('type')!r})")
    arch = dict(model_config(config), compile_blocks=bool(compile_blocks))
    if lin_amp_scale:
        arch["lin_amp_scale"] = float(lin_amp_scale)
    enc = build_fm(arch)
    if pretrained:
        sd = load_run(fm, device="cpu").model.state_dict()
        enc.load_state_dict(pad_input_columns(enc, sd), strict=True)
    head = CellDecoder(enc.d, n_bands=enc.band_emb.weight.shape[0], n_slot=enc.n_slot)
    return DenoiseModel(enc, head, tokenizer=tokenizer_config(config), freeze_encoder=freeze_encoder,
                        cond_charge=cond_charge, lr_head=lr_head)
