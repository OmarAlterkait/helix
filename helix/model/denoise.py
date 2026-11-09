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

``dist`` is the charge head's form (one will be chosen by a full-length comparison):

  point   a value per cell, squared error: a Gaussian of one fixed width for every cell (the
          original, and the default)
  gauss   + a log-width per cell (``dist_out``), trained by the Gaussian likelihood
          (:func:`gauss_nll_loss`, beta 0) -- with ``cond_charge``, on the charged cells; without
          it, on every queried cell as the squared error is
  cat     ONE categorical over the cell's charge in place of both heads: bin 0 "no charge"
          (q <= ``support_min_q``), bins 1..K uniform in y from y(``support_min_q``) to ``y_hi``,
          trained by cross-entropy against HL-Gauss targets (:func:`cat_targets`). Presence is
          1 - p0, the charge the mean of y over bins 1..K. Its cross-entropy is exactly the
          presence BCE plus the charged rows' conditional cross-entropy, so it is a hurdle --
          ``cond_charge`` -- by construction, and it refuses ``cond_charge=False``

``CellDecoder.forward`` returns ``(charge, presence_logit)`` for point and
``(charge, presence_logit, extra)`` otherwise: ``extra`` is the log-width (gauss) or the K + 1
logits (cat), ``charge`` the point estimate (gauss: the mean; cat: the conditional mean) and
``sigmoid(presence_logit)`` the probability of charge (cat: exactly 1 - p0). So every consumer of
the first two works on every head; :meth:`DenoiseModel.predict` returns just those two.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from helix.data.cells import Q0

#: Encoder parameters ``encode`` reads; everything else in the FM is frozen.
ENCODER_PREFIXES = ("embed.", "film.", "band_emb.", "plane_emb.", "enc.")
#: The charge head's forms (``CellDecoder(dist=...)``; module docstring).
DISTS = ("point", "gauss", "cat")
#: The gauss head's log-width is clamped here (y units: 1e-3 .. 20) in the loss and in its use.
LOG_SIGMA_RANGE = (-7.0, 3.0)


class CellDecoder(nn.Module):
    """Per-cell readout at cell resolution: cross-attention over the token neighbourhood, plus
    the covering tokens' raw coefficients; returns ``(charge, presence_logit)`` per cell, and the
    distribution's own output third when ``dist`` is not point (module docstring).

    Each cell's query (its offsets and the covering tokens' ``n_slot`` input slots per band, i.e.
    the coefficients themselves) attends to the encoder features of the (2r+1)^2 patches around
    it in every band (:func:`helix.model.cells_join.cell_neighbors`), with a learned embedding per
    (band, neighbour slot) and a learned null key so a cell with no token anywhere near still
    has one. (A per-band covering-token MLP predicts a 2-wire x 16-tick cell from one vector per
    16-wire x 128-tick patch, cannot compare the cell with the patch beside it, and spreads
    probability over the whole patch near activity: DENOISE at tag archive-2026-11, "Output model".)

    ``dist="cat"``: ``n_bins`` = K charged bins over y in [y(``support_min_q``), ``y_hi``], targets
    smoothed by a Gaussian of ``hl_sigma`` bin widths. ``dist="gauss"``: ``dist_out`` starts at
    zero (log-width 0), so a point head's weights loaded into it (``strict=False``: ``dist_out``
    is the only key they lack) start at the point head's charge.
    """

    def __init__(self, d, n_bands=4, n_nbr=9, n_slot=128, proj=128, heads=4, layers=2, hidden=512,
                 dist="point", n_bins=256, support_min_q=25.0, y_hi=7.5, hl_sigma=0.75):
        super().__init__()
        if dist not in DISTS:
            raise ValueError(f"dist must be one of {DISTS}, got {dist!r}")
        self.n_bands, self.n_nbr, self.heads, self.dist = n_bands, n_nbr, heads, dist
        n_aux = 13 * n_bands                                                    # helix.model.cells_join.cell_inputs
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
        self.out = nn.Linear(proj, int(n_bins) + 1 if dist == "cat" else 2)
        nn.init.normal_(self.pos, std=0.02); nn.init.normal_(self.null, std=0.02)
        if dist == "gauss":
            self.dist_out = nn.Linear(proj, 1)
            nn.init.zeros_(self.dist_out.weight); nn.init.zeros_(self.dist_out.bias)
        elif dist == "cat":
            edges = cat_edges(support_min_q, y_hi, n_bins)
            self.register_buffer("edges", edges, persistent=False)              # derived from the arguments
            self.register_buffer("centers", 0.5 * (edges[1:] + edges[:-1]), persistent=False)
            self.hl_sigma = float(hl_sigma) * float(edges[1] - edges[0])        # in y

    def forward(self, feats, idx, aux, nbr, B):
        """feats (N_tok, d); idx (n, n_bands) covering token or -1; aux (n, 13 n_bands); nbr
        (n, n_bands, n_nbr) neighbourhood tokens or -1; ``B`` the FM batch (its ``inp``/``occ``).
        -> ``(charge, presence_logit)``, each (n,) float32, + the log-width (n,) (gauss) or the
        K + 1 logits (n, K + 1) (cat)."""
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
        # The readout in fp32: under bf16 autocast a Linear's OUTPUT is bf16, and a later .float()
        # keeps its rounding -- a 0.031 step in y above y = 4 (~110k e-), ~3% in charge.
        with torch.autocast(q.device.type, enabled=False):
            out = self.out(q.float())
            if self.dist == "cat":
                return cat_mean(out, self.centers), cat_presence_logit(out), out
            if self.dist == "gauss":
                return out[:, 0], out[:, 1], self.dist_out(q.float()).squeeze(-1)
        return out[:, 0], out[:, 1]


class DenoiseModel(nn.Module):
    """The FM encoder + :class:`CellDecoder`; ``forward(batch) -> {loss, mse, bce, ...}`` (pimm's
    Trainer contract) on the query cells :class:`helix.data.denoise.DenoiseRows` put in the
    batch (``q_idx``, ``q_aux``, ``q_nbr``, ``q_y``, ``q_present``).

    Every head reports ``mse`` (its point estimate's squared error, on the rows ``cond_charge``
    regresses) and ``bce`` (its presence probability's), which pimm's InformationWriter logs;
    ``loss`` is what trains: ``mse + bce`` (point), ``nll + bce`` (gauss), ``ce`` (cat) -- so the
    loss compares checkpoints of one head, not heads."""

    def __init__(self, fm, head, *, tokenizer=None, freeze_encoder=False, cond_charge=False, lr_head=1e-2):
        super().__init__()
        self.fm, self.head = fm, head
        self.tokenizer = tokenizer                 # the FM run's PatchConfig (FMTrainer gives it to the data)
        self.freeze_encoder, self.cond_charge, self.lr_head = bool(freeze_encoder), bool(cond_charge), float(lr_head)
        if head.dist == "cat" and not self.cond_charge:
            raise ValueError("dist='cat' is a hurdle by construction (bin 0 is 'no charge', the charge is "
                             "conditional on the rest): set cond_charge=True")
        if head.dist == "gauss" and not self.cond_charge:
            # a Gaussian likelihood on the empty rows' exact zeros drives their width to the clamp,
            # where its gradient stops and their squared error is weighted ~1e6
            raise ValueError("dist='gauss' models the charge of charged cells: set cond_charge=True")
        for n, p in self.fm.named_parameters():
            if self.freeze_encoder or not n.startswith(ENCODER_PREFIXES):
                p.requires_grad_(False)

    def outputs(self, B, idx, aux, nbr):
        """The head's whole output per queried cell (:meth:`CellDecoder.forward`)."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            feats = self.fm.encode(B)
        return self.head(feats, idx, aux, nbr, B)

    def predict(self, B, idx, aux, nbr):
        """``(charge, presence_logit)`` per queried cell, float32, whatever the head."""
        return self.outputs(B, idx, aux, nbr)[:2]

    def forward(self, batch):
        B = dict(batch)
        B.setdefault("n_cells", B["plane_id"].shape[0])
        out = self.outputs(B, B["q_idx"], B["q_aux"], B["q_nbr"])
        y_hat, logit = out[0], out[1]
        y, present = B["q_y"].float(), B["q_present"].float()
        bce = nn.functional.binary_cross_entropy_with_logits(logit, present)
        if self.cond_charge:                       # charge only where there is charge
            mse = ((y_hat - y) ** 2 * present).sum() / present.sum().clamp(min=1)
        else:
            mse = nn.functional.mse_loss(y_hat, y)
        if self.head.dist == "gauss":
            nll = gauss_nll_loss(y_hat, out[2], y, present if self.cond_charge else torch.ones_like(present))
            return {"loss": nll + bce, "mse": mse.detach(), "bce": bce.detach(), "nll": nll.detach()}
        if self.head.dist == "cat":
            ce = cat_loss(out[2], y, present, self.head.edges, self.head.hl_sigma)
            return {"loss": ce, "mse": mse.detach(), "bce": bce.detach(), "ce": ce.detach()}
        return {"loss": mse + bce, "mse": mse.detach(), "bce": bce.detach()}

    def param_groups(self, base_lr, weight_decay=None):
        """The encoder's muP groups at ``base_lr`` (trainable parameters only: none when frozen)
        plus the head's at ``lr_head`` -- ``FMTrainer.build_optimizer``'s contract."""
        groups = self.fm.param_groups(base_lr, weight_decay=weight_decay)
        g = dict(params=list(self.head.parameters()), lr=self.lr_head)
        if weight_decay is not None:
            g["weight_decay"] = weight_decay
        return groups + [g]


# ---------------------------------------------------------------------------- distributional heads
def gauss_nll_loss(mu, log_sigma, y, w, beta=0.0):
    """Gaussian negative log-likelihood of y under (mu, sigma), up to the constant log sqrt(2 pi),
    weighted per row by ``w`` (the charged rows, or all) and normalised by its sum. The width is
    the network's own per cell, so each cell weighs by its achievable precision (inverse
    variance), set from the INPUT, and the mean stays unbiased. ``beta`` > 0 is beta-NLL
    (Seitzer et al. 2022: each row times stop-grad sigma^(2 beta)); 0, the plain likelihood, is
    the measured default (M14: beta 0.5 was worse)."""
    ls = log_sigma.float().clamp(*LOG_SIGMA_RANGE)
    nll = 0.5 * (y.float() - mu.float()) ** 2 * torch.exp(-2 * ls) + ls
    wt = w.to(nll.dtype) * (torch.exp(2 * beta * ls).detach() if beta else 1.0)
    return (nll * wt).sum() / w.to(nll.dtype).sum().clamp(min=1)


def cat_edges(support_min_q, y_hi, n_bins):
    """The cat head's K + 1 charged-bin edges in y = log1p(q / Q0): uniform from
    y(``support_min_q``) -- the presence threshold, bin 0's top -- to ``y_hi``."""
    lo = math.log1p(float(support_min_q) / Q0)
    if not float(y_hi) > lo or int(n_bins) < 1:
        raise ValueError(f"cat bins need y_hi > y(support_min_q) = {lo:.4g} and n_bins >= 1, "
                         f"got y_hi={y_hi}, n_bins={n_bins}")
    return torch.linspace(lo, float(y_hi), int(n_bins) + 1)


def cat_targets(y, present, edges, sigma):
    """(n, K + 1) target distributions: a charged row (``present``) puts on bins 1..K the mass a
    Gaussian of width ``sigma`` centred on its y (clamped into the edges) gives each bin,
    renormalised over 1..K (HL-Gauss: Imani & White 2018; Farebrother et al. 2024); an empty row
    is all bin 0."""
    yc = y.float().clamp(edges[0], edges[-1])[:, None]
    cdf = 0.5 * (1 + torch.erf((edges[None, :] - yc) / (sigma * 2 ** 0.5)))
    p = cdf[:, 1:] - cdf[:, :-1]
    p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)
    pres = present.float()[:, None]
    return torch.cat([1 - pres, p * pres], 1)


def cat_loss(logits, y, present, edges, sigma):
    """Mean cross-entropy of the K + 1 logits against :func:`cat_targets`. Since log p_k =
    log(1 - p0) + log p(k | charged), it is the presence BCE plus the charged rows' conditional
    cross-entropy, both averaged over every row."""
    t = cat_targets(y, present, edges, sigma)
    return -(t * logits.float().log_softmax(-1)).sum(-1).mean()


def cat_presence_logit(logits):
    """The logit of 1 - p0: ``sigmoid`` of it is exactly the probability of charge."""
    lg = logits.float()
    return torch.logsumexp(lg[:, 1:], -1) - lg[:, 0]


def _cond(logits):
    """The distribution over bins 1..K given charge (bin 0 dropped, renormalised)."""
    return logits.float()[:, 1:].softmax(-1)


def cat_mean(logits, centers):
    """The charge point estimate: the mean of y over bins 1..K given charge."""
    return (_cond(logits) * centers).sum(-1)


def cat_std(logits, centers):
    """The predictive standard deviation of y given charge (the bins' centres)."""
    p = _cond(logits)
    m = (p * centers).sum(-1, keepdim=True)
    return ((p * (centers - m) ** 2).sum(-1)).sqrt()


def cat_cdf(logits, edges, y):
    """The predictive CDF given charge at y (uniform within a bin): the PIT when y is the truth."""
    w = edges[1:] - edges[:-1]
    frac = ((y.float()[:, None] - edges[None, :-1]) / w[None, :]).clamp(0, 1)
    return (_cond(logits) * frac).sum(-1)


def cat_median(logits, edges):
    """The median of y given charge (uniform within a bin)."""
    p = _cond(logits)
    c = p.cumsum(-1)
    j = torch.searchsorted(c, torch.full_like(c[:, :1], 0.5)).clamp(max=p.shape[1] - 1)
    pj, below = p.gather(1, j), (c.gather(1, j) - p.gather(1, j))
    w = (edges[1:] - edges[:-1])[j]
    return (edges[:-1][j] + w * ((0.5 - below) / pj.clamp(min=1e-12)).clamp(0, 1)).squeeze(1)


def calibration_stats(head, out, y_true):
    """``(stat, width)`` per cell of a distributional head's output ``out`` (its forward's) against
    the truth ``y_true``: gauss -- z = (y_true - mu) / sigma and sigma; cat -- the PIT
    (:func:`cat_cdf`) and the predictive std, whose coverage says whether the width is calibrated
    (the evaluation no longer scores it: tag eval-probes-final)."""
    if head.dist == "gauss":
        sig = torch.exp(out[2].float().clamp(*LOG_SIGMA_RANGE))
        return (y_true.float() - out[0].float()) / sig, sig
    if head.dist == "cat":
        return cat_cdf(out[2], head.edges, y_true), cat_std(out[2], head.centers)
    raise ValueError(f"dist={head.dist!r} has no distribution to calibrate")


# ---------------------------------------------------------------------------- building one
PAD_KEYS = ("embed.weight",)         # the matrices an encoder input option widens (lin_amp_scale)


def pad_input_columns(module, state_dict):
    """``state_dict`` with the :data:`PAD_KEYS` matrices that have FEWER input columns than
    ``module``'s own (an encoder option such as ``lin_amp_scale`` added input features)
    zero-padded to fit: the old columns copied, the new ones zero, so the loaded model computes
    exactly what the checkpoint did until training moves the new weights. Any other mismatch is
    left for ``load_state_dict`` to refuse."""
    own = module.state_dict()
    out = dict(state_dict)
    for k, v in state_dict.items():
        o = own.get(k) if k in PAD_KEYS else None
        if o is not None and o.shape != v.shape and o.dim() == 2 and v.dim() == 2 \
                and o.shape[0] == v.shape[0] and o.shape[1] > v.shape[1]:
            w = o.new_zeros(o.shape)
            w[:, :v.shape[1]] = v
            out[k] = w
    return out


def build_denoise(fm, *, pretrained=True, freeze_encoder=False, cond_charge=False, lr_head=1e-2,
                  compile_blocks=True, lin_amp_scale=0.0, dist="point", support_min_q=25.0):
    """A :class:`DenoiseModel` on the FM run ``fm`` (a run directory, or one checkpoint of it).

    The encoder takes the run's architecture and tokenizer and, with ``pretrained``, its weights
    (``load_run``: the run's last checkpoint, or the one named); otherwise it is freshly initialised.
    ``lin_amp_scale`` > 0 adds the linear-amplitude input channel to that architecture (0 keeps
    the run's own); the run's weights load into it with the new columns at zero
    (:func:`pad_input_columns`). ``dist`` is the
    charge head (module docstring); ``support_min_q`` (e-) is the presence threshold of the labels
    (``DenoiseRows``'), the top of cat's bin 0. ``compile_blocks`` is how the encoder executes
    (``torch.compile`` per block), not what it is.
    """
    import json
    import os

    if freeze_encoder and lin_amp_scale:
        # the channel's input columns start at zero (pad_input_columns): frozen, they stay zero and
        # the option would silently do nothing
        raise ValueError("freeze_encoder with lin_amp_scale > 0: the new input columns start at zero "
                         "and would stay there; train the encoder, or set lin_amp_scale=0")
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
    head = CellDecoder(enc.d, n_bands=enc.band_emb.weight.shape[0], n_slot=enc.n_slot, dist=dist,
                       support_min_q=support_min_q)
    return DenoiseModel(enc, head, tokenizer=tokenizer_config(config), freeze_encoder=freeze_encoder,
                        cond_charge=cond_charge, lr_head=lr_head)
