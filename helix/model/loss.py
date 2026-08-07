"""Masked-coefficient objectives — extracted verbatim from the research tree
(``coeff_foundation_model/fm/model.py`` lines 370-451).

``losses`` (L2 / Gaussian-NLL), ``losses_fused`` (adds the alpha/beta variance
terms) and ``losses_cat`` (categorical head, K bins per slot).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def losses(occ, mu, logvar, B, tok_mask, vis_w=0.0, noisy=False):
    """Dual-target loss. CLEAN coeff is the target at every active slot:
      - MASKED active slots  -> INFERENCE (predict clean from cross-context only)
      - VISIBLE active slots -> DENOISING (predict clean from own noisy value+context)
    Because our input is noisy and target is clean, the visible term is NOT a trivial
    copy (as in pixel MAE) — it's the denoising objective we actually want. vis_w
    weights it (0 = masked-only, the original behavior). Occupancy BCE stays masked-only
    (visible occupancy is observed, so supervising it would just leak)."""
    mvalid = tok_mask[:, None] & B["valid"]
    bce = F.binary_cross_entropy_with_logits(occ[mvalid], B["occ"][mvalid]) if mvalid.any() else occ.sum() * 0

    cell_masked = tok_mask[B["cell"]]                       # per-active-row: is its cell masked?

    def _value(rowsel):
        if not rowsel.any():
            return mu.sum() * 0
        pred = mu[B["cell"], B["slot"]][rowsel]
        # noisy=True -> self-supervised: predict the NOISY input coeff (no clean truth needed)
        tgt = (B["inp"][B["cell"], B["slot"]] if noisy else B["target"])[rowsel]
        if logvar is not None:
            lv = logvar[B["cell"], B["slot"]][rowsel].clamp(-8, 8)
            return 0.5 * (((pred - tgt) ** 2) * torch.exp(-lv) + lv).mean()   # Gaussian NLL
        return F.mse_loss(pred, tgt)

    val = _value(cell_masked)                               # masked: inference
    if vis_w > 0:
        val = val + vis_w * _value(~cell_masked)            # visible: denoising
    return bce, val


def losses_fused(occ, mu, logvar, B, tok_mask, vis_w=0.0, noisy=False, alpha=0.0, beta=0.0, varb=None):
    """Same loss as losses() but computed DENSELY over the (n_cells, n_slot) grid with masks
    instead of advanced-indexing gathers (mu[cell,slot], occ[mvalid]). Removes the scatter
    (indexing_backward) and collapses to elementwise ops + masked reductions that torch.compile
    fuses into ~1 kernel. Needs the dense B["tgt"] (threaded through data._to_fm)."""
    valid, occ_t = B["valid"], B["occ"]                     # dense (n_cells, n_slot)
    tgt = B["inp"] if noisy else B["tgt"]                   # dense target
    mrow = tok_mask[:, None]                                # (n_cells, 1) masked cells
    m_occ = mrow & valid                                    # masked & valid slots (occupancy BCE support)
    bce_e = F.binary_cross_entropy_with_logits(occ, occ_t, reduction="none")
    bce = (bce_e * m_occ).sum() / m_occ.sum().clamp(min=1)
    if logvar is not None:
        lvc = logvar.clamp(-12, 8)
        v_e = 0.5 * (((mu - tgt) ** 2) * torch.exp(-lvc) + lvc)
        if beta > 0:                                        # beta-NLL (Seitzer 2022): undo NLL var-down-weighting
            v_e = v_e * (torch.exp(lvc).detach() ** beta)
    else:
        v_e = (mu - tgt) ** 2
    act = occ_t.bool() & valid & mrow                       # masked & ACTIVE & valid slots
    if alpha > 0:                                           # magnitude-importance weight: w = cosh(tgt)^(2a)/VARB[band]
        w = torch.cosh(tgt.clamp(-12, 12)) ** (2 * alpha)   # cosh(tgt)^2 = 1+(coeff/sigma)^2  -> up-weights bright coeffs
        if varb is not None:
            w = w / varb[B["band_id"]][:, None]             # per-band normalize (makes train loss = eval metric)
        w = w * act.sum().clamp(min=1) / (w * act).sum().clamp(min=1)   # mean-normalize over active (keep loss scale)
        v_e = v_e * w
    val = (v_e * act).sum() / act.sum().clamp(min=1)
    if vis_w > 0:
        av = occ_t.bool() & valid & ~mrow
        val = val + vis_w * (v_e * av).sum() / av.sum().clamp(min=1)
    return bce, val


def losses_cat(occ, logits, B, tok_mask, edges, vis_w=0.0):
    """Categorical (discretized-bin) value loss: CE of the true tgt-bin per masked&active slot + occ BCE.
    logits (n_cells, n_slot, K); edges (n_band, K+1) per-band bin boundaries in tgt(asinh) space.
    Bounded, scale-free — no 1/sigma^2 down-weighting, no log-normal-mean pathology (charge = sum p*centroid)."""
    valid, occ_t, tgt = B["valid"], B["occ"], B["tgt"]         # dense (n_cells, n_slot)
    mrow = tok_mask[:, None]
    m_occ = mrow & valid
    bce_e = F.binary_cross_entropy_with_logits(occ, occ_t, reduction="none")
    bce = (bce_e * m_occ).sum() / m_occ.sum().clamp(min=1)
    K = logits.shape[-1]
    ec = edges[B["band_id"]]                                   # (n_cells, K+1) per-cell edges
    # NOTE (see TODO.md 1): this materialises an (n_cells, n_slot, K-1)
    # intermediate — ~4 GiB at a full 31-40k-cell event with K=128, which OOMs an
    # 11 GB card before the model is the constraint. torch.bucketize computes the
    # same thing with no intermediate; left as-is for now because this function is
    # extracted verbatim and the swap wants an equivalence test first.
    binid = (tgt.unsqueeze(-1) >= ec[:, None, 1:-1]).sum(-1).clamp(0, K - 1)   # (n_cells, n_slot) true bin
    ce_e = F.cross_entropy(logits.reshape(-1, K), binid.reshape(-1), reduction="none").view_as(tgt)
    act = occ_t.bool() & valid & mrow
    val = (ce_e * act).sum() / act.sum().clamp(min=1)
    if vis_w > 0:
        av = occ_t.bool() & valid & ~mrow
        val = val + vis_w * (ce_e * av).sum() / av.sum().clamp(min=1)
    return bce, val
