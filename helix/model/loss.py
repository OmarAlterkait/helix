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


def bucketize_bins(tgt, band_id, edges, K):
    """True bin index per slot, ``(n_cells, n_slot)`` int64.

    A MODULE-LEVEL function rather than a loop inside ``losses_cat`` so that the
    test can call the shipped code. tests/test_losses_cat_binning.py used to
    carry its own copy of this loop labelled "what losses_cat now does" and
    compare THAT to the reference — so the tie-break and the guard below were
    pinned on a duplicate, and changing the real one would not have failed
    anything.

    The original expression was::

        binid = (tgt.unsqueeze(-1) >= edges[band][:, None, 1:-1]).sum(-1)

    which materialises ``(n_cells, n_slot, K-1)`` — the comparison is bool but
    ``.sum(-1)`` accumulates in int64, so at 8 B/element that is 4.66 GiB for a
    37k-cell event at K=128. It OOMed an 11 GB card mid-run, on the first real
    FMTrainer launch, at step 4.

    ``edges`` has only ``n_band`` distinct rows, so bucketizing per band gives
    the identical index with no intermediate at all. ``right=True`` reproduces
    the original's ``>=`` tie-break: a ``tgt`` landing exactly ON an edge goes
    up.
    """
    # A band with no row in `edges` would leave its slots UNWRITTEN below, and
    # torch.empty returns whatever was in memory — a loss that silently varies
    # between identical calls. The original indexed `edges[band_id]` directly and
    # raised IndexError; fail the same way, loudly.
    n_band = edges.shape[0]
    msg = (f"band_id >= {n_band}: `edges` has only {n_band} rows — the bin table "
           f"does not cover every band present. Derive edges with the same n_bands "
           f"the tokenizer emits (PatchConfig.n_bands).")
    if band_id.numel():
        # On GPU an async device assert: `int(band_id.max())` was a host sync every
        # step (13 per step with the boolean assignments below, ~20 % GPU idle at
        # d768). On CPU the loud IndexError stays.
        if band_id.is_cuda:
            torch._assert_async(band_id.max() < n_band, msg)
        elif int(band_id.max()) >= n_band:
            raise IndexError(f"band_id up to {int(band_id.max())} but " + msg)
    # Bucketize the whole tensor per band and select with `where`: `binid[sel] = ...`
    # with a boolean `sel` is a nonzero(), i.e. another host sync per band.
    bb = band_id.reshape(band_id.shape + (1,) * (tgt.dim() - band_id.dim()))
    binid = torch.zeros(tgt.shape, dtype=torch.long, device=tgt.device)
    for b in range(n_band):
        binid = torch.where(bb == b, torch.bucketize(tgt, edges[b, 1:-1].contiguous(), right=True), binid)
    return binid.clamp(0, K - 1)


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
    binid = bucketize_bins(tgt, B["band_id"], edges, K)        # (n_cells, n_slot)
    ce_e = F.cross_entropy(logits.reshape(-1, K), binid.reshape(-1), reduction="none").view_as(tgt)
    act = occ_t.bool() & valid & mrow
    val = (ce_e * act).sum() / act.sum().clamp(min=1)
    if vis_w > 0:
        av = occ_t.bool() & valid & ~mrow
        val = val + vis_w * (ce_e * av).sum() / av.sum().clamp(min=1)
    return bce, val
