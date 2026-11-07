"""The categorical objective: ``losses_cat`` over the dense grid, :func:`cat_head_sparse` over
only the slots with weight, and the per-band binning both share.

``cat_head_sparse``: ``val_head`` is ``Linear(d, n_slot * n_bins)``, and ``losses_cat`` runs
``cross_entropy`` over its whole ``(n_cells, n_slot, n_bins)`` output. Only
masked rows enter the objective, and the value term weights only their occupied,
valid slots — **4.2 % of the grid** on the production corpus. Over the whole
grid the rest is computed, upcast to fp32, saved for backward and multiplied by
zero: 4.8 GiB at a 32k-cell event, half the step's peak memory.

:func:`cat_head_sparse` computes the same sums over the same pairs: group the active pairs by slot,
pad each group to the largest, and do one ``baddbmm`` plus one
``cross_entropy``. Only the fp32 summation order differs from ``losses_cat``
(the loss moves by < 1e-5 relative; tests/test_attention.py).
"""
import torch
import torch.nn.functional as F


def bucketize_bins(tgt, band_id, edges, K):
    """True bin index per slot, ``(n_cells, n_slot)`` int64.

    A MODULE-LEVEL function rather than a loop inside ``losses_cat`` so that
    tests call the shipped code, not a copy of it.

    The direct expression::

        binid = (tgt.unsqueeze(-1) >= edges[band][:, None, 1:-1]).sum(-1)

    materialises ``(n_cells, n_slot, K-1)`` — the comparison is bool but
    ``.sum(-1)`` accumulates in int64, so at 8 B/element that is 4.66 GiB for a
    37k-cell event at K=128, enough to OOM an 11 GB card.

    ``edges`` has only ``n_band`` distinct rows, so bucketizing per band gives
    the identical index with no intermediate at all. ``right=True`` is the
    ``>=`` tie-break: a ``tgt`` landing exactly ON an edge goes up.
    """
    # A band with no row in `edges` would leave its slots UNWRITTEN below, and
    # torch.empty returns whatever was in memory — a loss that silently varies
    # between identical calls. Fail loudly instead.
    n_band = edges.shape[0]
    msg = (f"band_id >= {n_band}: `edges` has only {n_band} rows — the bin table "
           f"does not cover every band present. Derive edges with the same n_bands "
           f"the tokenizer emits (PatchConfig.n_bands).")
    if band_id.numel():
        # On GPU an async device assert: `int(band_id.max())` would be a host
        # sync every step (~20 % GPU idle at d768 with the per-band boolean
        # assignments it replaces). On CPU the loud IndexError.
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


def losses_cat(occ, logits, B, tok_mask, edges):
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
    return bce, val


def cat_head_sparse(model, feat, B, rows):
    """-> (occupancy BCE, value CE) over the masked rows ``rows`` of the batch,
    whose decoded features are ``feat``; the sums ``losses_cat`` forms."""
    NS, K = model.n_slot, model.n_bins
    valid, occ_t, tgt = B["valid"][rows], B["occ"][rows], B["tgt"][rows]

    occ = model.occ_head(feat) * model.readout_mult
    bce_e = F.binary_cross_entropy_with_logits(occ, occ_t, reduction="none")
    bce = (bce_e * valid).sum() / valid.sum().clamp(min=1)

    act = occ_t.bool() & valid
    denom = act.sum().clamp(min=1)
    ci, si = act.nonzero(as_tuple=True)
    if ci.numel() == 0:
        return bce, feat.sum() * 0

    binid = bucketize_bins(tgt[ci, si][:, None], B["band_id"][rows][ci],
                           model.bin_edges, K).squeeze(1)
    # Group by slot so one GEMM covers every pair sharing a val_head row block.
    o = torch.argsort(si)
    ci_s, bin_s = ci[o], binid[o]
    cnt = torch.zeros(NS, dtype=torch.long, device=si.device).scatter_add_(
        0, si, torch.ones_like(si))                         # bincount would sync for its size
    mx = int(cnt.max())
    starts = torch.cat([cnt.new_zeros(1), cnt.cumsum(0)[:-1]])
    col = torch.arange(mx, device=feat.device)[None, :]
    keep = col < cnt[:, None]                                  # (NS, mx)
    flat = (starts[:, None] + col).clamp(max=ci.numel() - 1)

    W = model.val_head.weight.view(NS, K, -1)
    bias = model.val_head.bias.view(NS, K)
    f = feat[ci_s[flat]]                                       # (NS, mx, d)
    lg = torch.baddbmm(bias[:, None, :].to(f.dtype), f,
                       W.transpose(1, 2).to(f.dtype)) * model.readout_mult
    ce = F.cross_entropy(lg.reshape(-1, K).float(), bin_s[flat].reshape(-1),
                         reduction="none")
    return bce, (ce.view(NS, mx) * keep).sum() / denom
