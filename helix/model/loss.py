"""The categorical objective: ``losses_cat`` over the dense grid, and the
per-band binning it and :func:`helix.model.head.cat_head_sparse` share.
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
