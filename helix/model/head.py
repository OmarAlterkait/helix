"""The categorical objective, evaluated only where it has weight.

``val_head`` is ``Linear(d, n_slot * n_bins)``, and ``losses_cat`` runs
``cross_entropy`` over its whole ``(n_cells, n_slot, n_bins)`` output. At
``vis_w == 0`` only masked rows enter the objective, and the value term weights
only their occupied, valid slots — **4.2 % of the grid** on the production
corpus. The rest was computed, upcast to fp32, saved for backward and multiplied
by zero: 4.8 GiB at a 32k-cell event, half the step's peak memory.

This computes the same sums over the same pairs: group the active pairs by slot,
pad each group to the largest, and do one ``baddbmm`` plus one
``cross_entropy``. Only the fp32 summation order differs from ``losses_cat``
(the loss moves by < 1e-5 relative; tests/test_serial.py).
"""
import torch
import torch.nn.functional as F

from helix.model.loss import bucketize_bins


def cat_head_sparse(model, feat, B, rows):
    """-> (occupancy BCE, value CE) over the masked rows ``rows`` of the batch,
    whose decoded features are ``feat``; the sums ``losses_cat`` forms at
    vis_w == 0."""
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
    cnt = torch.bincount(si[o], minlength=NS)
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
