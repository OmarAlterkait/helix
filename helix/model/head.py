"""The categorical value head, evaluated only where the objective weights it.

``val_head`` is ``Linear(d, n_slot * n_bins)`` and ``losses_cat`` runs
``cross_entropy`` over its whole ``(n_cells, n_slot, n_bins)`` output. But the
value term weights only ``occ & valid & masked`` slots, and on the production
corpus that is **4.2 % of the grid**::

    grid slots                3,910,272
    slots the loss weights      165,324   (1 / 23.7)

The other 95.8 % are computed, upcast to fp32 by autocast's ``cross_entropy``
policy, saved for backward, and multiplied by zero. Measured cost: 10.1 bytes
per logit element, **4.8 GiB at a 32k-cell event, 51 % of the step's peak
memory**, and the shape that OOMed an 11 GB card at step 4 (see
``loss.bucketize_bins``'s docstring).

This computes the same sum over the same pairs: group the active pairs by slot,
pad each slot-group to the batch maximum, and do one ``baddbmm`` plus one
``cross_entropy`` — four kernels rather than four per slot. Measured on an A100:
**1.16x faster, 0.55x the memory** (``docs/PERFORMANCE.md`` §7).

It is not bit-exact. Only the summation order and the GEMM shape change, so the
difference is fp32 accumulation: across five real events the loss moves by
**-0.0001 % to -0.0008 %**, and the gradient difference (6.9e-3 max relative)
sits INSIDE the step's own run-to-run nondeterminism (6.2e-3 for re-running the
identical dense step twice — SDPA's backward accumulates with atomics).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from helix.model.loss import bucketize_bins


def cat_head_sparse(model, feat, B, tok_mask, rows=None):
    """-> (occupancy BCE, value CE), same sums as ``losses_cat`` at vis_w == 0.

    ``rows`` names which batch rows ``feat`` covers, for the masked-only forward
    (``fastpath.forward_feat(..., masked_only=True)``). ``None`` means every row,
    which is the ordinary contract.
    """
    if model.vis_w:
        raise ValueError("cat_head_sparse implements vis_w == 0 only; the "
                         "visible term needs the complementary pair set")
    NS, K = model.n_slot, model.n_bins
    sel = slice(None) if rows is None else rows
    valid, occ_t, tgt = B["valid"][sel], B["occ"][sel], B["tgt"][sel]
    band = B["band_id"][sel]
    mrow = True if rows is not None else tok_mask[:, None]

    occ = model.occ_head(feat) * model.readout_mult
    m_occ = (valid & mrow) if rows is None else valid
    bce_e = F.binary_cross_entropy_with_logits(occ, occ_t, reduction="none")
    bce = (bce_e * m_occ).sum() / m_occ.sum().clamp(min=1)

    act = occ_t.bool() & valid
    if rows is None:
        act = act & mrow
    denom = act.sum().clamp(min=1)
    ci, si = act.nonzero(as_tuple=True)
    if ci.numel() == 0:
        return bce, feat.sum() * 0

    binid = bucketize_bins(tgt[ci, si][:, None], band[ci], model.bin_edges, K).squeeze(1)
    # Group by slot so one GEMM covers every pair sharing a val_head row block.
    o = torch.argsort(si)
    ci_s, bin_s = ci[o], binid[o]
    cnt = torch.bincount(si[o], minlength=NS)
    mx = int(cnt.max())
    starts = torch.cat([cnt.new_zeros(1), cnt.cumsum(0)[:-1]])
    col = torch.arange(mx, device=feat.device)[None, :]
    keep = col < cnt[:, None]                                  # (NS, mx)
    flat = (starts[:, None] + col).clamp(max=max(ci.numel() - 1, 0))

    W = model.val_head.weight.view(NS, K, -1)
    bias = model.val_head.bias.view(NS, K)
    f = feat[ci_s[flat]]                                       # (NS, mx, d)
    lg = torch.baddbmm(bias[:, None, :].to(f.dtype), f,
                       W.transpose(1, 2).to(f.dtype)) * model.readout_mult
    ce = F.cross_entropy(lg.reshape(-1, K).float(), bin_s[flat].reshape(-1),
                         reduction="none")
    return bce, (ce.view(NS, mx) * keep).sum() / denom
