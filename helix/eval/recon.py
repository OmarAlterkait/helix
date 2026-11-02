"""Masked reconstruction: the training objective and its grid-free read-backs, on held-out events.

ONE implementation, used twice: :class:`helix.integrations.pimm.eval.CoeffFMEvaluator` (the
validation pass inside a training run, sums all-reduced across ranks) and :class:`Recon` (the
``recon`` metric of ``python -m helix.eval``, one process over the probe split). Both mask with
the same seeded generator per batch index, so the same event is masked the same way by both.

Reported (pooled over tokens, never a mean of per-event means: dense events are both heavier
and harder, corr(per-event CE, token count) = 0.79, so the unweighted mean is ~0.09 nats
optimistic):

  loss = bce + val   occupancy BCE and value cross-entropy, each over its own support
  var_expl           1 - MSE / Var of the posterior-mean asinh reconstruction against the
                     target, on masked & valid & occupied slots -- grid-free, so it survives a
                     change of bin table where cross-entropy does not
  charge_closure     sum E[|X|] / sum |X_true| (X = coeff/sigma), nominal 1.0
  charge_resid       (sum E[X] - sum X_true) / sum |X_true|, nominal 0.0

No pimm import.
"""

from __future__ import annotations

import contextlib

import torch

#: Grid-free accumulators, summed per batch (and across ranks).
GF_KEYS = ("sse", "sy", "syy", "nv", "chg_pred", "chg_true", "chg_pred_s", "chg_true_s")
#: The pooled loss terms; each is a weighted sum over its own support.
POOLED = ("bce", "val")
#: The evaluator's default mask seed; batch i of rank r is masked with seed + 100003 r + i.
MASK_SEED = 7


def acc_grid_free(core, B, mask, logits, gf):
    """Accumulate var_expl / charge-closure sums for one batch.

    Both are computed on the SAME support the value loss uses (masked & valid & truly occupied)
    and pooled over tokens by the caller. Two read-outs, deliberately: ``cent_asinh`` -> the
    posterior mean in TOKEN space, for var_expl (the target's own space, no sinh); ``cent_ratio``
    -> the posterior mean of ``raw/sigma`` = E[sinh t], for charge (sinh of the asinh-space mean
    is Jensen-biased ~31% low; see ``tokenize.decode_categorical``).

    Charge uses E[|X|], NOT |E[X]|: coefficients are near-symmetric about zero, so |E[X]| shrinks
    toward zero with posterior width and the ratio falls below 1 for a perfectly calibrated model
    (measured on real checkpoints: 0.8177 against 0.9676, oracle 0.9991).
    """
    ca, cr = core.bin_cent_asinh, core.bin_cent_ratio
    tgt, occ_t, valid = B["tgt"], B["occ"].bool(), B["valid"].bool()
    sel = mask[:, None] & valid & occ_t
    if not bool(sel.any()):
        return
    band = B["band_id"].long()
    # einsum is on autocast's lower-precision list (bf16 there, ~3e-3 relative); autocast off
    # keeps both its memory and float32
    with torch.autocast(device_type=logits.device.type, enabled=False):
        p = torch.softmax(logits.float(), -1)                 # (n_cells, n_slot, K)
        crb = cr[band].float()
        rec_a = torch.einsum("csk,ck->cs", p, ca[band].float())
        rec_r = torch.einsum("csk,ck->cs", p, crb)            # E[X]
        rec_abs = torch.einsum("csk,ck->cs", p, crb.abs())    # E[|X|]
    y = tgt.float()[sel]
    d = rec_a[sel] - y
    gf["sse"] += float((d * d).sum())
    gf["sy"] += float(y.sum())
    gf["syy"] += float((y * y).sum())
    gf["nv"] += float(sel.sum())
    t_sel = torch.sinh(y)
    gf["chg_pred"] += float(rec_abs[sel].sum())
    gf["chg_true"] += float(t_sel.abs().sum())
    gf["chg_pred_s"] += float(rec_r[sel].sum())
    gf["chg_true_s"] += float(t_sel.sum())


def heads_loss(core, B, mask, gf):
    """The objective AND the grid-free sums from ONE ``raw_heads`` call.

    ``FMModel.forward`` plus a second ``raw_heads`` for the metrics would be a second full
    encoder+decoder pass, and evaluation is ~15% of a run's wall time already. So the
    objective is reproduced here from a single pass, with ``forward``'s own batch contract.
    """
    from helix.model.loss import losses_cat
    assert torch.isfinite(core.bin_edges).all(), \
        "evaluation requires set_bins(edges) first -- the edges buffer is still unset (NaN)."
    core.require_batch_keys(B)
    occ, val = core.raw_heads(B, mask)
    bce, vloss = losses_cat(occ, val, B, mask, core.bin_edges)
    acc_grid_free(core, B, mask, val, gf)
    return {"loss": bce + vloss, "bce": bce.detach(), "val": vloss.detach()}


def pool(out, B, mask, totals, counts):
    """Add one batch's loss terms to ``totals``/``counts``, weighted by their supports: each term
    already divided by its own support inside the loss, so the support is its weight."""
    mrow = mask[:, None]
    w = {"bce": float((mrow & B["valid"].bool()).sum()),
         "val": float((B["occ"].bool() & B["valid"].bool() & mrow).sum())}
    for k in POOLED:
        totals[k] = totals.get(k, 0.0) + float(out[k]) * w[k]
        counts[k] = counts.get(k, 0.0) + w[k]


def average(totals, counts, gf):
    """The reported metrics from the (reduced) sums."""
    avg = {k: v / max(counts.get(k, 0.0), 1e-9) for k, v in totals.items()}
    avg["loss"] = avg.get("bce", 0.0) + avg.get("val", 0.0)       # rebuilt: different denominators
    if gf["nv"] > 0:
        mean_y = gf["sy"] / gf["nv"]
        var_y = max(gf["syy"] / gf["nv"] - mean_y * mean_y, 1e-12)
        avg["var_expl"] = 1.0 - (gf["sse"] / gf["nv"]) / var_y
        if gf["chg_true"] > 1e-9:
            avg["charge_closure"] = gf["chg_pred"] / gf["chg_true"]
            # the signed RESIDUAL over the unsigned total: the signed ratio is a small
            # difference of large numbers and ill-conditioned
            avg["charge_resid"] = (gf["chg_pred_s"] - gf["chg_true_s"]) / gf["chg_true"]
    return avg


def mask_for(core, B, i, rank=0, seed=MASK_SEED, mode="random", ratio=None, n_planes=1):
    """The evaluation mask of batch ``i`` on ``rank``: the same every evaluation."""
    gen = torch.Generator(device=B["plane_id"].device).manual_seed(seed + 100003 * rank + i)
    return core.make_mask(B, mode=mode, ratio=ratio, n_planes=n_planes, gen=gen)


def autocast(device, dtype=torch.bfloat16, enabled=True):
    return torch.autocast(device_type=device.type, dtype=dtype) if enabled and device.type == "cuda" \
        else contextlib.nullcontext()


class Recon:
    """The ``recon`` metric: :func:`heads_loss` over the evaluated events, pooled."""

    name = "recon"

    def __init__(self, ctx):
        self.core = ctx.run.model
        self.totals, self.counts, self.n = {}, {}, 0
        self.gf = dict.fromkeys(GF_KEYS, 0.0)

    def add(self, ev):
        B = ev.batch
        mask = mask_for(self.core, B, ev.index)
        with torch.no_grad(), autocast(B["plane_id"].device):
            out = heads_loss(self.core, B, mask, self.gf)
        pool(out, B, mask, self.totals, self.counts)
        self.n += 1

    def result(self):
        return dict(average(self.totals, self.counts, self.gf), n_events=self.n)
