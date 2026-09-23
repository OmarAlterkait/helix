"""I6 — the four free on-checkpoint tests, and the metric that makes them decidable.

The probe resolves ~0.013 at best on this corpus (per-group r std 0.344 over at
most 388 events). Every objective-side change has an expected effect below that,
so the probe CANNOT adjudicate them. CRPS and the per-slot CE are means over
millions of token-slots, so their standard error is orders of magnitude tighter.
This script establishes those, then uses them for the three tests that need a
metric.

  A. CRPS, CE and calibration for the categorical head -- the metric every
     later objective experiment is judged on.
  B. ORACLE READ-BACK: feed the TRUE bin one-hot through the centroid table.
     Closes the bin-count question permanently (expected var_expl >= 0.999).
  C. PAD-MASK A/B at inference. If masking the duplicate pad keys IMPROVES the
     held-out metric, the padding is hurting weights that were trained with it;
     if it degrades, the weights absorbed it and removing it needs a retrain.
     Either answer settles whether the varlen work is worth doing.
  D. CARRIER RECONSTRUCTION. The actual Darcet test: are the massive-activation
     tokens' OWN contents reconstructed worse? Matched on active-slot count,
     because carriers are low-occupancy and low-occupancy cells differ anyway.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prof"))
from common import build, emit, load_events, to_device
import helix.model.serial as S
from helix.model.loss import bucketize_bins

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", default="/sdf/data/neutrino/omara/archive/"
                                      "fm_coolbase_r1_8run_artifact")
ap.add_argument("--events", type=int, default=24)
ap.add_argument("--start", type=int, default=19000, help="held-out end of the corpus")
A = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

from helix.model import artifact as _art
_a = _art.load(A.artifact)
model = _art.build(_a, device=dev)
print("operating point:", _a.op)
print(f"loaded artifact: d={model.d} blocks={len(model.enc)} n_bins={model.n_bins}",
      flush=True)
K, NS = model.n_bins, model.n_slot
R = {"artifact": A.artifact, "events": A.events}

edges = model.bin_edges.float()                       # (n_band, K+1)
cent_a = model.bin_cent_asinh.float()                 # (n_band, K)
# The OPEN outer bins are stored as +-1e18, not +-inf -- `helix/data/bins.py`'s
# docstring says "extended to +-inf" but the serialised sentinel is finite, so
# torch.isfinite does NOT identify them and a CRPS integral over their width
# returns ~1e14. Detect by magnitude and give them the median closed width.
OPEN = 1e6
closed = edges.abs() < OPEN
w = (edges[:, 1:] - edges[:, :-1])
wmed = w[closed[:, 1:] & closed[:, :-1]].median()
lo = torch.where(closed[:, :-1], edges[:, :-1], edges[:, 1:2] - wmed)
hi = torch.where(closed[:, 1:], edges[:, 1:], edges[:, -2:-1] + wmed)
assert bool((hi > lo).all()), "outer-bin substitution produced a non-positive width"
R["crps_outer_bin_width"] = float(wmed)


def crps_binned(p, band, y):
    """CRPS of a binned predictive distribution against a scalar target.

    integral over x of (F(x) - 1{x >= y})^2, with F a step function on the bin
    edges. O(K) per sample, exact for this forecast form.
    """
    l, h = lo[band], hi[band]                         # (P, K)
    Fc = p.cumsum(-1)
    yb = y[:, None]
    below = h <= yb
    above = l >= yb
    inside = ~(below | above)
    out = torch.where(below, Fc ** 2 * (h - l), torch.zeros_like(Fc))
    out = out + torch.where(above, (Fc - 1) ** 2 * (h - l), torch.zeros_like(Fc))
    split = Fc ** 2 * (yb - l).clamp(min=0) + (Fc - 1) ** 2 * (h - yb).clamp(min=0)
    out = out + torch.where(inside, split, torch.zeros_like(Fc))
    return out.sum(-1)


_orig = S.uniform_attn


def attn_padmasked(q, k, v, order, g):
    T, h, hd = q.shape
    npad = ((T + g - 1) // g) * g
    nb = npad // g
    idx = order[torch.arange(npad, device=order.device).clamp(max=T - 1)]
    grp = lambda x: x[idx].view(nb, g, h, hd).permute(0, 2, 1, 3)
    valid = (torch.arange(npad, device=q.device) < T).view(nb, 1, 1, g)
    o = F.scaled_dot_product_attention(grp(q), grp(k.to(q.dtype)), grp(v.to(q.dtype)),
                                       attn_mask=valid)
    o = o.permute(0, 2, 1, 3).reshape(npad, h, hd)[:T]
    out = o.new_empty(T, h, hd); out[order] = o
    return out


def head_stats(B, mask):
    """CE, CRPS, calibration and var_expl over the MASKED active slots."""
    with torch.no_grad(), torch.autocast(dev, torch.bfloat16, enabled=(dev == "cuda")):
        feat = model.forward_feat(B, mask)
        logits = (model.val_head(feat) * model.readout_mult).view(-1, NS, K)
    act = B["occ"].bool() & B["valid"] & mask[:, None]
    ci, si = act.nonzero(as_tuple=True)
    band = B["band_id"][ci]
    tgt = B["tgt"][ci, si].float()
    lg = logits[ci, si].float()
    p = lg.softmax(-1)
    binid = bucketize_bins(tgt[:, None], band, model.bin_edges, K).squeeze(1)
    ce = F.cross_entropy(lg, binid, reduction="none")
    crps = crps_binned(p, band, tgt)
    mu = (p * cent_a[band]).sum(-1)                   # posterior-mean read-back
    var_expl = 1 - ((mu - tgt) ** 2).sum() / ((tgt - tgt.mean()) ** 2).sum()
    # calibration: rank of the true bin under the predictive CDF (PIT)
    pit = (p.cumsum(-1).gather(1, binid[:, None]).squeeze(1)
           - 0.5 * p.gather(1, binid[:, None]).squeeze(1))
    return dict(n=int(ci.numel()), ce=float(ce.mean()), crps=float(crps.mean()),
                var_expl=float(var_expl), pit_mean=float(pit.mean()),
                pit_std=float(pit.std())), (ci, si, band, tgt, ce)


evs, _ = load_events(A.events, start=A.start)
print(f"held-out events from index {A.start}", flush=True)

# ---------------------------------------------- A + C: metric, and the pad A/B
R["pad_ab"] = {}
agg = {}
for tag, fn in (("shipped", _orig), ("pad-masked", attn_padmasked)):
    S.uniform_attn = fn
    tot = dict(n=0, ce=0.0, crps=0.0, var_expl=[], pit_mean=[], pit_std=[])
    try:
        for i, e in enumerate(evs):
            B = to_device({k: v for k, v in e.items() if torch.is_tensor(v)}, dev)
            B["n_cells"] = B["plane_id"].shape[0]
            g = torch.Generator(device=dev); g.manual_seed(1000 + i)
            m = model.make_mask(B, mode="random", gen=g)
            s, _ = head_stats(B, m)
            tot["n"] += s["n"]; tot["ce"] += s["ce"] * s["n"]; tot["crps"] += s["crps"] * s["n"]
            tot["var_expl"].append(s["var_expl"]); tot["pit_mean"].append(s["pit_mean"])
            tot["pit_std"].append(s["pit_std"])
    finally:
        S.uniform_attn = _orig
    R["pad_ab"][tag] = dict(n_slots=tot["n"], ce=tot["ce"]/tot["n"],
                            crps=tot["crps"]/tot["n"],
                            var_expl=float(np.mean(tot["var_expl"])),
                            var_expl_sem=float(np.std(tot["var_expl"])/np.sqrt(len(evs))),
                            pit_mean=float(np.mean(tot["pit_mean"])),
                            pit_std=float(np.mean(tot["pit_std"])))
    r = R["pad_ab"][tag]
    print(f"  {tag:11s} n={r['n_slots']:>9,}  CE {r['ce']:.5f}  CRPS {r['crps']:.5f}  "
          f"var_expl {r['var_expl']:.4f}+-{r['var_expl_sem']:.4f}  "
          f"PIT {r['pit_mean']:.3f}+-{r['pit_std']:.3f}", flush=True)
a, b = R["pad_ab"]["shipped"], R["pad_ab"]["pad-masked"]
R["pad_ab"]["delta"] = dict(ce=b["ce"]-a["ce"], crps=b["crps"]-a["crps"],
                            var_expl=b["var_expl"]-a["var_expl"])
print(f"\n  pad-masked - shipped:  dCE {b['ce']-a['ce']:+.5f}  "
      f"dCRPS {b['crps']-a['crps']:+.5f}  dvar_expl {b['var_expl']-a['var_expl']:+.5f}")
print("  READ: negative dCE/dCRPS means masking the pads HELPS weights trained")
print("  with them -- the padding is a live defect, not something absorbed.")

# ------------------------------------------------------- B: oracle read-back
tot_n = tot_se = tot_var = 0.0; ys = []
for i, e in enumerate(evs):
    B = to_device({k: v for k, v in e.items() if torch.is_tensor(v)}, dev)
    B["n_cells"] = B["plane_id"].shape[0]
    act = B["occ"].bool() & B["valid"]
    ci, si = act.nonzero(as_tuple=True)
    band = B["band_id"][ci]; tgt = B["tgt"][ci, si].float()
    binid = bucketize_bins(tgt[:, None], band, model.bin_edges, K).squeeze(1)
    oracle = cent_a[band].gather(1, binid[:, None]).squeeze(1)   # true bin -> centroid
    tot_se += float(((oracle - tgt) ** 2).sum()); ys.append(tgt.cpu())
ys = torch.cat(ys)
R["oracle_readback"] = dict(var_expl=float(1 - tot_se / float(((ys - ys.mean())**2).sum())),
                            n=int(ys.numel()))
print(f"\n  oracle read-back var_expl = {R['oracle_readback']['var_expl']:.6f} "
      f"over {R['oracle_readback']['n']:,} slots")
print("  READ: >=0.999 means bin RESOLUTION costs nothing; cross it off.")

# ---------------------------------------------- D: do carriers reconstruct worse?
print("\n  carrier vs non-carrier reconstruction, matched on active-slot count",
      flush=True)
rows = []
for i, e in enumerate(evs[:8]):
    B = to_device({k: v for k, v in e.items() if torch.is_tensor(v)}, dev)
    B["n_cells"] = B["plane_id"].shape[0]
    with torch.no_grad(), torch.autocast(dev, torch.bfloat16, enabled=(dev == "cuda")):
        x = model.encode_layers(B, {len(model.enc)})[len(model.enc)].float()
    a = x.abs(); med = a.median()
    carrier = (a.max(1).values > 100 * med)                       # I4's definition
    g = torch.Generator(device=dev); g.manual_seed(1000 + i)
    m = model.make_mask(B, mode="random", gen=g)
    _, (ci, si, band, tgt, ce) = head_stats(B, m)
    nact = (B["occ"].bool() & B["valid"]).sum(1)
    # per-cell mean CE over its masked active slots
    cell_ce = torch.zeros(B["n_cells"], device=dev).index_add_(0, ci, ce)
    cell_n = torch.zeros(B["n_cells"], device=dev).index_add_(0, ci, torch.ones_like(ce))
    ok = (cell_n > 0) & m
    for occ_k in (1, 2, 3):
        sel = ok & (nact == occ_k)
        c1 = sel & carrier; c0 = sel & ~carrier
        if int(c1.sum()) < 5 or int(c0.sum()) < 5:
            continue
        rows.append(dict(event=i, occ=occ_k,
                         n_carrier=int(c1.sum()), n_other=int(c0.sum()),
                         ce_carrier=float((cell_ce[c1]/cell_n[c1]).mean()),
                         ce_other=float((cell_ce[c0]/cell_n[c0]).mean())))
R["carrier_recon"] = rows
if rows:
    for occ_k in (1, 2, 3):
        rs = [r for r in rows if r["occ"] == occ_k]
        if not rs:
            continue
        cc = float(np.mean([r["ce_carrier"] for r in rs]))
        co = float(np.mean([r["ce_other"] for r in rs]))
        nc = int(np.sum([r["n_carrier"] for r in rs]))
        print(f"    active slots = {occ_k}:  carrier CE {cc:.4f} (n={nc})  "
              f"vs other CE {co:.4f}   delta {cc-co:+.4f}")
    R["carrier_summary"] = {f"occ{k}": dict(
        carrier=float(np.mean([r["ce_carrier"] for r in rows if r["occ"] == k])),
        other=float(np.mean([r["ce_other"] for r in rows if r["occ"] == k])))
        for k in (1, 2, 3) if any(r["occ"] == k for r in rows)}
print("  READ: carriers reconstructing WORSE at matched occupancy is the Darcet")
print("  signature -- the model sacrificed those tokens to use as scratch space.")

emit("i6_metrics", R)
