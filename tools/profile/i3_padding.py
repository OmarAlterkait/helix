"""I3 — is the attended padding actually costing anything, at the REAL operating point?

MULTI_EVENT_BATCHING.md measures the pad perturbation at T=30,976 (8% of tokens
moved >1%) and docs/REVIEW_FIELD.md §1.1 points out training runs the encoder at
T~7,625, where the same table's neighbouring row reads 56.3%. Neither number was
taken at the training token count with trained weights.

Three variants through the real 12-layer encoder:

  shipped    npad = ceil(T/g)*g, pads are duplicates of the last token, attended
  ceil-fix   grouped_cross's own formula: nb first, then gg = ceil(T/nb)
  masked     shipped blocks, pad KEYS masked out -- the reference the other two
             are perturbations of

Reported as the deviation each causes on the encoder output, in units of the
feature magnitude, and as the share of tokens moved by more than 1% and 10%.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prof"))
from common import build, emit, load_events, to_device
import helix.model.serial as S
from helix.model.fm import rope_angles

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="a trained checkpoint (.pth)")
ap.add_argument("--events", type=int, default=4)
A = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

model = build(device=dev)
R = {"ckpt": A.ckpt, "loaded": False}
if os.path.exists(A.ckpt):
    sd = torch.load(A.ckpt, map_location="cpu", weights_only=False)
    for k in ("state_dict", "model", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    R["loaded"] = True
    R["missing"] = len(missing); R["unexpected"] = len(unexpected)
    print(f"loaded {A.ckpt}: {len(missing)} missing, {len(unexpected)} unexpected",
          flush=True)
else:
    print("CHECKPOINT ABSENT -- running on random init, magnitudes not meaningful",
          flush=True)
model.eval()

_orig = S.uniform_attn


def attn_ceilfix(q, k, v, order, g):
    """grouped_cross's block geometry applied to self attention: pick nb from g,
    then size the block to fit, so the pad is at most nb-1 rows."""
    T, h, hd = q.shape
    nb = (T + g - 1) // g
    gg = (T + nb - 1) // nb
    npad = nb * gg
    idx = order[torch.arange(npad, device=order.device).clamp(max=T - 1)]
    grp = lambda x: x[idx].view(nb, gg, h, hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(grp(q), grp(k.to(q.dtype)), grp(v.to(q.dtype)))
    o = o.permute(0, 2, 1, 3).reshape(npad, h, hd)[:T]
    out = o.new_empty(T, h, hd); out[order] = o
    return out


def _masked(q, k, v, order, g, nb, gg):
    """A given block geometry with the duplicate pad KEYS masked out -- the
    reference THAT geometry is a perturbation of."""
    T, h, hd = q.shape
    npad = nb * gg
    g = gg
    idx = order[torch.arange(npad, device=order.device).clamp(max=T - 1)]
    grp = lambda x: x[idx].view(nb, g, h, hd).permute(0, 2, 1, 3)
    valid = (torch.arange(npad, device=q.device) < T).view(nb, 1, 1, g)
    o = F.scaled_dot_product_attention(grp(q), grp(k.to(q.dtype)), grp(v.to(q.dtype)),
                                       attn_mask=valid)
    o = o.permute(0, 2, 1, 3).reshape(npad, h, hd)[:T]
    out = o.new_empty(T, h, hd); out[order] = o
    return out


def attn_shipped_masked(q, k, v, order, g):
    T = q.shape[0]; npad = ((T + g - 1) // g) * g
    return _masked(q, k, v, order, g, npad // g, g)


def attn_ceilfix_masked(q, k, v, order, g):
    T = q.shape[0]; nb = (T + g - 1) // g
    return _masked(q, k, v, order, g, nb, (T + nb - 1) // nb)


def encode_visible(B, mask):
    vis = ~mask
    vi = vis.nonzero(as_tuple=True)[0]
    x = model._emb(B, vi)
    hd = model.d // model.heads
    at = rope_angles(B["t_phys"][vis], hd, *model.lam_t)
    aw = rope_angles(B["wire_pos"][vis], hd, *model.lam_w)
    sched = model._sched(B["plane_id"][vis], B["t_phys"][vis], B["wire_pos"][vis])
    for blk, (o, g, uw) in zip(model.enc, sched):
        x = S._self(blk, x, at, aw if uw else None, o, g, None)
    return x


evs, _ = load_events(A.events)
R["events"] = []
print(f"\nmean|feature| is the scale everything below is relative to\n", flush=True)
for e in evs:
    B = to_device({k: v for k, v in e.items() if torch.is_tensor(v)}, dev)
    B["n_cells"] = B["plane_id"].shape[0]
    g = torch.Generator(device=dev); g.manual_seed(0)
    mask = model.make_mask(B, mode="random", gen=g)
    T = int((~mask).sum())
    row = dict(name=e["_name"], n_cells=int(B["n_cells"]), T_visible=T,
               pad_gp=((T + model.gp - 1)//model.gp)*model.gp - T,
               pad_gd=((T + model.gd - 1)//model.gd)*model.gd - T)
    outs = {}
    for tag, fn in (("shipped_masked", attn_shipped_masked),
                    ("ceilfix_masked", attn_ceilfix_masked),
                    ("shipped", _orig), ("ceil-fix", attn_ceilfix)):
        S.uniform_attn = fn
        try:
            with torch.no_grad(), torch.autocast(dev, torch.bfloat16, enabled=(dev == "cuda")):
                outs[tag] = encode_visible(B, mask).float()
        finally:
            S.uniform_attn = _orig
    scale = outs["shipped_masked"].abs().mean()
    row["max_abs_value"] = {t: float(o.abs().max()) for t, o in outs.items()}
    row["mean_abs_value"] = {t: float(o.abs().mean()) for t, o in outs.items()}
    COMPARE = (("shipped", "shipped_masked", "padding effect, shipped geometry"),
               ("ceil-fix", "ceilfix_masked", "padding effect, ceil geometry"),
               ("ceilfix_masked", "shipped_masked", "partition change, no padding"))
    for tag, reftag, what in COMPARE:
        ref = outs[reftag]
        d = (outs[tag] - ref).abs()
        tok = d.max(1).values / scale
        key = f"{tag}_vs_{reftag}"
        row[key] = dict(what=what, mean_feat=float(scale), max_abs=float(d.max()),
                        frac_gt_1pct=float((tok > 0.01).float().mean()),
                        frac_gt_10pct=float((tok > 0.10).float().mean()),
                        p50_rel=float(tok.median()), p99_rel=float(tok.quantile(0.99)))
        print(f"{e['_name'][:8]:8s} {T:6d} {row['pad_gp']:6d} {what:34s} "
              f"max|d| {d.max():8.3f}  p50 {tok.median():7.4f}  p99 {tok.quantile(0.99):7.4f}  "
              f">1% {100*row[key]['frac_gt_1pct']:5.1f}%  >10% {100*row[key]['frac_gt_10pct']:5.1f}%",
              flush=True)
    R["events"].append(row)

print()
for key in ("shipped_vs_shipped_masked", "ceil-fix_vs_ceilfix_masked",
            "ceilfix_masked_vs_shipped_masked"):
    f1 = float(np.mean([r[key]["frac_gt_1pct"] for r in R["events"]]))
    f10 = float(np.mean([r[key]["frac_gt_10pct"] for r in R["events"]]))
    mx = float(np.mean([r[key]["max_abs"] for r in R["events"]]))
    p50 = float(np.mean([r[key]["p50_rel"] for r in R["events"]]))
    R.setdefault("summary", {})[key] = dict(frac_gt_1pct=f1, frac_gt_10pct=f10,
                                            max_abs=mx, p50_rel=p50)
    print(f"  {R['events'][0][key]['what']:34s} max|d| {mx:8.3f}  p50 {p50:7.4f}  "
          f">1% {100*f1:5.1f}%  >10% {100*f10:5.1f}%")
print("\n  activation magnitudes (sanity: a broken reference shows up here):")
for t in R["events"][0]["max_abs_value"]:
    print(f"    {t:16s} mean|f| {R['events'][0]['mean_abs_value'][t]:8.4f}  "
          f"max|f| {R['events'][0]['max_abs_value'][t]:10.3f}")
emit("i3_padding", R)
