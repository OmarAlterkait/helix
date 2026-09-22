"""I2 — can a cross-plane pair actually reach each other?

The grouped attention sorts tokens by a physics key and chops the sorted list
into contiguous blocks of `g`. Two tokens in different blocks do not interact in
that layer, at all. The science the model exists for is cross-plane
triangulation: the same 3D deposit seen in U, V and Y of one TPC volume.

So the question that decides whether the boundary findings in
docs/REVIEW_FIELD.md §3 matter is measurable with no training:

  for token pairs that are CANDIDATE cross-plane partners -- same volume,
  different view, coincident in TOFF-corrected drift time -- what fraction land
  in the same attention block, per layer, and as a union over the 12-layer
  schedule?

Answered for the training regime (encoder over the visible 25%) and the probe
regime (encoder over all tokens), because those are different token counts at
the same fixed `g` -- §1.2.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prof"))
from common import M113, build, emit, load_events

ap = argparse.ArgumentParser()
ap.add_argument("--events", type=int, default=6)
ap.add_argument("--dt", type=float, default=8.0, help="drift-time coincidence window (ticks)")
ap.add_argument("--max_pairs", type=int, default=400000)
A = ap.parse_args()
VIEWS = 3

evs, _ = load_events(A.events)
model = build(device="cpu")          # weights irrelevant: _sched reads geometry only
R = {"dt_window": A.dt, "gp": model.gp, "gd": model.gd, "events": []}


def candidate_pairs(plane, t, rng, max_pairs):
    """Same volume, different view, |dt| <= window. t_phys is TOFF-corrected so
    the three views of one volume share a zero (tokenize.assemble)."""
    vol = (plane // VIEWS).numpy()
    view = (plane % VIEWS).numpy()
    tt = t.numpy()
    key = vol.astype(np.int64) * 10**7 + np.round(tt / A.dt).astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    ii, jj = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        grp = order[a:b]
        if len(grp) < 2:
            continue
        v = view[grp]
        for x in range(len(grp)):
            m = v != v[x]
            if not m.any():
                continue
            part = grp[m]
            ii.append(np.full(len(part), grp[x])); jj.append(part)
    if not ii:
        return np.empty(0, int), np.empty(0, int)
    ii = np.concatenate(ii); jj = np.concatenate(jj)
    keep = ii < jj
    ii, jj = ii[keep], jj[keep]
    if len(ii) > max_pairs:
        s = rng.choice(len(ii), max_pairs, replace=False)
        ii, jj = ii[s], jj[s]
    return ii, jj


def recall(model, plane, t, wire, ii, jj, g_scale=1.0):
    """Per-layer and union co-block fractions, own-block and own+neighbour."""
    gp, gd = model.gp, model.gd
    model.gp, model.gd = int(round(gp * g_scale)), int(round(gd * g_scale))
    try:
        sched = model._sched(plane, t, wire)
    finally:
        model.gp, model.gd = gp, gd
    T = plane.shape[0]
    own = np.zeros(len(ii), bool); nbr = np.zeros(len(ii), bool)
    per = []
    for o, g, _uw in sched:
        pos = np.empty(T, np.int64)
        pos[o.numpy()] = np.arange(T)
        bi, bj = pos[ii] // g, pos[jj] // g
        so = bi == bj
        sn = np.abs(bi - bj) <= 1
        own |= so; nbr |= sn
        per.append((float(so.mean()), float(sn.mean())))
    return dict(per_layer_own=[p[0] for p in per],
                per_layer_neighbour=[p[1] for p in per],
                union_own=float(own.mean()), union_neighbour=float(nbr.mean()))


rng = np.random.default_rng(0)
for e in evs[:A.events]:
    plane, t, wire = e["plane_id"], e["t_phys"], e["wire_pos"]
    N = plane.shape[0]
    gen = torch.Generator().manual_seed(0)
    mask = torch.rand(N, generator=gen) < 0.75            # production ratio
    vis = ~mask
    row = dict(n_cells=int(N), n_visible=int(vis.sum()))

    for tag, sel, scale in (("train (visible 25%)", vis, 1.0),
                            ("probe (all tokens)", torch.ones(N, dtype=torch.bool), 1.0),
                            ("probe, g scaled to match", torch.ones(N, dtype=torch.bool),
                             float(N) / float(vis.sum()))):
        p_, t_, w_ = plane[sel], t[sel], wire[sel]
        ii, jj = candidate_pairs(p_, t_, rng, A.max_pairs)
        if len(ii) == 0:
            row[tag] = dict(pairs=0); continue
        r = recall(model, p_, t_, w_, ii, jj, scale)
        r["pairs"] = int(len(ii)); r["tokens"] = int(sel.sum()); r["g_scale"] = scale
        row[tag] = r
        print(f"  {e['_name'][:22]:22s} {tag:26s} T={int(sel.sum()):6d} "
              f"pairs={len(ii):7d}  union own {r['union_own']:.3f}  "
              f"union own+nbr {r['union_neighbour']:.3f}", flush=True)
    R["events"].append(row)

# ---- sweep g at the training regime -------------------------------------
print("\n== block size sweep, training regime ==", flush=True)
e = evs[0]
plane, t, wire = e["plane_id"], e["t_phys"], e["wire_pos"]
N = plane.shape[0]
gen = torch.Generator().manual_seed(0)
vis = ~(torch.rand(N, generator=gen) < 0.75)
p_, t_, w_ = plane[vis], t[vis], wire[vis]
ii, jj = candidate_pairs(p_, t_, rng, A.max_pairs)
R["g_sweep"] = {}
for scale in (0.5, 1.0, 2.0, 4.0):
    r = recall(model, p_, t_, w_, ii, jj, scale)
    R["g_sweep"][f"x{scale}"] = r
    print(f"  gp/gd x{scale:<4}  union own {r['union_own']:.3f}   "
          f"union own+nbr {r['union_neighbour']:.3f}", flush=True)
print("\n  READ: union own >0.95 means the boundary findings (REVIEW_FIELD §3)")
print("  are not where the science is; the gap between own and own+neighbour is")
print("  what Reformer's 'and one chunk back' would buy for free.")

emit("i2_coblock", R)
