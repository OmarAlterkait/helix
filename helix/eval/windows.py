"""The evaluation windows and the detection-floor numbers, on the fixed fine grid.

Every model is scored on the same 2-wire x 16-tick cells (:mod:`helix.data.cells`) whatever its
patch size, so a bigger patch cannot lose sub-patch detail without the score moving. A cell no
token covers is scored as predicted 0, never dropped: which cells are scored never depends on the
tokenizer.

Truth (per event, from the cells record; no tokenizer, no checkpoint):

  map rows   fine cells to score a charge map on: every cell with truth charge (capped), cells
             near signal, cells at random kept coefficients (noise-dominated)
  windows    32 wires x 128 ticks (16 x 8 fine cells), all cells kept:
               iso   one isolated, compact particle (>=90% of its plane charge inside, <5% of
                     the window from anything else)
               pair  two such particles, 4-24 wires or 32-96 ticks apart
               bg    kept coefficients but no truth charge within +-16 w / +-128 t
             Windows are placed at a RANDOM offset around the particle(s). A window centred on
             truth rewards a blurred prediction: its centroid lands on the window centre, i.e.
             on the answer.

Numbers (:func:`score_cells`) from a predicted fine-cell charge map, each with a [16, 84]%
interval over an event bootstrap:

  map_r            Pearson r per (event, plane), Fisher-z mean
  presence_auc     any truth charge vs none, cell level
  faint_auc        faint cells (0 < q < 30th pct of nonzero) vs empty
  floor_eff1pct_*  isolated-particle windows above the 99th percentile of background windows
                   (1% false-positive rate), by energy
  loc_w_* / loc_t_*  |centroid(prediction) - centroid(truth)|, wires / ticks
  sep_dip_*        close pairs: min along the segment / lower peak (0 = two separate peaks,
                   >=1 = one blob), median by separation

AUCs count a tie as a half (a window or cell predicted exactly 0 is common). The bootstrap
resamples events; an event drawn k times weighs k in every number.

Measured noise between two training seeds of one recipe (d768, 29.2k steps): map/presence/faint
AUC within 0.003, floor efficiency within 0.02, energetic localization within 0.01 wire.

numpy only.
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import FT, FW, Q0, fkey, unkey

WW, WT = 32, 128                     # window, wires x raw ticks
ENERGY_BINS = (0.0, 0.1, 0.2, 0.5, 1.0, 1e9)            # MeV
SEP_BINS = {"w": (4, 8, 16, 24), "t": (32, 64, 96)}       # wires / ticks
#: per event: signal map rows at most, and windows per kind at most
CAP_SIGNAL, N_ISO, N_PAIR, N_BG = 3000, 60, 40, 60
KINDS = ("iso", "pair", "bg")
SEP_KEYS = tuple(f"{ax}<={b}" for ax, bins in SEP_BINS.items() for b in bins)
ENERGY_NAMES = tuple(f"{ENERGY_BINS[i]:g}-{ENERGY_BINS[i + 1]:g}" for i in range(len(ENERGY_BINS) - 1))
#: rows a group needs for its correlation to count (fisher_z), everywhere
MIN_ROWS = 21

__all__ = ["WW", "WT", "MIN_ROWS", "window_cells", "event_truth", "window_stats", "score_cells",
           "plane_group", "group_event", "fisher_z", "ranks", "auc", "bootstrap"]


def window_cells(g, w0, t0):
    """The 16 x 8 fine-cell keys of the window at pixel origin (w0, t0), wire-major."""
    fw = w0 // FW + np.arange(WW // FW)
    ft = t0 // FT + np.arange(WT // FT)
    W, T = np.meshgrid(fw, ft, indexing="ij")
    return fkey(np.full(W.size, g), W.ravel(), T.ravel())


def event_truth(cells, coeff_cells, rng, windows=True):
    """The map rows and windows of one event, from its cells record.

    ``cells``: a :mod:`helix.data.cells` record (tracks numbered by row, energies in
    ``trk_energy``). ``coeff_cells``: fine-cell keys of the event's kept coefficients (any band,
    :func:`helix.data.cells.coeff_cells`). Returns ``mrows``/``mq`` (map rows and their truth
    charge) and, with ``windows``, ``wkey``/``wq`` (window cells in window order, :func:`window_cells`'
    order inside each, and their charge) and ``meta`` (one dict per window). The map rows are
    drawn first, so they are the same with or without the windows.
    """
    E = dict(enumerate(np.asarray(cells["trk_energy"], np.float64).tolist()))
    uk = np.asarray(cells["key"], np.int64)
    o = np.argsort(uk)
    uk_s, fq_s = uk[o], np.asarray(cells["q"], np.float64)[o]
    fq = np.asarray(cells["q"], np.float64)
    ck = np.unique(np.asarray(coeff_cells, np.int64))

    def charge(keys):                                         # truth charge of fine cells, 0 if none
        if not len(uk_s):
            return np.zeros(len(keys))
        pos = np.clip(np.searchsorted(uk_s, keys), 0, len(uk_s) - 1)
        return np.where(uk_s[pos] == keys, fq_s[pos], 0.0)

    # One row per (cell, particle), placed at the cell's origin. Every box below is grid-aligned
    # (origins are multiples of FW / FT, paddings too), so a row is inside a box exactly when its
    # pixels are.
    G, cw_, ct_ = unkey(uk[np.asarray(cells["ct_cell"], np.int64)])
    W, T = cw_ * FW, ct_ * FT
    Q, TRK = np.asarray(cells["ct_q"], np.float64), np.asarray(cells["ct_track"], np.int64)

    # map rows: signal, near signal, random kept coefficients
    sig = uk[fq > 0]
    if len(sig) > CAP_SIGNAL:
        sig = rng.choice(sig, CAP_SIGNAL, replace=False)
    sg, sw, st = unkey(sig)
    near = fkey(sg, np.maximum(sw + rng.integers(-8, 9, len(sig)), 0), st + rng.integers(-8, 9, len(sig)))
    near = near[np.isin(near, ck)]
    coeff = rng.choice(ck, min(len(sig) // 2 + 1, len(ck)), replace=False) if len(ck) else ck
    mrows = np.unique(np.concatenate([sig, near, coeff]))
    mq = charge(mrows)
    if not windows:
        return dict(mrows=mrows, mq=mq)

    # per (plane, particle): charge, and its pixel charge centroid (stored: finer than a cell)
    pu, pinv = np.unique((G.astype(np.int64) << 40) | TRK, return_inverse=True)
    pinv = pinv.ravel()
    pq = np.bincount(pinv, weights=Q)
    pcw, pct = np.asarray(cells["pt_wire"], np.float64), np.asarray(cells["pt_tick"], np.float64)

    # window queries are searchsorted slices returning row INDICES; an event-sized mask per
    # query saturates memory bandwidth across workers
    o_gw = np.lexsort((W, G))
    GW = G[o_gw].astype(np.int64) * (1 << 24) + W[o_gw]

    def contents(g, w0, t0, pad_w=0, pad_t=0):
        lo = np.searchsorted(GW, g * (1 << 24) + (w0 - pad_w))
        hi = np.searchsorted(GW, g * (1 << 24) + (w0 + WW + pad_w))
        idx = o_gw[lo:hi]
        return idx[(T[idx] >= t0 - pad_t) & (T[idx] < t0 + WT + pad_t)]

    def origin(cw, ct):            # clamped at 0: a negative wire corrupts the packed key
        return max(0, int(round((cw - WW / 2) / FW)) * FW), max(0, int(round((ct - WT / 2) / FT)) * FT)

    wins = []
    good = np.nonzero(pq > 0)[0]
    rng.shuffle(good)
    for j in good[:4000]:
        if len(wins) >= N_ISO:                                 # only iso windows so far
            break
        g = int(pu[j] >> 40)
        w0, t0 = origin(pcw[j] + rng.uniform(-6, 6), pct[j] + rng.uniform(-32, 32))
        idx = contents(g, w0, t0)
        if not len(idx):
            continue
        mine = pinv[idx] == j
        qin, qall = Q[idx[mine]].sum(), Q[idx].sum()
        if qin < 0.9 * pq[j] or qall - qin > 0.05 * qall:
            continue
        wins.append(("iso", g, w0, t0, [E.get(int(TRK[idx[mine][0]]), 0.0)], [(pcw[j] - w0, pct[j] - t0)]))

    n_p = 0
    for g in np.unique(pu >> 40).tolist():
        js = np.nonzero(((pu >> 40) == g) & (pq > 0))[0]
        if len(js) < 2 or n_p >= N_PAIR:
            continue
        cwj, ctj = pcw[js], pct[js]
        for a in rng.permutation(len(js))[:400]:
            dw, dt = np.abs(cwj - cwj[a]), np.abs(ctj - ctj[a])
            cand = np.nonzero(((dw >= 4) & (dw <= 24) & (dt <= 96)) | ((dt >= 32) & (dt <= 96) & (dw <= 24)))[0]
            for b in cand[:3]:
                if b == a:
                    continue
                ja, jb = js[a], js[b]
                w0, t0 = origin((pcw[ja] + pcw[jb]) / 2 + rng.uniform(-3, 3),
                                (pct[ja] + pct[jb]) / 2 + rng.uniform(-16, 16))
                idx = contents(g, w0, t0)
                if not len(idx):
                    continue
                inA, inB = pinv[idx] == ja, pinv[idx] == jb
                qa, qb, qall = Q[idx[inA]].sum(), Q[idx[inB]].sum(), Q[idx].sum()
                if qa < 0.9 * pq[ja] or qb < 0.9 * pq[jb] or qall - qa - qb > 0.05 * qall:
                    continue
                wins.append(("pair", g, w0, t0,
                             [E.get(int(TRK[idx[inA][0]]), 0.0), E.get(int(TRK[idx[inB][0]]), 0.0)],
                             [(pcw[ja] - w0, pct[ja] - t0), (pcw[jb] - w0, pct[jb] - t0)]))
                n_p += 1
                break
            if n_p >= N_PAIR:
                break

    n_b = 0
    for x in (rng.permutation(ck)[:3000].tolist() if len(ck) else []):
        if n_b >= N_BG:
            break
        g, fw, ft = (int(v) for v in unkey(x))
        w0, t0 = max(0, fw * FW - WW // 2), max(0, ft * FT - WT // 2)
        if len(contents(g, w0, t0, 16, 128)):
            continue
        wins.append(("bg", g, w0, t0, [], []))
        n_b += 1

    wkey = np.concatenate([window_cells(g, w0, t0) for _, g, w0, t0, _, _ in wins]) if wins \
        else np.zeros(0, np.int64)
    meta = [dict(kind=k, g=g, w0=w0, t0=t0, E=e, cen=c) for k, g, w0, t0, e, c in wins]
    return dict(mrows=mrows, mq=mq, wkey=wkey, wq=charge(wkey), meta=meta)


def window_stats(score, truth, meta):
    """Per-window columns from a per-cell ``score`` and the truth charge ``truth``, both flat over
    the windows' cells in window order: ``ev``, ``kind`` (index into :data:`KINDS`), ``score``
    (the window's maximum), and, by kind, ``eb`` (energy bin) and ``loc_w``/``loc_t`` (iso; -1 /
    NaN otherwise), ``sep`` (index into :data:`SEP_KEYS`) and ``dip`` (pair)."""
    shape = (WW // FW, WT // FT)
    n = len(meta)
    P = np.asarray(score, np.float64).reshape((n,) + shape)
    Qt = np.asarray(truth, np.float64).reshape((n,) + shape)
    gi, gj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    out = dict(ev=np.array([m["ev"] for m in meta], np.int64),
               kind=np.array([KINDS.index(m["kind"]) for m in meta], np.int64),
               score=P.max((1, 2)), eb=np.full(n, -1), loc_w=np.full(n, np.nan), loc_t=np.full(n, np.nan),
               sep=np.full(n, -1), dip=np.full(n, np.nan))
    for i, mt in enumerate(meta):
        if mt["kind"] == "iso":
            out["eb"][i] = np.searchsorted(ENERGY_BINS, mt["E"][0], side="right") - 1
            wgt = np.maximum(P[i] - np.median(P[i]), 0)
            if wgt.sum() > 0 and Qt[i].sum() > 0:
                out["loc_w"][i] = abs((wgt * gi).sum() / wgt.sum() - (Qt[i] * gi).sum() / Qt[i].sum()) * FW
                out["loc_t"][i] = abs((wgt * gj).sum() / wgt.sum() - (Qt[i] * gj).sum() / Qt[i].sum()) * FT
        elif mt["kind"] == "pair":
            (wa, ta), (wb, tb) = mt["cen"]
            ca, cb = (int(wa // FW), int(ta // FT)), (int(wb // FW), int(tb // FT))
            dw, dt = abs(ca[0] - cb[0]), abs(ca[1] - cb[1])
            nstep = max(dw, dt)
            if nstep < 2:
                continue
            axis, dist = ("w", dw * FW) if dw * FW / 32 >= dt * FT / 128 else ("t", dt * FT)
            line = [(round(ca[0] + (cb[0] - ca[0]) * s / nstep), round(ca[1] + (cb[1] - ca[1]) * s / nstep))
                    for s in range(nstep + 1)]
            M = P[i]
            c = lambda a, b: M[min(max(a, 0), shape[0] - 1), min(max(b, 0), shape[1] - 1)]   # noqa: E731
            pa = max(c(ca[0] + u, ca[1] + v) for u in (-1, 0, 1) for v in (-1, 0, 1))
            pb = max(c(cb[0] + u, cb[1] + v) for u in (-1, 0, 1) for v in (-1, 0, 1))
            lo = min(pa, pb)
            bins = SEP_BINS[axis]
            out["sep"][i] = SEP_KEYS.index(f"{axis}<={next((b for b in bins if dist <= b), bins[-1])}")
            out["dip"][i] = 1.0 if lo <= 0 else min(1.5, min(c(a, b) for a, b in line[1:-1]) / lo)
    return out


def score_cells(p, y, e, g, wscore, wq, meta, n_boot=200, seed=0):
    """Every floor number of one per-cell prediction, each ``[point, lo, hi]``.

    Map rows: prediction ``p`` and truth ``y`` (both ``log1p(q / Q0)``) of events ``e``, plane
    gids ``g``. Windows: the per-cell window score ``wscore`` and truth charge ``wq``, flat in
    window order, and ``meta`` (one dict per window, its event as ``ev``).
    """
    Wc = window_stats(wscore, wq, meta)
    e = np.asarray(e, np.int64)
    evs = np.unique(np.concatenate([e, Wc["ev"]]))
    re, we = np.searchsorted(evs, e), np.searchsorted(evs, Wc["ev"])
    q = np.expm1(np.asarray(y, np.float64)) * Q0
    q30 = np.quantile(q[q > 0], 0.3) if (q > 0).any() else 0.0
    pos, neg, faint = (q > 0).astype(float), (q == 0).astype(float), ((q > 0) & (q < q30)).astype(float)
    r = ranks(p)
    z = fisher_z(p, y, plane_group(e, g))
    zv = np.array(list(z.values()), np.float64)
    ze = np.searchsorted(evs, group_event(np.array(list(z), np.int64)))
    iso, bg = Wc["kind"] == KINDS.index("iso"), Wc["kind"] == KINDS.index("bg")

    def stat(m):
        wr, ww, mz = m[re].astype(np.float64), m[we], m[ze]
        rep = lambda x, sel: np.repeat(x[sel], ww[sel])                                  # noqa: E731
        med = lambda v: float(np.median(v)) if len(v) else float("nan")                  # noqa: E731
        out = {"map_r": float(np.tanh((mz * zv).sum() / mz.sum())) if mz.sum() else float("nan"),
               "presence_auc": auc(r, wr * pos, wr * neg), "faint_auc": auc(r, wr * faint, wr * neg)}
        b = rep(Wc["score"], bg)
        thr = np.quantile(b, 0.99) if len(b) else np.inf
        for i, nm in enumerate(ENERGY_NAMES):
            sel = iso & (Wc["eb"] == i)
            s = rep(Wc["score"], sel)
            out[f"floor_eff1pct_{nm}"] = float((s > thr).mean()) if len(s) else float("nan")
            out[f"loc_w_{nm}"] = med(rep(Wc["loc_w"], sel & np.isfinite(Wc["loc_w"])))
            out[f"loc_t_{nm}"] = med(rep(Wc["loc_t"], sel & np.isfinite(Wc["loc_t"])))
        for j, k in enumerate(SEP_KEYS):
            out[f"sep_dip_{k}"] = med(rep(Wc["dip"], Wc["sep"] == j))
        return out

    return bootstrap(stat, len(evs), n_boot, seed)


def plane_group(e, g):
    """One id per (event, plane gid), for a within-group correlation; :func:`group_event`
    inverts it."""
    return np.asarray(e, np.int64) * 16 + np.asarray(g, np.int64)


def group_event(k):
    return np.asarray(k, np.int64) // 16


def fisher_z(p, y, group, min_rows=MIN_ROWS):
    """``{group: arctanh(Pearson r)}`` of prediction ``p`` against truth ``y`` within each group
    with at least ``min_rows`` rows and variance in both. One pass over the rows (per-group sums),
    not one per group."""
    keys, inv, n = np.unique(np.asarray(group), return_inverse=True, return_counts=True)
    inv, G = inv.ravel(), len(keys)
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    dp = p - (np.bincount(inv, p, G) / n)[inv]               # centred first: the raw sums cancel
    dy = y - (np.bincount(inv, y, G) / n)[inv]
    spp, syy, spy = (np.bincount(inv, w, G) for w in (dp * dp, dy * dy, dp * dy))
    ok = (n >= min_rows) & (spp > 0) & (syy > 0)
    r = np.clip(spy[ok] / np.sqrt(spp[ok] * syy[ok]), -0.999, 0.999)
    return dict(zip(keys[ok].tolist(), np.arctanh(r).tolist()))


def ranks(s):
    """Dense ranks of the scores ``s`` (equal scores share one), for :func:`auc`."""
    return np.unique(np.asarray(s, np.float64), return_inverse=True)[1].ravel()


def auc(r, wpos, wneg):
    """Mann-Whitney AUC over dense ranks ``r``, with each row's weight as a positive ``wpos`` and
    as a negative ``wneg``: P(positive > negative) + P(tie) / 2. NaN without both."""
    n = int(r.max()) + 1 if len(r) else 0
    P, N = np.bincount(r, wpos, n), np.bincount(r, wneg, n)
    den = P.sum() * N.sum()
    return float((P * (np.cumsum(N) - 0.5 * N)).sum() / den) if den > 0 else float("nan")


def bootstrap(stat, n_events, n_boot=200, seed=0):
    """``{key: [point, lo, hi]}``: ``stat(m) -> {key: value}`` of per-event weights ``m`` (the
    point: every event once), and its [16, 84]% interval over ``n_boot`` resamples of the
    events with replacement (``m``: how often each was drawn)."""
    point = stat(np.ones(n_events, np.int64))
    rng = np.random.default_rng(seed)
    boots = [stat(np.bincount(rng.integers(0, n_events, n_events), minlength=n_events)) for _ in range(n_boot)]
    out = {}
    for k, v in point.items():
        bv = np.array([b.get(k, np.nan) for b in boots], float)
        lo, hi = np.nanpercentile(bv, [16, 84]) if np.isfinite(bv).any() else (np.nan, np.nan)
        out[k] = [round(float(v), 4), round(float(lo), 4), round(float(hi), 4)]
    return out
