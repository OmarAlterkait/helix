"""The evaluation windows and the detection floor, on the fixed fine grid.

Every model is scored on the same 2-wire x 16-tick cells (:mod:`helix.data.cells`) whatever its
patch size, so a bigger patch cannot lose sub-patch detail without the score moving. A cell no
token covers is scored as predicted 0, never dropped: which cells are scored never depends on the
tokenizer.

Truth (per event, from the cells record; no tokenizer, no checkpoint):

  map rows   fine cells a probe is FITTED on: every cell with truth charge (capped), cells near
             signal, cells at random kept coefficients (noise-dominated). Not scored.
  windows    32 wires x 128 ticks (16 x 8 fine cells), all cells kept:
               iso   one isolated, compact particle (>=90% of its plane charge inside, <5% of
                     the window from anything else), placed at a RANDOM offset around it (a
                     window centred on truth would put a blurred prediction's centre on the
                     answer)
               bg    kept coefficients but no truth charge within +-16 w / +-128 t; up to
                     :data:`N_BG_PLANE` per plane, so every view has its own background

The number (:func:`score_windows`): a window's score is its largest cell score; an iso window
is detected when it scores above the 99th percentile of the background windows OF ITS VIEW (U,
V or Y, both volumes) -- a 1% false-positive rate per view, since the views differ in noise
and one pooled threshold is set by the views with most background (measured: Y ran at 2.4-3.2%
under a pooled 1%). Reported as the detected fraction by the particle's deposited energy (its
whole track's step energy; it tracks the charge in the window, Spearman 0.94-0.96 on every
view at ~20k e/MeV), with a [16, 84]% interval over an event bootstrap.

Also here: the within-group R^2 the along-wire probe reports (:func:`group_r2`).

numpy only.
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import FT, FW, fkey, unkey

WW, WT = 32, 128                     # window, wires x raw ticks
ENERGY_BINS = (0.0, 0.1, 0.2, 0.5, 1e9)                  # MeV
#: per event: signal map rows at most, iso windows at most, bg windows per plane at most
CAP_SIGNAL, N_ISO, N_BG_PLANE = 3000, 60, 10
KINDS = ("iso", "bg")
VIEWS = ("U", "V", "Y")
ENERGY_NAMES = tuple(f"{ENERGY_BINS[i]:g}-{ENERGY_BINS[i + 1]:g}" for i in range(len(ENERGY_BINS) - 1))

__all__ = ["WW", "WT", "window_cells", "event_truth", "score_windows", "plane_group", "group_event",
           "group_r2", "bootstrap"]


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
    charge) and, with ``windows``, ``wkey`` (window cells in window order, :func:`window_cells`'
    order inside each) and ``meta`` (one dict per window: ``kind``, ``g``, ``w0``, ``t0``, ``E``
    -- the iso particle's energy, NaN for bg). The map rows are drawn first, so they are the
    same with or without the windows.
    """
    E = dict(enumerate(np.asarray(cells["trk_energy"], np.float64).tolist()))
    uk = np.asarray(cells["key"], np.int64)
    fq = np.asarray(cells["q"], np.float64)
    ck = np.unique(np.asarray(coeff_cells, np.int64))

    # map rows: signal, near signal, random kept coefficients
    sig = uk[fq > 0]
    if len(sig) > CAP_SIGNAL:
        sig = rng.choice(sig, CAP_SIGNAL, replace=False)
    sg, sw, st = unkey(sig)
    near = fkey(sg, np.maximum(sw + rng.integers(-8, 9, len(sig)), 0), st + rng.integers(-8, 9, len(sig)))
    near = near[np.isin(near, ck)]
    coeff = rng.choice(ck, min(len(sig) // 2 + 1, len(ck)), replace=False) if len(ck) else ck
    mrows = np.unique(np.concatenate([sig, near, coeff]))
    o = np.argsort(uk)
    uk_s, fq_s = uk[o], fq[o]
    if len(uk_s):
        pos = np.clip(np.searchsorted(uk_s, mrows), 0, len(uk_s) - 1)
        mq = np.where(uk_s[pos] == mrows, fq_s[pos], 0.0)
    else:
        mq = np.zeros(len(mrows))
    if not windows:
        return dict(mrows=mrows, mq=mq)

    # One row per (cell, particle), placed at the cell's origin. Every box below is grid-aligned
    # (origins are multiples of FW / FT, paddings too), so a row is inside a box exactly when its
    # pixels are.
    G, cw_, ct_ = unkey(uk[np.asarray(cells["ct_cell"], np.int64)])
    W, T = cw_ * FW, ct_ * FT
    Q, TRK = np.asarray(cells["ct_q"], np.float64), np.asarray(cells["ct_track"], np.int64)

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
        if len(wins) >= N_ISO:
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
        wins.append(("iso", g, w0, t0, E.get(int(TRK[idx[mine][0]]), 0.0)))

    if len(ck):
        cg = unkey(ck)[0]
        for g in np.unique(cg).tolist():
            n_b = 0
            for x in rng.permutation(ck[cg == g])[:500].tolist():
                if n_b >= N_BG_PLANE:
                    break
                _, fw, ft = (int(v) for v in unkey(x))
                w0, t0 = max(0, fw * FW - WW // 2), max(0, ft * FT - WT // 2)
                if len(contents(g, w0, t0, 16, 128)):
                    continue
                wins.append(("bg", g, w0, t0, np.nan))
                n_b += 1

    wkey = np.concatenate([window_cells(g, w0, t0) for _, g, w0, t0, _ in wins]) if wins \
        else np.zeros(0, np.int64)
    meta = [dict(kind=k, g=g, w0=w0, t0=t0, E=e) for k, g, w0, t0, e in wins]
    return dict(mrows=mrows, mq=mq, wkey=wkey, meta=meta)


def score_windows(score, meta, n_boot=200, seed=0):
    """The floor of one per-cell ``score`` (flat over the windows' cells in window order; larger
    = more charge): ``floor_eff1pct_<E bin>`` (each ``[point, lo, hi]``) and the window counts.
    ``meta``: one dict per window, its event as ``ev``."""
    n = len(meta)
    s = np.asarray(score, np.float64).reshape(n, -1).max(1) if n else np.zeros(0)
    ev = np.array([m["ev"] for m in meta], np.int64)
    kind = np.array([KINDS.index(m["kind"]) for m in meta], np.int64)
    view = np.array([m["g"] % 3 for m in meta], np.int64)
    eb = np.array([np.searchsorted(ENERGY_BINS, m["E"], side="right") - 1 if m["kind"] == "iso" else -1
                   for m in meta], np.int64)
    evs, we = np.unique(ev, return_inverse=True)
    iso, bg = kind == 0, kind == 1

    def stat(m):
        w = m[we]
        thr = np.full(3, np.inf)
        for v in range(3):
            b = np.repeat(s[bg & (view == v)], w[bg & (view == v)])
            if len(b):
                thr[v] = np.quantile(b, 0.99)
        det = s > thr[view]
        out = {}
        for i, nm in enumerate(ENERGY_NAMES):
            sel = iso & (eb == i)
            out[f"floor_eff1pct_{nm}"] = float((w * det)[sel].sum() / w[sel].sum()) if w[sel].sum() else float("nan")
        return out

    out = bootstrap(stat, len(evs), n_boot, seed)
    out["n_windows"] = {f"{k}_{VIEWS[v]}": int(((kind == i) & (view == v)).sum())
                        for i, k in enumerate(KINDS) for v in range(3)}
    return out


def plane_group(e, g):
    """One id per (event, plane gid); :func:`group_event` inverts it."""
    return np.asarray(e, np.int64) * 16 + np.asarray(g, np.int64)


def group_event(k):
    return np.asarray(k, np.int64) // 16


def group_r2(p, y, group):
    """Per group: ``(keys, sse, sst)`` of prediction ``p`` against truth ``y`` once both are
    centred within the group, so a per-group offset costs nothing. The pooled within-group
    ``R^2 = 1 - sum(sse) / sum(sst)``; unlike a correlation it also charges a prediction of the
    wrong scale."""
    keys, inv = np.unique(np.asarray(group), return_inverse=True)
    inv, G = inv.ravel(), len(keys)
    n = np.bincount(inv, minlength=G)
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    dp = p - (np.bincount(inv, p, G) / n)[inv]
    dy = y - (np.bincount(inv, y, G) / n)[inv]
    return keys, np.bincount(inv, (dy - dp) ** 2, G), np.bincount(inv, dy * dy, G)


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
