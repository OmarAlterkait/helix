"""Resolution at a FIXED physical scale, independent of patch size.

The along-wire probe (``helix.probe.alongwire``) scores features per PATCH, so a
bigger patch is scored on coarser targets and can lose sub-patch detail without
the score moving. This module defines truth and metrics on a fixed fine grid --
2 wires x 16 raw ticks, finer than any patch under test -- so models with
different tokenizers are scored on the same cells.

Truth (per event, from the simulation's ``hits`` + ``step``; no tokenizer, no
checkpoint):

  map rows   fine cells to train/score a charge-map probe on: every cell with
             truth charge (capped), cells near signal, cells at random kept
             coefficients (noise-dominated)
  windows    32 wires x 128 ticks (16 x 8 fine cells), all cells kept:
               iso   one isolated, compact particle (>=90% of its plane charge
                     inside, <5% of the window from anything else)
               pair  two such particles, 4-24 wires or 32-96 ticks apart
               bg    kept coefficients but no truth charge within +-16 w / +-128 t
             Windows are placed at a RANDOM offset around the particle(s). A
             window centred on truth rewards a blurred prediction: its centroid
             lands on the window centre, i.e. on the answer.

Metrics (``scalars``) from a predicted fine-cell charge map:

  map_r            Pearson r per (event, plane), Fisher-z mean
  presence_auc     any truth charge vs none, cell level
  faint_auc        faint cells (0 < q < 30th pct of nonzero) vs empty
  floor_eff1pct_*  isolated-particle windows above the 99th percentile of
                   background windows (1% false-positive rate), by energy;
                   floor_eff5pct_* and floor_auc_* as diagnostics. A cell no
                   token covers must be predicted 0 by the caller: the probe never
                   saw an all-zero input, and its extrapolated constant otherwise
                   sets every window's maximum (scripts/eval_resolution.py does).
  loc_w_* / loc_t_*  |centroid(prediction) - centroid(truth)|, wires / ticks
  sep_dip_*        close pairs: min along the segment / lower peak (0 = two
                   separate peaks, >=1 = one blob), median by separation, also
                   split by whether a TOKEN EDGE lies between the two (a probe's
                   prediction can step at a token boundary)

Measured noise between two training seeds of one recipe (d768, 29.2k steps):
map/presence/faint AUC within 0.003, floor efficiency within 0.02, energetic
localization within 0.01 wire. Floor AUC (0.04-0.05) and faint-deposit
localization (0.5 wire) were too noisy and are not reported.

numpy only: importable without torch or pimm (tests/test_boundary.py).
"""

from __future__ import annotations

import numpy as np

FW, FT = 2, 16                       # fine cell, wires x raw ticks
WW, WT = 32, 128                     # window
_TOFF = 4096                         # fine-tick offset inside the packed key
ENERGY_BINS = (0.0, 0.1, 0.2, 0.5, 1.0, 1e9)            # MeV
SEP_BINS = {"w": (4, 8, 16, 24), "t": (32, 64, 96)}       # wires / ticks

__all__ = ["FW", "FT", "WW", "WT", "ENERGY_BINS", "SEP_BINS", "fkey", "unkey",
           "window_cells", "event_truth", "auc", "window_stats", "scalars"]


def fkey(g, fw, ft):
    """Pack (plane_gid, fine wire, fine tick) into one int64. Coordinates must
    be >= 0 (wire) / > -4096 (tick): a negative wire sets the plane bits."""
    return (np.asarray(g, np.int64) << 40) | (np.asarray(fw, np.int64) << 20) \
        | (np.asarray(ft, np.int64) + _TOFF)


def unkey(k):
    k = np.asarray(k, np.int64)
    return k >> 40, (k >> 20) & 0xFFFFF, (k & 0xFFFFF) - _TOFF


def window_cells(g, w0, t0):
    """The 16 x 8 fine-cell keys of the window at pixel origin (w0, t0)."""
    fw = w0 // FW + np.arange(WW // FW)
    ft = t0 // FT + np.arange(WT // FT)
    W, T = np.meshgrid(fw, ft, indexing="ij")
    return fkey(np.full(W.size, g), W.ravel(), T.ravel())


def _fourier(x):
    ang = 2 * np.pi * x[..., None] * np.array([1, 2, 3], np.float32)
    return np.concatenate([np.sin(ang), np.cos(ang)], -1).reshape(x.shape[0], -1)


def cell_neighbors(keys, cell_key, band_lengths, cfg, radius=1):
    """Per fine cell and band: the tokens of the (2r+1) x (2r+1) patches around
    the one covering it, -> (n, n_bands, (2r+1)**2) int64, -1 where no token.

    Found by shifting the covering PATCH's block coordinates, so a cell with no
    token of its own in a band still sees the tokens beside it -- the empty cells
    next to activity are exactly those. Slot ``(2r+1)**2 // 2`` is the covering
    token, the same index :func:`cell_inputs` returns. Order: wire offset major,
    tick offset minor, each from -r to r.
    """
    from helix.model.tokenize import cell_key as pack, pixel_cells, unpack_cell_key

    g, fw, ft = unkey(np.asarray(keys, np.int64))
    pc = pixel_cells(g, fw * FW, ft * FT + FT // 2, band_lengths, cfg)          # (n, nb)
    pg, pb, wb, tb = unpack_cell_key(pc.ravel())
    ck = np.asarray(cell_key, np.int64)
    order = np.argsort(ck); cks = ck[order]
    offs = [(dw, dt) for dw in range(-radius, radius + 1) for dt in range(-radius, radius + 1)]
    out = np.full((pc.size, len(offs)), -1, np.int64)
    for j, (dw, dt) in enumerate(offs):
        w2, t2 = wb + dw, tb + dt
        ok = (w2 >= 0) & (t2 >= 0)
        if not ok.any() or not len(cks):
            continue
        k2 = pack(pg[ok], pb[ok], w2[ok] * cfg.pw, t2[ok] * cfg.pt, cfg)
        pos = np.clip(np.searchsorted(cks, k2), 0, len(cks) - 1)
        out[np.nonzero(ok)[0], j] = np.where(cks[pos] == k2, order[pos], -1)
    return out.reshape(pc.shape[0], pc.shape[1], len(offs))


def cell_inputs(keys, cell_key, band_lengths, cfg):
    """Per fine cell: which token covers it in each band, and where inside it.

    ``keys``: fine-cell keys (:func:`fkey`). ``cell_key``: the tokenized event's
    per-token cell keys (``B["cell_key"]``). Returns ``idx`` (n, n_bands) int64 --
    the covering token's row, -1 if none -- and ``aux`` (n, 13 * n_bands) float32:
    a presence bit per band, then the cell's continuous offset inside each band's
    token (wire, tau), Fourier-encoded. The frozen probe (scripts/eval_resolution.py)
    and the trainable denoising head (helix.model.denoise) both read exactly this,
    so a frozen and a fine-tuned number differ only in what was trained.
    """
    from helix.model.tokenize import pixel_cells

    nb, pw, pt = cfg.n_bands, cfg.pw, cfg.pt
    g, fw, ft = unkey(np.asarray(keys, np.int64))
    w, t = fw * FW, ft * FT + FT // 2
    pc = pixel_cells(g, w, t, band_lengths, cfg)
    ck = np.asarray(cell_key, np.int64)
    order = np.argsort(ck); cks = ck[order]
    if len(cks):
        pos = np.clip(np.searchsorted(cks, pc), 0, len(cks) - 1)
        idx = np.where(cks[pos] == pc, order[pos], -1)
    else:
        idx = np.full(pc.shape, -1, np.int64)
    dec = (1 << np.asarray(cfg.lev)).astype(np.float64)
    toff = np.asarray(cfg.toff)[g % 3]
    offs = []
    for b in range(nb):
        tau = (t + toff) / dec[b] - cfg.delta[b]
        offs += [(w % pw + 0.5) / pw, (tau / pt) % 1.0]
    aux = np.concatenate([(idx >= 0).astype(np.float32), _fourier(np.stack(offs, 1).astype(np.float32))], 1)
    return idx.astype(np.int64), aux.astype(np.float32)


def event_truth(pix, coeff_cells, rng, cap_signal=3000, n_iso=60, n_pair=40, n_bg=60):
    """Truth for one event.

    ``pix``: dict of per-pixel arrays ``g`` (plane_gid), ``w``, ``t``, ``q`` and
    ``trk`` (a global particle id), plus ``E``: {trk: deposited MeV}.
    ``coeff_cells``: fine-cell keys of the event's kept coefficients (any band).
    Returns the arrays a stage-1 evaluation needs: ``mrows``/``mq`` (map rows and
    their truth charge), ``wkey``/``wq``/``wid`` (window cells, charge, window
    index) and ``meta`` (one dict per window).
    """
    G, W, T, Q, TRK = (np.asarray(pix[k]) for k in ("g", "w", "t", "q", "trk"))
    E = pix["E"]
    uk, inv = np.unique(fkey(G, W // FW, T // FT), return_inverse=True)
    fq = np.bincount(inv, weights=Q)
    qmap = dict(zip(uk.tolist(), fq.tolist()))
    ck = np.unique(np.asarray(coeff_cells, np.int64))

    # map rows: signal, near signal, random kept coefficients
    sig = uk[fq > 0]
    if len(sig) > cap_signal:
        sig = rng.choice(sig, cap_signal, replace=False)
    sg, sw, st = unkey(sig)
    near = fkey(sg, np.maximum(sw + rng.integers(-8, 9, len(sig)), 0), st + rng.integers(-8, 9, len(sig)))
    near = near[np.isin(near, ck)]
    coeff = rng.choice(ck, min(len(sig) // 2 + 1, len(ck)), replace=False) if len(ck) else ck
    mrows = np.unique(np.concatenate([sig, near, coeff]))
    mq = np.array([qmap.get(x, 0.0) for x in mrows.tolist()])

    # per (plane, particle): charge, charge centroid
    pu, pinv = np.unique((G.astype(np.int64) << 40) | TRK, return_inverse=True)
    pq = np.bincount(pinv, weights=Q)
    pcw = np.bincount(pinv, weights=Q * W) / np.maximum(pq, 1e-12)
    pct = np.bincount(pinv, weights=Q * T) / np.maximum(pq, 1e-12)

    # window queries are searchsorted slices returning pixel INDICES; an
    # event-sized mask per query saturates memory bandwidth across workers
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
        if sum(w[0] == "iso" for w in wins) >= n_iso:
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
        if len(js) < 2 or n_p >= n_pair:
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
            if n_p >= n_pair:
                break

    n_b = 0
    for x in (rng.permutation(ck)[:3000].tolist() if len(ck) else []):
        if n_b >= n_bg:
            break
        g, fw, ft = (int(v) for v in unkey(x))
        w0, t0 = max(0, fw * FW - WW // 2), max(0, ft * FT - WT // 2)
        if len(contents(g, w0, t0, 16, 128)):
            continue
        wins.append(("bg", g, w0, t0, [], []))
        n_b += 1

    wk = [window_cells(g, w0, t0) for _, g, w0, t0, _, _ in wins]
    wq = [np.array([qmap.get(x, 0.0) for x in k.tolist()]) for k in wk]
    for arr in wk + [mrows]:
        g_, fw_, _ = unkey(arr)
        if len(arr) and ((g_ < 0).any() or (fw_ < 0).any()):
            raise ValueError("negative cell coordinate in a key")
    meta = [dict(kind=k, g=g, w0=w0, t0=t0, E=e, cen=c) for k, g, w0, t0, e, c in wins]
    return dict(mrows=mrows, mq=mq,
                wkey=np.concatenate(wk) if wk else np.zeros(0, np.int64),
                wq=np.concatenate(wq) if wq else np.zeros(0),
                wid=np.concatenate([np.full(len(k), n) for n, k in enumerate(wk)]) if wk else np.zeros(0, int),
                meta=meta)


def near_windows(keys, q, rng, n=60, margin=1, pad=(8, 8)):
    """Charge-free windows NEAR activity, as pixel origins ``[(g, w0, t0)]``.

    No charge cell within ``margin`` cells of the window, but at least one within
    ``pad`` (wire, tick) cells -- 16 wires / 128 ticks at the default, exactly the
    surroundings the floor's own noise windows must keep EMPTY. Those measure the
    false-positive rate far from activity only, while isolated deposits sit near
    it; a model that leaves a haze around activity is credited for it unless the
    threshold is also set here. ``keys``/``q``: the event's fine cells and charge.
    Up to ``n`` windows, shared equally over the planes that carry charge.
    """
    keys = np.asarray(keys, np.int64)[np.asarray(q) > 0]
    if not len(keys):
        return []
    G, CW, CT = unkey(keys)
    ww, wt = WW // FW, WT // FT
    planes = np.unique(G)
    per = -(-n // len(planes))
    out = []
    for g in planes.tolist():
        m = G == g
        o = max(pad) + margin + 1                                 # grid offset: origins may sit left of 0
        occ = np.zeros((int(CW[m].max()) + ww + 2 * o + 1, int(CT[m].max()) + wt + 2 * o + 1), np.int32)
        occ[CW[m] + o, CT[m] + o] = 1
        S = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

        def box(i0, i1, j0, j1):                                  # charge cells in [i0,i1) x [j0,j1), grid coords
            i0, i1 = np.clip(i0, 0, occ.shape[0]), np.clip(i1, 0, occ.shape[0])
            j0, j1 = np.clip(j0, 0, occ.shape[1]), np.clip(j1, 0, occ.shape[1])
            return S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]

        I, J = np.meshgrid(np.arange(o, occ.shape[0] - ww), np.arange(o, occ.shape[1] - wt), indexing="ij")
        I, J = I.ravel(), J.ravel()
        empty = box(I - margin, I + ww + margin, J - margin, J + wt + margin) == 0
        near = box(I - pad[0], I + ww + pad[0], J - pad[1], J + wt + pad[1]) > 0
        cand = np.nonzero(empty & near)[0]
        for c in rng.permutation(cand)[:per].tolist():
            out.append((int(g), int(I[c] - o) * FW, int(J[c] - o) * FT))
    return out[:n] if len(out) > n else out


def near_scalars(W, snr_bins=((0, 1e-9), (1e-9, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 1e9))):
    """Floor quantities that need near-activity noise windows (kind ``bgn``) and,
    when present, per-window annotations on the isolated deposits (``snr``: the
    deposit's optimal matched-filter SNR, 0 if the clean sensor holds none of it;
    ``clean``: foreign charge <= 25% of its own; scripts/noise_vs_hits.py).

    ``near_fpr_at_far{1,5}pct``: the fraction of near-activity windows the
    far-from-activity threshold flags -- the haze; ``near_eff{1,5}pct_<bin>``: the
    efficiency per energy bin when the threshold is set near activity;
    ``snr_eff1pct_{far,near}_<lo-hi>``: efficiency by deposit SNR at either threshold.
    """
    bg = np.array([w["score"] for w in W if w["kind"] == "bg"])
    bn = np.array([w["score"] for w in W if w["kind"] == "bgn"])
    out = {}
    if not len(bg) or not len(bn):
        return out
    iso = [w for w in W if w["kind"] == "iso"]
    for pct in (1, 5):
        tf, tn = np.quantile(bg, 1 - pct / 100), np.quantile(bn, 1 - pct / 100)
        out[f"near_fpr_at_far{pct}pct"] = float(np.mean(bn > tf))
        for i in range(len(ENERGY_BINS) - 1):
            s = np.array([w["score"] for w in iso if w["eb"] == i])
            if len(s):
                out[f"near_eff{pct}pct_{ENERGY_BINS[i]:g}-{ENERGY_BINS[i + 1]:g}"] = float(np.mean(s > tn))
        if pct == 1 and any("snr" in w for w in iso):
            for lo, hi in snr_bins:
                s = np.array([w["score"] for w in iso if w.get("clean") and lo <= w.get("snr", -1) < hi])
                lab = "nosignal" if hi <= 1e-9 else f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}-inf"
                if len(s):
                    out[f"snr_eff1pct_far_{lab}"] = float(np.mean(s > tf))
                    out[f"snr_eff1pct_near_{lab}"] = float(np.mean(s > tn))
    return out


def auc(pos, neg):
    """Mann-Whitney AUC, ties broken by order (fine at these sample sizes)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.concatenate([pos, neg]).argsort().argsort() + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def window_stats(pred, truth, meta, starts, ends, pw, pt, delta0, toff):
    """Per-window quantities from a predicted charge map ``pred`` and the truth
    ``truth`` (both flat over window cells; window n is ``[starts[n], ends[n])``).

    ``pw``/``pt`` are the tokenizer's patch sizes, ``delta0`` the A4 band's tick
    offset and ``toff`` a {plane%3: offset} map, for the token-edge flag.
    """
    shape = (WW // FW, WT // FT)
    gi, gj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    out = []
    for n, mt in enumerate(meta):
        P = pred[starts[n]:ends[n]].reshape(shape)
        Qt = truth[starts[n]:ends[n]].reshape(shape)
        rec = dict(ev=mt.get("ev"), kind=mt["kind"], score=float(P.max()))
        if mt["kind"] == "iso":
            rec["eb"] = int(np.searchsorted(ENERGY_BINS, mt["E"][0], side="right") - 1)
            wgt = np.maximum(P - np.median(P), 0)
            if wgt.sum() > 0 and Qt.sum() > 0:
                cp = ((wgt * gi).sum() / wgt.sum(), (wgt * gj).sum() / wgt.sum())
                ct = ((Qt * gi).sum() / Qt.sum(), (Qt * gj).sum() / Qt.sum())
                rec["loc"] = (abs(cp[0] - ct[0]) * FW, abs(cp[1] - ct[1]) * FT)
        if mt["kind"] == "pair":
            (wa, ta), (wb, tb) = mt["cen"]
            ca, cb = (int(wa // FW), int(ta // FT)), (int(wb // FW), int(tb // FT))
            dw, dt = abs(ca[0] - cb[0]), abs(ca[1] - cb[1])
            nstep = max(dw, dt)
            if nstep >= 2:
                axis, dist = ("w", dw * FW) if dw * FW / 32 >= dt * FT / 128 else ("t", dt * FT)
                line = [(round(ca[0] + (cb[0] - ca[0]) * s / nstep), round(ca[1] + (cb[1] - ca[1]) * s / nstep))
                        for s in range(nstep + 1)]

                def dip(M):
                    c = lambda i, j: M[min(max(i, 0), shape[0] - 1), min(max(j, 0), shape[1] - 1)]
                    pa = max(c(ca[0] + u, ca[1] + v) for u in (-1, 0, 1) for v in (-1, 0, 1))
                    pb = max(c(cb[0] + u, cb[1] + v) for u in (-1, 0, 1) for v in (-1, 0, 1))
                    lo = min(pa, pb)
                    return 1.0 if lo <= 0 else min(1.5, min(c(i, j) for i, j in line[1:-1]) / lo)

                bins = SEP_BINS[axis]
                rec["sep"] = f"{axis}<={next((b for b in bins if dist <= b), bins[-1])}"
                rec["dip"], rec["dip_truth"] = dip(P), dip(Qt)
                if axis == "w":
                    rec["edge"] = int((mt["w0"] + wa) // pw) != int((mt["w0"] + wb) // pw)
                else:
                    off = toff[mt["g"] % 3]
                    tau = lambda t: (mt["t0"] + t + off) / 16 - delta0          # A4 band tick
                    rec["edge"] = int(tau(ta) // pt) != int(tau(tb) // pt)
        out.append(rec)
    return out


def scalars(p, q, keyz, W, q30):
    """Every reported number from map rows (prediction ``p``, truth charge
    ``q``), per-(event, plane) Fisher z's ``keyz`` and window stats ``W``."""
    r = {"map_r": float(np.tanh(np.mean(keyz))) if len(keyz) else float("nan"),
         "presence_auc": auc(p[q > 0], p[q == 0]),
         "faint_auc": auc(p[(q > 0) & (q < q30)], p[q == 0])}
    bg = np.array([w["score"] for w in W if w["kind"] == "bg"])
    thr = np.quantile(bg, 0.99) if len(bg) else np.inf
    thr5 = np.quantile(bg, 0.95) if len(bg) else np.inf
    names = [f"{ENERGY_BINS[i]:g}-{ENERGY_BINS[i + 1]:g}" for i in range(len(ENERGY_BINS) - 1)]
    for i, nm in enumerate(names):
        s = np.array([w["score"] for w in W if w["kind"] == "iso" and w["eb"] == i])
        r[f"floor_eff1pct_{nm}"] = float((s > thr).mean()) if len(s) else float("nan")
        r[f"floor_eff5pct_{nm}"] = float((s > thr5).mean()) if len(s) else float("nan")    # diagnostic
        r[f"floor_auc_{nm}"] = auc(s, bg)                                                   # diagnostic
        lw = [w["loc"] for w in W if w["kind"] == "iso" and w["eb"] == i and "loc" in w]
        r[f"loc_w_{nm}"] = float(np.median([x[0] for x in lw])) if lw else float("nan")
        r[f"loc_t_{nm}"] = float(np.median([x[1] for x in lw])) if lw else float("nan")
    for ax, bins in SEP_BINS.items():
        for b in bins:
            k = f"{ax}<={b}"
            d = np.array([w["dip"] for w in W if w.get("sep") == k])
            r[f"sep_dip_{k}"] = float(np.median(d)) if len(d) else float("nan")
            for edge, nm in ((True, "edge"), (False, "same")):
                de = np.array([w["dip"] for w in W if w.get("sep") == k and w.get("edge") == edge])
                r[f"sep_dip_{nm}_{k}"] = float(np.median(de)) if len(de) >= 10 else float("nan")
    return r
