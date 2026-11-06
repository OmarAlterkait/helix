#!/usr/bin/env python3
"""First-principles A: the denoiser's case-control sampling, under the hits and the complete truth.

    scripts/helix_run.sh python baselines/uboone_sp/causes/firstp_sampling.py \
        --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/firstp/sampling --workers 32

The training rows of every denoiser checkpoint (M6d, M6dc, lb_*) come from
``helix.data.denoise.DenoiseEvents.rows`` with ONE recipe (meta['args'], identical
across them; firstp_args.py): sig_cap 20000, near 1, neg 1, cov 2, win 48 windows,
near_any 1, win_near 24 windows. The rows are drawn USING the truth, so the model
learns P(charged | x) under the training prior r_train, not the deployment prior
r_deploy (all token-covered cells). Per event we

  * draw the rows exactly as training does (asserted equal to DenoiseEvents.rows
    under the same seed), once with the HITS truth as support and once with the
    COMPLETE truth (denoise_truth_complete), ``--draws`` draws each, keep only
    token-covered rows (the trainer drops the rest);
  * enumerate ALL token-covered fine cells (the deployment population);
  * stratify every cell by density = true charge within +-8 fine cells, the cell
    excluded (dense/analyze.py), computed with the hits truth ('hd') and with the
    complete truth ('cd'); and by plane;
  * count charged (q > 0 -- the presence target is y > 0) rows and cells.

Writes per event ``ev_<run>_<tag>_<pos>.npz`` (count tensors); firstp_sampling_sum.py
reduces them into causes/results/firstp_sampling.json/.md.
"""
import argparse
import zlib
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
HITS = "/global/cfs/cdirs/m5238/users/oalter/ost61/denoise_truth"
COMPLETE = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
RUNS = ["run_0027575715", "run_0027587651", "run_0027651463", "run_0027654870",
        "run_0027663748", "run_0027668746", "run_0027670361", "run_0027719646"]
RECIPE = dict(sig_cap=20000, neg_per_sig=1.0, near_per_sig=1.0, cov_per_sig=2.0, win_per_event=48,
              near_any_per_sig=1.0, win_near_per_event=24)
PARTS = ["sig", "near", "neg", "cov", "win", "near_any", "win_near"]
DB = np.array([0.5, 1e4, 1e5, 1e6, 4e6])                 # density edges (e-): iso | <=10k | ... | >4M
DLAB = ["iso", "<=10k", "10k-100k", "100k-1M", "1M-4M", ">4M"]
QE = np.array([0, 1e-3, 1, 5, 25, 102, 500, 2043.26, 8e3, 16e3, 64e3, 1e12])    # cell charge bins (e-)
QLAB = ["<1e-3", "1e-3-1", "1-5", "5-25", "25-102", "102-500", "500-Q0", "Q0-8k", "8-16k", "16-64k", ">64k"]
SUP = ["hits", "complete"]
R = 8                                                     # density half-width, fine cells


def dbin(d):
    return np.searchsorted(DB, d, side="right")


class Grid:
    """Per-plane dense fine-cell grids of one truth: charge, +-8 box sums, distance to charge."""

    def __init__(self, keys, q, gids, nfw, t_lo, t_hi):
        from helix.probe.resolution import unkey
        g, w, t = unkey(keys)
        self.t_lo = min(t_lo, int(t.min()) if len(t) else t_lo)
        self.t_hi = max(t_hi, int(t.max()) + 1 if len(t) else t_hi)
        self.nw = {int(gg): max(int(nfw[i]), int(w[g == gg].max()) + 1 if (g == gg).any() else 0) for i, gg in enumerate(gids)}
        self.Q, self.S, self.C = {}, {}, {}
        for gg, nw in self.nw.items():
            m = g == gg
            Q = np.zeros((nw, self.t_hi - self.t_lo))
            np.add.at(Q, (w[m], t[m] - self.t_lo), q[m])
            self.Q[gg] = Q
            self.S[gg] = np.pad(Q.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            self.C[gg] = np.pad((Q > 0).astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))   # exact counts
        self._dist = {}

    def box(self, g, w, t, r):
        """Sum of charge in [w-r, w+r] x [t-r, t+r] (fine cells) per cell (vectorised, one plane)."""
        S = self.S[g]; nw, nt = S.shape[0] - 1, S.shape[1] - 1
        tt = t - self.t_lo
        i0, i1 = np.clip(w - r, 0, nw), np.clip(w + r + 1, 0, nw)
        j0, j1 = np.clip(tt - r, 0, nt), np.clip(tt + r + 1, 0, nt)
        return S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]

    def q_at(self, g, w, t):
        Q = self.Q[g]; tt = t - self.t_lo
        ok = (w >= 0) & (w < Q.shape[0]) & (tt >= 0) & (tt < Q.shape[1])
        out = np.zeros(len(w)); out[ok] = Q[w[ok], tt[ok]]
        return out

    def density(self, g, w, t):
        return self.box(g, w, t, R) - self.q_at(g, w, t)

    def dist(self, g, thr):
        """Chessboard distance (fine cells) of every grid cell to the nearest cell with q > thr."""
        if (g, thr) not in self._dist:
            from scipy.ndimage import distance_transform_cdt
            occ = self.Q[g] > thr
            self._dist[(g, thr)] = (distance_transform_cdt(~occ, metric="chessboard").astype(np.float32)
                                    if occ.any() else np.full(occ.shape, 9999, np.float32))
        return self._dist[(g, thr)]

    def win_dist(self, g, w0, t0, nw, nt, thr):
        """Min distance over the fine-cell box [w0, w0+nw) x [t0, t0+nt) to charge > thr."""
        D = self.dist(g, thr); a = t0 - self.t_lo
        sub = D[max(w0, 0):max(w0 + nw, 0), max(a, 0):max(a + nt, 0)]
        return float(sub.min()) if sub.size else np.nan


def rows_with_parts(ds, ce, tkeys, tq, rng):
    """DenoiseEvents.rows, recording which rule produced each row (same rng consumption)."""
    from helix.model.tokenize import tick_of_tau
    from helix.probe.resolution import FT, FW, fkey, unkey, near_windows, window_cells
    k = ce.band < ds.cfg.n_bands
    ct = tick_of_tau(ce.tau[k], ce.plane_gid[k], ce.band[k], ds.cfg)
    ck = np.unique(fkey(ce.plane_gid[k], ce.wire[k] // FW, np.floor(ct / FT).astype(np.int64)))
    sig = tkeys[tq > 0]
    n_sig_pre = len(sig)
    if len(sig) > ds.sig_cap:
        sig = rng.choice(sig, ds.sig_cap, replace=False)
    parts = {"sig": sig}
    if len(sig) and ds.near_per_sig > 0:
        n = int(len(sig) * ds.near_per_sig)
        base = rng.choice(sig, n, replace=True)
        g, w, t = unkey(base)
        near = fkey(g, np.maximum(w + rng.integers(-8, 9, n), 0), np.maximum(t + rng.integers(-8, 9, n), 0))
        parts["near"] = near[np.isin(near, ck)]
    if len(ck):
        parts["neg"] = rng.choice(ck, min(int(len(sig) * ds.neg_per_sig) + 1, len(ck)), replace=False)
    if ds.cov_per_sig > 0 and k.any():
        n = int(len(sig) * ds.cov_per_sig) + 1
        j = rng.integers(0, int(k.sum()), n)
        b, g = ce.band[k][j], ce.plane_gid[k][j]
        wire = (ce.wire[k][j] // ds.cfg.pw) * ds.cfg.pw + rng.integers(0, ds.cfg.pw, n)
        tau = (ce.tau[k][j] // ds.cfg.pt) * ds.cfg.pt + rng.integers(0, ds.cfg.pt, n)
        dec = (1 << np.asarray(ds.cfg.lev))[b]
        tick = tick_of_tau(tau, g, b, ds.cfg) + rng.random(n) * dec
        parts["cov"] = fkey(g, np.maximum(wire, 0) // FW, np.floor(np.maximum(tick, 0) / FT).astype(np.int64))
    wins, wins_near = [], []
    if ds.win_per_event > 0 and len(ck):
        x = ds._noise_windows(ck, tkeys, rng)
        parts["win"] = x
        if len(x):
            g, w, t = unkey(x.reshape(-1, 128)[:, 0])           # window_cells order: first cell = origin
            wins = list(zip(g.tolist(), w.tolist(), t.tolist()))
    if len(sig) and ds.near_any_per_sig > 0:
        n = int(len(sig) * ds.near_any_per_sig)
        g, w, t = unkey(rng.choice(sig, n, replace=True))
        parts["near_any"] = fkey(g, np.maximum(w + rng.integers(-8, 9, n), 0), np.maximum(t + rng.integers(-8, 9, n), 0))
    if ds.win_near_per_event > 0:
        nw = near_windows(tkeys, tq, rng, n=ds.win_near_per_event)
        parts["win_near"] = np.concatenate([window_cells(g, w0, t0) for g, w0, t0 in nw]) if nw else np.zeros(0, np.int64)
        wins_near = [(g, w0 // FW, t0 // FT) for g, w0, t0 in nw]
    rows = np.unique(np.concatenate(list(parts.values())))
    mask = np.zeros(len(rows), np.int64)
    for i, nm in enumerate(PARTS):
        if nm in parts and len(parts[nm]):
            mask[np.isin(rows, parts[nm])] |= 1 << i
    return rows, mask, ck, n_sig_pre, wins, wins_near


def win_candidates(ck, grid):
    """All distinct _noise_windows that a kept coefficient could centre (the whole candidate set)."""
    from helix.probe.resolution import FT, FW, WT, WW, unkey
    g, fw, ft = unkey(ck)
    w0, t0 = np.maximum(0, fw - WW // FW // 2), np.maximum(0, ft - WT // FT // 2)
    pw_, pt_ = 16 // FW, 128 // FT
    ok = np.zeros(len(ck), bool)
    for gg in np.unique(g).tolist():
        m = g == gg
        if gg not in grid.C:
            ok[m] = True; continue
        S = grid.C[gg]; nw, nt = S.shape[0] - 1, S.shape[1] - 1
        a = t0[m] - grid.t_lo
        i0, i1 = np.clip(w0[m] - pw_, 0, nw), np.clip(w0[m] + WW // FW + pw_, 0, nw)
        j0, j1 = np.clip(a - pt_, 0, nt), np.clip(a + WT // FT + pt_, 0, nt)
        occ = (S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0])
        ok[m] = occ == 0          # no charged cell (q > 0) in the padded box
    u = np.unique(np.stack([g[ok], w0[ok], t0[ok]], 1), axis=0) if ok.any() else np.zeros((0, 3), np.int64)
    return u, int(ok.sum())


def near_candidates(keys, q, margin=1, pad=(8, 8)):
    """Per plane: the number of candidate origins of helix.probe.resolution.near_windows."""
    from helix.probe.resolution import FT, FW, WT, WW, unkey
    keys = np.asarray(keys, np.int64)[np.asarray(q) > 0]
    out = {}
    if not len(keys):
        return out
    G, CW, CT = unkey(keys)
    ww, wt = WW // FW, WT // FT
    for g in np.unique(G).tolist():
        m = G == g
        o = max(pad) + margin + 1
        occ = np.zeros((int(CW[m].max()) + ww + 2 * o + 1, int(CT[m].max()) + wt + 2 * o + 1), np.int32)
        occ[CW[m] + o, CT[m] + o] = 1
        S = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

        def box(i0, i1, j0, j1):
            i0, i1 = np.clip(i0, 0, occ.shape[0]), np.clip(i1, 0, occ.shape[0])
            j0, j1 = np.clip(j0, 0, occ.shape[1]), np.clip(j1, 0, occ.shape[1])
            return S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]

        I, J = np.meshgrid(np.arange(o, occ.shape[0] - ww), np.arange(o, occ.shape[1] - wt), indexing="ij")
        I, J = I.ravel(), J.ravel()
        empty = box(I - margin, I + ww + margin, J - margin, J + wt + margin) == 0
        near = box(I - pad[0], I + ww + pad[0], J - pad[1], J + wt + pad[1]) > 0
        out[int(g)] = int((empty & near).sum())
    return out


def one_event(job):
    run, path, pos, event, out_dir, draws, seed = job
    import hdf5plugin  # noqa: F401
    from helix.data.denoise import DenoiseEvents
    from helix.probe.resolution import FT, FW, cell_inputs, fkey, unkey
    tag = os.path.basename(path).replace("sim_wire_coeff_", "").replace(".h5", "")
    out = os.path.join(out_dir, f"ev_{run}_{tag}_{pos:03d}.npz")
    if os.path.exists(out):
        return out, "exists"
    t00 = time.time()
    ds = DenoiseEvents("", [], "", items=[], **RECIPE)
    c = ds._shard_cfg(path)
    ce, B, bl = ds.tokens(path, pos)
    truths = {}
    for nm, root in (("hits", HITS), ("complete", COMPLETE)):
        z = np.load(os.path.join(root, run, f"truth_{tag}.npz"))
        a, b = int(z["offsets"][pos]), int(z["offsets"][pos + 1])
        assert int(z["event"][pos]) == event, (nm, path, pos)
        truths[nm] = (z["keys"][a:b].astype(np.int64), z["q"][a:b].astype(np.float64))
    gids = [int(x) for x in c["gids"]]
    nfw = [-(-int(n) // FW) for n in c["n_wires"]]
    t_lo, t_hi = -4, -(-int(bl[0]) * 16 // FT) + 4
    grids = {nm: Grid(*truths[nm], gids, nfw, t_lo, t_hi) for nm in SUP}
    nP, nD, nQ = 3, len(DLAB), len(QE) - 1

    # ---- deployment population: every token-covered fine cell
    U = []
    for i, g in enumerate(gids):
        W, T = np.meshgrid(np.arange(nfw[i]), np.arange(t_lo, t_hi), indexing="ij")
        U.append(fkey(np.full(W.size, g), W.ravel(), T.ravel()))
    U = np.concatenate(U)
    cov = np.zeros(len(U), bool)
    for s in range(0, len(U), 400000):
        idx, _ = cell_inputs(U[s:s + 400000], B["cell_key"], bl, ds.cfg)
        cov[s:s + 400000] = (idx >= 0).any(1)
    U = U[cov]
    ug, uw, ut = unkey(U)

    def strat(keys):
        """plane, q_hits, q_complete, hd bin, cd bin for fine-cell keys."""
        g, w, t = unkey(keys)
        P = np.asarray(g) % 3
        qh, qc, hd, cd = (np.zeros(len(keys)) for _ in range(4))
        for gg in np.unique(g).tolist():
            m = g == gg
            qh[m] = grids["hits"].q_at(gg, w[m], t[m]); qc[m] = grids["complete"].q_at(gg, w[m], t[m])
            hd[m] = grids["hits"].density(gg, w[m], t[m]); cd[m] = grids["complete"].density(gg, w[m], t[m])
        return P, qh, qc, dbin(hd), dbin(cd)

    uP, uqh, uqc, uhd, ucd = strat(U)
    res = {}
    # deploy[stratifier hd/cd][plane][dbin][charged-hits][charged-complete]
    for sname, sb in (("hd", uhd), ("cd", ucd)):
        H = np.zeros((nP, nD, 2, 2), np.int64)
        np.add.at(H, (uP, sb, (uqh > 0).astype(int), (uqc > 0).astype(int)), 1)
        res[f"deploy_{sname}"] = H
    # charge of covered vs all truth cells (coverage of the truth itself)
    for nm in SUP:
        k_, q_ = truths[nm]
        inU = np.isin(k_, U)
        res[f"truth_cov_{nm}"] = np.array([len(k_), int(inU.sum()), q_.sum(), q_[inU].sum()])
        res[f"truth_qhist_{nm}"] = np.histogram(q_, QE)[0]
        res[f"truth_qhist_cov_{nm}"] = np.histogram(q_[inU], QE)[0]

    # ---- training rows, per support
    win_rows = []
    for si, nm in enumerate(SUP):
        tk, tq = truths[nm]
        T = np.zeros((2, nP, nD, len(PARTS) + 1, 2), np.int64)    # [stratifier][plane][dbin][part|all][charged]
        Qh = np.zeros((nQ,), np.int64)                            # charge hist of trained charged rows (support q)
        Qh_hits = np.zeros((nQ + 1,), np.int64)                   # ... of those rows' HITS charge (bin 0: hits 0)
        parts_n = np.zeros(len(PARTS) + 1, np.int64)
        n_rows = 0
        for d in range(draws):
            rng = np.random.default_rng((seed, si, d, zlib.crc32(f'{run}{tag}{pos}'.encode())))
            rows, mask, ck, n_sig_pre, wins, wins_near = rows_with_parts(ds, ce, tk, tq, rng)
            if d == 0:      # exactness check against the trainer's own rows()
                ref, _y = ds.rows(ce, tk, tq, np.random.default_rng((seed, si, d, zlib.crc32(f'{run}{tag}{pos}'.encode()))))
                assert np.array_equal(ref, rows), (len(ref), len(rows))
            idx, _ = cell_inputs(rows, B["cell_key"], bl, ds.cfg)
            ok = (idx >= 0).any(1)
            rows, mask = rows[ok], mask[ok]
            P, qh, qc, hd, cd = strat(rows)
            qs = qh if nm == "hits" else qc
            chg = (qs > 0).astype(int)
            for st, sb in ((0, hd), (1, cd)):
                np.add.at(T, (st, P, sb, len(PARTS), chg), 1)
                for pi in range(len(PARTS)):
                    m = (mask >> pi) & 1 == 1
                    np.add.at(T, (st, P[m], sb[m], pi, chg[m]), 1)
            sel = (mask & 1 == 1) & (qs > 0)
            Qh += np.histogram(qs[sel], QE)[0]
            Qh_hits += np.histogram(qh[sel], np.concatenate([[-1, 1e-9], QE[1:]]))[0]
            for pi in range(len(PARTS)):
                parts_n[pi] += int(((mask >> pi) & 1).sum())
            parts_n[-1] += len(rows); n_rows += len(rows)
            if d == 0:
                # window distances: support charge, hits charge, complete > 0/25/102/500 e-
                for kind, wl, nwf, ntf in (("win", wins, 16, 8), ("win_near", wins_near, 16, 8)):
                    for (g, w0, t0) in wl:
                        dd = [grids[nm].win_dist(g, w0, t0, nwf, ntf, 0.0), grids["hits"].win_dist(g, w0, t0, nwf, ntf, 0.0)]
                        dd += [grids["complete"].win_dist(g, w0, t0, nwf, ntf, th) for th in (0.0, 25.0, 102.0, 500.0)]
                        qin = [float(grids[x].Q[g][max(w0, 0):w0 + nwf, max(t0 - grids[x].t_lo, 0):t0 - grids[x].t_lo + ntf].max(initial=0))
                               for x in SUP]
                        win_rows.append([si, 0 if kind == "win" else 1, g % 3] + dd + qin)
                res[f"nsig_pre_{nm}"] = n_sig_pre
                res[f"nwin_{nm}"] = len(wins); res[f"nwin_near_{nm}"] = len(wins_near)
        res[f"train_{nm}"] = T; res[f"qtrain_{nm}"] = Qh; res[f"qtrain_hitsq_{nm}"] = Qh_hits
        res[f"parts_{nm}"] = parts_n
        # all candidate noise windows (not just the 960 tried) and near-window origins
        u, nok = win_candidates(ck, grids[nm])
        res[f"wincand_{nm}"] = np.array([len(u), nok, len(ck)])
        nc = near_candidates(tk, tq)
        res[f"nearcand_{nm}"] = np.array([nc.get(g, 0) for g in gids])
    res["windist"] = np.asarray(win_rows, np.float64).reshape(-1, 11)
    res["n_cov"] = len(U)
    np.savez(out + ".tmp.npz", **res)
    os.replace(out + ".tmp.npz", out)
    return out, f"{time.time() - t00:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0010")
    ap.add_argument("--per-run", type=int, default=25)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    import json
    import h5py
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for run in RUNS:
        path = os.path.join(CORPUS, run, f"sim_wire_coeff_{a.shard}.h5")
        hold = json.load(open(os.path.join(CORPUS, run, "holdout.json")))
        held = {(e["source_file"], int(e["event"])) for s in ("probe", "val") for e in hold[s]}
        with h5py.File(path, "r") as f:
            ev = f["ident"]["event"][:].astype(np.int64)
            sf = [x.decode() if isinstance(x, bytes) else str(x) for x in f["ident"]["source_file"][:]]
        tr = [(p, e) for p, (e, s) in enumerate(zip(ev.tolist(), sf)) if (s, e) not in held]
        for p, e in tr[:a.per_run]:
            jobs.append((run, path, p, e, a.out, a.draws, a.seed))
    print(f"{len(jobs)} train events", flush=True)
    with get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:
        for o, msg in pool.imap_unordered(one_event, jobs):
            print(os.path.basename(o), msg, flush=True)


if __name__ == "__main__":
    main()
