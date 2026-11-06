#!/usr/bin/env python3
"""Stage 1 of the dense-activity comparison, v2 models (M4n / M5d raw and presence-gated, M5dc gated):
per-event records on identical cells. Same records as collect.py (see its docstring).

    scripts/helix_run.sh python baselines/uboone_sp/dense/collect2.py --events 260-387 \
        --cls /global/cfs/.../dense2/cls_fix,/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/cls_fit \
        --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/dense2/rec

--cls is a comma list searched in order (two cls_fit planes sit on a hung OST and were recomputed).
Gated = raw charge where sigmoid(presence) > 0.5, else 0 (eval_denoise "_gated").

Inputs (all full-plane fine-cell maps, readout only, 2 wires x 16 ticks):
  --cls      dense/cls_maps.py output: default, tuned_q, tuned_any, oracle, truth
  --model    dense/model_dump2.py outputs (M4n, M5d, M5dc; CFS);
             cells no token covers are 0 (the model's actual output)
  --ptruth   dense/particle_truth.py output (majority particle per cell, energies)

Per event one npz of records (every record carries the plane; events are files):
  sig_*   every truth-charged cell: T, local density (true charge and number of
          charged cells within +-8 cells, the cell excluded), majority-particle
          share, particle dominance of the +-8 box, n particles >= 10%, token
          coverage, and every method's score.
  emp_*   truth-empty cells, sampled with a density-dependent rate (weight = 1/rate):
          density, coverage, scores.
  fc      exact counts of empty cells above 0.05 / 0.25 / 1 Q0 per method, plane and
          fine density bin (FC_EDGES), with the empty-cell totals.
  reg16 / reg4   region tiles of 16 x 8 and 4 x 4 cells: true and predicted sums,
          number of charged cells (charged tiles all; empty tiles sampled 1/50).
  regfc*  exact counts / sums of predicted charge in empty tiles.
  par_*   per particle and plane: own charge, energy, cells where it is the
          majority, truth and prediction summed over them, overlap.
  sep_*   truth double peaks along wire / time (peaks >= 1 Q0, 2-4 cells apart,
          valley <= 0.5 x the weaker peak) and every charge method's values there.
  evt     per plane: true / predicted totals, map correlations (all cells, footprint).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
Q0 = 2043.258
R = 8                                                     # density box half-width, cells
CHARGE = ["cls_default", "cls_tuned_q", "cls_oracle", "M4n_q", "M4n_g", "M5d_q", "M5d_g", "M5dc_g"]
DETECT = ["cls_tuned_any", "M4n_p", "M5d_p", "M5dc_p"]
METHODS = CHARGE + DETECT
FC_EDGES = np.concatenate([[-1.0, 0.5], np.logspace(2, 8, 49)])   # dens_q bins: [0], then log 100 e- .. 1e8 e-
THR = np.array([0.05, 0.25, 1.0]) * Q0
MODEL = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/dense2/model"
PTRUTH = "/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/ptruth"


def box(X, r=R):
    """Sum over the (2r+1)^2 box around every cell (zero outside), exact integral image."""
    X = np.asarray(X, np.float64)
    P = np.pad(X, ((r + 1, r), (r + 1, r)))
    S = P.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]


def emp_rate(d):
    return np.select([d < 1, d < 5e3, d < 5e4, d < 5e5], [0.003, 0.03, 0.1, 0.3], 1.0)


def dominance(T, cells, pc, pt, pq, tmax, ncol):
    """Box (+-R) charge of each charged cell's majority particle / box total charge (cell included)."""
    tid, pti = np.unique(pt, return_inverse=True)
    ncell = T.size
    keys = pti.astype(np.int64) * ncell + pc
    o = np.argsort(keys); keys, vals = keys[o], pq[o].astype(np.float64)
    m = np.searchsorted(tid, tmax)
    fw, ft = cells // ncol, cells % ncol
    nwc = T.shape[0]
    own = np.zeros(len(cells))
    for dw in range(-R, R + 1):
        for dt in range(-R, R + 1):
            w2, t2 = fw + dw, ft + dt
            ok = (w2 >= 0) & (w2 < nwc) & (t2 >= 0) & (t2 < ncol)
            k2 = m[ok] * ncell + w2[ok] * ncol + t2[ok]
            i = np.minimum(np.searchsorted(keys, k2), len(keys) - 1)
            hit = keys[i] == k2
            v = np.zeros(ok.sum()); v[hit] = vals[i[hit]]
            own[ok] += v
    return own


def tiles(X, a, b):
    nw, nt = (X.shape[0] // a) * a, (X.shape[1] // b) * b
    return X[:nw, :nt].reshape(nw // a, a, nt // b, b).sum((1, 3)).ravel()


def sep_pairs(T, maps, plane):
    """Truth double peaks along each axis; per pair the methods' peak and valley values."""
    rows = []
    for ax in (0, 1):
        A = T if ax == 0 else T.T
        Ms = {n: (X if ax == 0 else X.T) for n, X in maps.items()}
        D = Ms.pop("_dens")
        n = A.shape[0]
        prev = np.vstack([np.full((1, A.shape[1]), -1.0), A[:-1]])
        nxt = np.vstack([A[1:], np.full((1, A.shape[1]), -1.0)])
        lm = (A >= prev) & (A >= nxt) & (A >= Q0)
        for s in (2, 3, 4):
            i, j = np.nonzero(lm[:n - s] & lm[s:])
            if not len(i):
                continue
            p1, p2 = A[i, j], A[i + s, j]
            between = np.stack([A[i + k, j] for k in range(1, s)], 1)
            pmin = np.minimum(p1, p2)
            keep = (between.max(1) < pmin) & (between.min(1) <= 0.5 * pmin)
            i, j, p1, p2, between, pmin = i[keep], j[keep], p1[keep], p2[keep], between[keep], pmin[keep]
            kv = 1 + np.argmin(between, 1)
            row = [np.full(len(i), plane), np.full(len(i), ax), np.full(len(i), s), pmin, between.min(1), D[i + kv, j]]
            for nm in CHARGE:
                X = Ms[nm]
                b = np.stack([X[i + k, j] for k in range(1, s)], 1)
                row += [np.minimum(X[i, j], X[i + s, j]), b.min(1)]
            rows.append(np.stack(row, 1))
    return np.concatenate(rows) if rows else np.zeros((0, 6 + 2 * len(CHARGE)))


def per_event(args):
    ev, cls_dir, out_dir = args
    out = os.path.join(out_dir, f"ev{ev:03d}.npz")
    if os.path.exists(out):
        return ev, "exists"
    rng = np.random.default_rng(7000 + ev)
    Mz = np.load(os.path.join(MODEL, f"ev{ev:03d}.npz"))
    PT = np.load(os.path.join(PTRUTH, f"ev{ev:03d}.npz"))
    trkE = dict(zip(PT["trk_id"].tolist(), PT["trk_E"].tolist()))
    acc = {k: [] for k in ("sig", "emp", "reg16", "reg4", "par", "sep", "evt")}
    fc = np.zeros((len(CHARGE), 3, len(FC_EDGES) - 1, 4))
    regfc = {16: np.zeros((len(CHARGE), 3, 5)), 4: np.zeros((len(CHARGE), 3, 5))}
    for g in range(6):
        p = g % 3
        F = np.load(next(p_ for p_ in (os.path.join(d_, f"ev{ev:03d}_g{g}.npz") for d_ in cls_dir.split(",")) if os.path.exists(p_)))
        T = F["truth"].astype(np.float64)
        to = int(Mz["toff"])
        crop = lambda X: X[:T.shape[0], to:to + T.shape[1]].astype(np.float64)
        maps = {"cls_default": F["default"], "cls_tuned_q": F["tuned_q"], "cls_oracle": F["oracle"],
                "cls_tuned_any": F["tuned_any"],
                "M4n_q": crop(Mz[f"q_M4n_g{g}"]), "M4n_p": crop(Mz[f"p_M4n_g{g}"]),
                "M5d_q": crop(Mz[f"q_M5d_g{g}"]), "M5d_p": crop(Mz[f"p_M5d_g{g}"]),
                "M5dc_q": crop(Mz[f"q_M5dc_g{g}"]), "M5dc_p": crop(Mz[f"p_M5dc_g{g}"])}
        for t_ in ("M4n", "M5d", "M5dc"):                     # eval_denoise's "_gated" map
            maps[f"{t_}_g"] = np.where(maps[f"{t_}_p"] > 0.5, maps[f"{t_}_q"], 0.0)
        del maps["M5dc_q"]                                     # hurdle model: its raw charge map is never used
        maps = {k: np.asarray(v, np.float64) for k, v in maps.items()}
        cov = crop(Mz[f"cov_g{g}"]) > 0
        has = T > 0
        bq = box(T)
        dq = np.where(has, bq - T, bq); dq[dq < 1.0] = 0.0
        dn = np.rint(box(has.astype(np.float64)) - has).astype(np.int32)
        ncol = T.shape[1]
        # ---------- charged cells
        cells = PT[f"c_{g}"]
        assert np.array_equal(np.flatnonzero(has.ravel()), cells), (ev, g)
        Tf = T.ravel()[cells]
        own = dominance(T, cells, PT[f"pc_{g}"], PT[f"pt_{g}"], PT[f"pq_{g}"], PT[f"tmax_{g}"], ncol)
        dom = own / bq.ravel()[cells]
        cols = [np.full(len(cells), p), Tf, dq.ravel()[cells], dn.ravel()[cells],
                PT[f"qmax_{g}"] / np.maximum(PT[f"q_{g}"], 1e-9), dom, PT[f"n10_{g}"], cov.ravel()[cells]]
        cols += [maps[m].ravel()[cells] for m in METHODS]
        acc["sig"].append(np.stack(cols, 1).astype(np.float32))
        # ---------- empty cells: exact false-charge counts + density-weighted sample
        emp = ~has
        de = dq[emp]
        db = np.clip(np.searchsorted(FC_EDGES, de, side="right") - 1, 0, len(FC_EDGES) - 2)
        tot = np.bincount(db, minlength=len(FC_EDGES) - 1)
        for i, m in enumerate(CHARGE):
            x = maps[m][emp]
            fc[i, p, :, 0] += tot
            for k, t in enumerate(THR):
                fc[i, p, :, k + 1] += np.bincount(db, weights=(x > t), minlength=len(FC_EDGES) - 1)
        ie = np.flatnonzero(emp.ravel())
        r = emp_rate(dq.ravel()[ie])
        sel = rng.random(len(ie)) < r
        ie, r = ie[sel], r[sel]
        cols = [np.full(len(ie), p), 1.0 / r, dq.ravel()[ie], dn.ravel()[ie], cov.ravel()[ie]]
        cols += [maps[m].ravel()[ie] for m in METHODS]
        acc["emp"].append(np.stack(cols, 1).astype(np.float32))
        # ---------- regions
        for (a, b), key in (((16, 8), "reg16"), ((4, 4), "reg4")):
            Ts, Ns = tiles(T, a, b), tiles(has.astype(np.float64), a, b)
            Ps = [tiles(maps[m], a, b) for m in CHARGE]
            Ds = tiles(dq * has, a, b) / np.maximum(Ns, 1)                # mean density of the tile's charged cells
            ch = Ts > 0
            keep = ch | (rng.random(len(Ts)) < 0.02)
            acc[key].append(np.stack([np.full(keep.sum(), p), np.where(ch[keep], 1.0, 50.0), Ts[keep], Ns[keep], Ds[keep]]
                                     + [x[keep] for x in Ps], 1).astype(np.float32))
            for i, x in enumerate(Ps):
                xe = x[~ch]
                regfc[a][i, p] += [len(xe), (xe > 0.25 * Q0).sum(), (xe > Q0).sum(), (xe > 4 * Q0).sum(), xe.sum()]
        # ---------- particles on this plane
        pc, pt, pq = PT[f"pc_{g}"], PT[f"pt_{g}"], PT[f"pq_{g}"].astype(np.float64)
        tid, inv = np.unique(pt, return_inverse=True)
        cell_tot = np.zeros(T.size); cell_tot[cells] = PT[f"q_{g}"]
        share = pq / cell_tot[pc]
        own_tot = np.bincount(inv, weights=pq, minlength=len(tid))
        own_mixed = np.bincount(inv, weights=pq * (share < 0.9), minlength=len(tid))
        own_dw = np.bincount(inv, weights=pq * dq.ravel()[pc], minlength=len(tid))
        mi = np.searchsorted(tid, PT[f"tmax_{g}"])
        nmaj = np.bincount(mi, minlength=len(tid))
        tmaj = np.bincount(mi, weights=Tf, minlength=len(tid))
        ownmaj = np.bincount(mi, weights=PT[f"qmax_{g}"].astype(np.float64), minlength=len(tid))
        keep = (own_tot >= Q0) & (nmaj > 0)
        cols = [np.full(keep.sum(), p), np.array([trkE.get(t, np.nan) for t in tid[keep]]), own_tot[keep],
                own_mixed[keep] / own_tot[keep], own_dw[keep] / own_tot[keep], nmaj[keep], tmaj[keep], ownmaj[keep]]
        cols += [np.bincount(mi, weights=maps[m].ravel()[cells], minlength=len(tid))[keep] for m in CHARGE]
        acc["par"].append(np.stack(cols, 1))
        # ---------- separation
        acc["sep"].append(sep_pairs(T, dict({m: maps[m] for m in CHARGE}, _dens=dq), p).astype(np.float32))
        # ---------- event level
        foot = dq + T > 0
        row = [p, T.sum()]
        for m in CHARGE:
            X = maps[m]
            r_all = np.corrcoef(X.ravel(), T.ravel())[0, 1]
            r_log = np.corrcoef(np.log1p(X.ravel() / Q0), np.log1p(T.ravel() / Q0))[0, 1]
            r_ft = np.corrcoef(np.log1p(X[foot] / Q0), np.log1p(T[foot] / Q0))[0, 1]
            row += [X.sum(), r_all, r_log, r_ft]
        acc["evt"].append(np.array(row))
    res = {k: (np.concatenate(v) if k != "evt" else np.stack(v)) for k, v in acc.items()}
    np.savez(out + ".tmp.npz", fc=fc, regfc16=regfc[16], regfc4=regfc[4], **res)
    os.replace(out + ".tmp.npz", out)
    return ev, f"sig {len(res['sig'])} emp {len(res['emp'])} sep {len(res['sep'])} par {len(res['par'])}"


def main():
    sys.path.insert(0, os.path.join(BASE, "v2"))
    from classical import parse_events
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--cls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(methods=METHODS, charge=CHARGE, detect=DETECT, fc_edges=FC_EDGES.tolist(), R=R, Q0=Q0,
                   sig_cols=["plane", "T", "dens_q", "dens_n", "maj_share", "dominance", "n10", "cov"] + METHODS,
                   emp_cols=["plane", "w", "dens_q", "dens_n", "cov"] + METHODS,
                   reg_cols=["plane", "w", "T", "n_charged", "mean_dens"] + CHARGE,
                   regfc_cols=["n_empty", "gt025Q0", "gt1Q0", "gt4Q0", "sum"],
                   par_cols=["plane", "E", "own", "mixed_frac", "dens_w", "n_maj", "T_maj", "own_maj"] + CHARGE,
                   sep_cols=["plane", "axis", "s", "Tpmin", "Tvalley", "dens_valley"]
                   + [f"{m}_{x}" for m in CHARGE for x in ("pmin", "vmin")],
                   evt_cols=["plane", "T"] + [f"{m}_{x}" for m in CHARGE for x in ("sum", "r", "rlog", "rlog_foot")]),
              open(os.path.join(a.out, "columns.json"), "w"), indent=1)
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for ev, msg in pool.imap_unordered(per_event, [(e, a.cls, a.out) for e in parse_events(a.events)]):
            print(f"ev{ev:03d}: {msg}", flush=True)


if __name__ == "__main__":
    main()
