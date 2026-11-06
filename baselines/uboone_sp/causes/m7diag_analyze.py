#!/usr/bin/env python3
"""M7 diagnostics from m7diag_build.py records (test ev260-387; calibration ev000-063 minus ev041).

    scripts/helix_run.sh python baselines/uboone_sp/causes/m7diag_analyze.py \
        --out baselines/uboone_sp/causes/results/m7_diagnostics.json

Q1 bright / top-end charge (cells, tiles, particles; truth-binned linearity with saturation split and
   the mechanism checks; error correlation structure and what the shared part correlates with).
Q2 the faintest predicted calo bins: truth/pred = presence x conditional charge x sub-25 e- factor.
Q3 distribution fidelity: law of total variance, per-cell charge distributions, dE/dx-like widths.
Conventions as bright_analyze.py: r = pred / truth - 1 clipped to [-1, 1]; hw68 = half the 16-84% spread.
Nothing is fitted on test events; the only fits (Q1b pred-binned calibration, spectrum slope) use the
calibration events.
"""
import argparse
import glob
import json
import os

import numpy as np

C3 = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
M7 = os.path.join(C3, "m7diag")
BB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "bright_budget.json")
PL = "UVY"
Q0 = 2043.2578108909702
BR = 16000.0
MODELS = ("M6d", "M6dc", "M7d", "M7dc")
HURDLE = {"M6d": False, "M6dc": True, "M7d": False, "M7dc": True, "M7d32": False, "M7dc32": True}
DEDGE = np.array([-1, 0.5, 1e4, 1e5, 1e6, 4e6, np.inf]); DLAB = ["iso", "<=10k", "10k-100k", "100k-1M", "1M-4M", ">4M"]


# ----------------------------------------------------------------------------------------- io
def load(d):
    cols = json.load(open(os.path.join(d, "columns.json")))
    R, T, P, labs = [], [], [], None
    for f in sorted(glob.glob(os.path.join(d, "ev*.npz"))):
        z = np.load(f)
        lb = [str(x) for x in z["models"]]
        if labs is None:
            labs = lb
        assert lb == labs, (f, lb, labs)
        R.append(z["rec"]); T.append(z["tiles"]); P.append(z["parts"])
    rec = np.concatenate(R); del R
    til = np.concatenate(T); par = np.concatenate([p for p in P if len(p)])
    nb, mc = len(cols["rec_cols"]), cols["rec_model_cols"]
    X = {c: rec[:, i] for i, c in enumerate(cols["rec_cols"])}
    for j, l in enumerate(labs):
        for i, c in enumerate(mc):
            X[f"{l}_{c}"] = rec[:, nb + j * len(mc) + i]
        X[f"{l}_gated"] = np.where(X[f"{l}_p"] > 0.5, X[f"{l}_q"], 0).astype(np.float32)
    nt, tm = len(cols["tile_cols"]), cols["tile_model_cols"]
    Tl = {c: til[:, i] for i, c in enumerate(cols["tile_cols"])}
    Pa = {c: par[:, i] for i, c in enumerate(cols["part_cols"])}
    npc, pm = len(cols["part_cols"]), cols["part_model_cols"]
    for j, l in enumerate(labs):
        for i, c in enumerate(tm):
            Tl[f"{l}_{c}"] = til[:, nt + j * len(tm) + i]
            Pa[f"{l}_{c}"] = par[:, npc + j * len(pm) + i]
    return X, Tl, Pa, labs


# ----------------------------------------------------------------------------------------- stats
def rel(pred, tru):
    return np.clip(np.asarray(pred, np.float64) / np.maximum(np.asarray(tru, np.float64), 1e-9) - 1.0, -1.0, 1.0)


def stats(r):
    r = np.asarray(r, np.float64)
    if len(r) < 10:
        return None
    p16, p50, p84 = np.percentile(r, [16, 50, 84])
    return dict(n=int(len(r)), median=float(p50), hw68=float((p84 - p16) / 2), mean=float(r.mean()), var=float(r.var()))


def ratio_stats(x):
    """median / hw68 / mean of a positive ratio (not clipped)."""
    x = np.asarray(x, np.float64)
    if len(x) < 10:
        return None if len(x) == 0 else dict(n=int(len(x)), median=float(np.median(x)))
    p16, p50, p84 = np.percentile(x, [16, 50, 84])
    return dict(n=int(len(x)), median=float(p50), hw68=float((p84 - p16) / 2), mean=float(x.mean()))


def eta2(r, b):
    if len(r) < 20:
        return None
    u, inv = np.unique(b, return_inverse=True)
    mu = np.bincount(inv, weights=r) / np.bincount(inv)
    return float(((mu[inv] - r.mean()) ** 2).mean() / max(r.var(), 1e-30))


def deciles(x):
    q = np.unique(np.percentile(x, np.linspace(0, 100, 11)[1:-1]))
    return np.searchsorted(q, x)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def w1_log(a, b, n=2000):
    """Wasserstein-1 between the distributions of log10 a and log10 b (quantile grid)."""
    qs = np.linspace(0.0005, 0.9995, n)
    return float(np.mean(np.abs(np.quantile(np.log10(a), qs) - np.quantile(np.log10(b), qs))))


def maps_of(labs):
    out = []
    for l in labs:
        out += [(l, "q"), (l, "gated"), (l, "calo"), (l, "clean")]
    return out


# ----------------------------------------------------------------------------------------- Q1a
def q1a(X, Tl, Pa, labs, floor_events):
    res = dict(cells={}, tiles={}, particles={}, floor={})
    plane = X["g"] % 3
    for P in range(3):
        m0 = (X["Tf"] >= BR) & (plane == P)
        res["cells"][PL[P]] = {f"{l}_{k}": stats(rel(X[f"{l}_{k}"][m0], X["Tf"][m0])) for l, k in maps_of(labs)}
        res["cells"][PL[P]]["hits_truth"] = stats(rel(X["Th"][m0], X["Tf"][m0]))
        res["cells"][PL[P]]["n_sat"] = int((X["sat"][m0] > 0).sum())
        for lev, lab in ((0, "2x2"), (1, "4x4"), (2, "16x8")):
            mt = (Tl["level"] == lev) & (Tl["g"] % 3 == P) & (Tl["Tf"] >= 64e3)
            res["tiles"][f"{PL[P]}_{lab}"] = {f"{l}_{k}": stats(rel(Tl[f"{l}_{k}"][mt], Tl["Tf"][mt])) for l, k in maps_of(labs)}
        mp = (Pa["g"] % 3 == P) & (Pa["E"] >= 1.0)
        res["particles"][PL[P]] = {f"{l}_{k}": stats(rel(Pa[f"{l}_{k}"][mp], Pa["q"][mp])) for l, k in maps_of(labs)}
        res["particles"][PL[P]]["hits_truth"] = stats(rel(Pa["Th_share"][mp], Pa["q"][mp]))
    # floors: bright_budget.json numbers (floor.py, ev260-307) + the models on the same events
    bb = json.load(open(BB))["floor"]
    fe = np.isin(X["ev"], floor_events)
    tfe = np.isin(Tl["ev"], floor_events)
    for P in range(3):
        f = bb[PL[P]]
        row = {}
        for lev in ("cell", "2x2", "4x4", "16x8"):
            for inp in ("clean", "noisy", "corpus"):
                row[f"{lev}_{inp}_oracle"] = f[f"{lev}_{inp}_oracle"]["hw68"]
                tk = {iv: f[f"{lev}_{inp}_{iv}"]["hw68"] for iv in ("tik0.0001", "tik0.001", "tik0.01") if f.get(f"{lev}_{inp}_{iv}")}
                if tk:
                    best = min(tk, key=tk.get)
                    row[f"{lev}_{inp}_best_tik"] = tk[best]; row[f"{lev}_{inp}_best_tik_lambda"] = best
        m0 = fe & (X["Tf"] >= BR) & (plane == P)
        for l, k in maps_of(labs):
            row[f"cell_model_{l}_{k}"] = stats(rel(X[f"{l}_{k}"][m0], X["Tf"][m0]))
            for lev, lab in ((0, "2x2"), (1, "4x4"), (2, "16x8")):
                mt = tfe & (Tl["level"] == lev) & (Tl["g"] % 3 == P) & (Tl["Tf"] >= 64e3)
                row[f"{lab}_model_{l}_{k}"] = stats(rel(Tl[f"{l}_{k}"][mt], Tl["Tf"][mt]))
        for l in labs:
            if row[f"cell_model_{l}_calo"] is None:
                continue
            hm = row[f"cell_model_{l}_calo"]["hw68"]; hf = row["cell_corpus_best_tik"]
            row[f"ineff_share_{l}_calo_vs_corpus_tik"] = float(1 - hf ** 2 / hm ** 2)
            hm = res["cells"][PL[P]][f"{l}_calo"]["hw68"]
            row[f"ineff_share_{l}_calo_allevents_vs_corpus_tik"] = float(1 - hf ** 2 / hm ** 2)
        res["floor"][PL[P]] = row
    return res


# ----------------------------------------------------------------------------------------- Q1b
TB = np.array([16e3, 64e3, 256e3, 1e6, 2e6, np.inf]); TBL = ["16-64k", "64-256k", "256k-1M", "1-2M", ">2M"]


def bf16_round(y):
    u = np.asarray(y, np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def pred_calibration(Xc, m, P):
    """bright_analyze.calibrate: median truth/pred in 20 log bins of the prediction on CALIBRATION events."""
    edges = np.logspace(np.log10(8000), np.log10(4e6), 21)
    sc = (Xc["g"] % 3 == P) & (Xc[m] > 8000) & (Xc["Tf"] > 0)
    ratio = Xc["Tf"][sc] / Xc[m][sc]
    b = np.clip(np.searchsorted(edges, Xc[m][sc]) - 1, 0, 19)
    fac = np.array([np.median(ratio[b == i]) if (b == i).sum() >= 30 else np.nan for i in range(20)])
    good = np.isfinite(fac)
    fac = np.interp(np.arange(20), np.flatnonzero(good), fac[good])
    return edges, fac


def q1b(X, Xc, labs):
    res = dict(truth_binned={}, pred_binned={}, mechanism={})
    plane = X["g"] % 3
    maps = [f"{l}_{k}" for l in labs for k in ("q", "calo")]
    for P in range(3):
        for s_lab, smask in (("unsat", X["sat"] == 0), ("sat", X["sat"] > 0), ("all", np.ones(len(X["sat"]), bool))):
            for i, bl in enumerate(TBL):
                m0 = (plane == P) & smask & (X["Tf"] >= TB[i]) & (X["Tf"] < TB[i + 1])
                res["truth_binned"][f"{PL[P]}_{s_lab}_{bl}"] = {mp: ratio_stats(X[mp][m0] / X["Tf"][m0]) for mp in maps}
        for i, bl in enumerate(TBL):
            for mp in ("M7dc_calo", "M7d_calo", "M6dc_calo", "M6d_calo", "M7dc32_calo"):
                if mp not in X:
                    continue
                m0 = (plane == P) & (X[mp] >= TB[i]) & (X[mp] < TB[i + 1])
                st = ratio_stats(X["Tf"][m0] / X[mp][m0])
                if st:
                    st["sum_ratio"] = float(X["Tf"][m0].sum() / X[mp][m0].sum())
                    st["n_sat"] = int((X["sat"][m0] > 0).sum())
                res["pred_binned"][f"{PL[P]}_{bl}_{mp}"] = st
    # ---- mechanism checks
    mech = {}
    # (1) the y range the model was trained on: complete-truth y of charged cells on calibration events
    yT_cal = np.log1p(Xc["Tf"][Xc["Tf"] > 25] / Q0)
    mech["cal_truth_y"] = dict(max=float(yT_cal.max()), p99_9=float(np.percentile(yT_cal, 99.9)),
                              p99_99=float(np.percentile(yT_cal, 99.99)),
                              frac_cells_gt_1M=float((Xc["Tf"] > 1e6).sum() / (Xc["Tf"] > 25).sum()),
                              n_cells_gt_1M=int((Xc["Tf"] > 1e6).sum()), n_cells_gt_2M=int((Xc["Tf"] > 2e6).sum()),
                              n_charged=int((Xc["Tf"] > 25).sum()), n_events=int(len(np.unique(Xc["ev"]))))
    for P in range(3):
        mP = plane == P
        row = {}
        yT = np.log1p(X["Tf"][mP] / Q0)
        for l in labs:
            yP = np.log1p(X[f"{l}_q"][mP] / Q0)
            row[f"{l}_max_y_pred"] = float(yP.max()); row["max_y_truth"] = float(yT.max())
            for i, bl in enumerate(TBL[2:], 2):
                m = (X["Tf"][mP] >= TB[i]) & (X["Tf"][mP] < TB[i + 1]) & (X["sat"][mP] == 0)
                if m.sum() >= 5:
                    row[f"{l}_ylogres_{bl}"] = dict(n=int(m.sum()), mean_yP_minus_yT=float((yP[m] - yT[m]).mean()),
                                                   max_yP=float(yP[m].max()), max_yT=float(yT[m].max()))
            # slopes in log space, unsaturated cells with Tf >= 256 k: pred on truth and truth on pred
            m = (X["Tf"][mP] >= 256e3) & (X["sat"][mP] == 0) & (yP > 0)
            if m.sum() > 20:
                b1 = np.polyfit(yT[m], yP[m], 1)[0]
                m2 = (yP >= np.log1p(256e3 / Q0)) & (X["sat"][mP] == 0)
                b2 = np.polyfit(yP[m2], yT[m2], 1)[0] if m2.sum() > 20 else None
                row[f"{l}_slope_ypred_on_ytruth_ge256k"] = float(b1)
                row[f"{l}_slope_ytruth_on_ypred_ge256k"] = None if b2 is None else float(b2)
        # (2) bf16 step: grid occupancy and bf16 vs fp32 at the top
        for l in ("M7dc", "M7d"):
            yP = np.log1p(X[f"{l}_q"][mP] / Q0)
            m = yP > 4
            row[f"{l}_frac_y_on_bf16_grid_y_gt4"] = float((np.abs(yP[m] - bf16_round(yP[m])) < 2e-4).mean())
            if f"{l}32_q" in X:
                y32 = np.log1p(X[f"{l}32_q"][mP] / Q0)
                row[f"{l}32_frac_y_on_bf16_grid_y_gt4"] = float((np.abs(y32[m] - bf16_round(y32[m])) < 2e-4).mean())
                for i, bl in enumerate(TBL):
                    mm = (X["Tf"][mP] >= TB[i]) & (X["Tf"][mP] < TB[i + 1]) & (X["sat"][mP] == 0)
                    if mm.sum() >= 5:
                        row[f"{l}_bf16_minus_fp32_logq_{bl}"] = dict(mean=float((yP[mm] - y32[mm]).mean()),
                                                                     rms=float(np.sqrt(((yP[mm] - y32[mm]) ** 2).mean())))
        # (3) the calo / Jensen factor at the top
        for l in labs:
            for i, bl in enumerate(TBL):
                mm = (X["Tf"][mP] >= TB[i]) & (X["Tf"][mP] < TB[i + 1])
                q_ = X[f"{l}_q"][mP][mm]
                if mm.sum() >= 5 and (q_ > 0).any():
                    k = q_ > 0
                    row[f"{l}_calo_over_q_{bl}"] = float(np.median(X[f"{l}_calo"][mP][mm][k] / q_[k]))
                    row[f"{l}_pcal_{bl}"] = float(np.median(X[f"{l}_pcal"][mP][mm]))
        # (4) posterior-shrinkage estimate: truth-binned shift ~ -s sigma^2 (power-law prior, Gaussian log error)
        #     s = -dlnN/dlnT of the complete-truth cell spectrum (calibration events); sigma = hw68 of
        #     ln(Tf/calo) per PREDICTED bin on the calibration events (M7dc).
        mc = (Xc["g"] % 3 == P)
        lt = np.log(Xc["Tf"][mc][Xc["Tf"][mc] >= 8e3])
        for i, bl in enumerate(TBL):
            lo, hi = np.log(TB[i]), np.log(min(TB[i + 1], 8e6))
            e = np.linspace(lo - 0.35, hi + 0.35, 6) if np.isfinite(hi) else None
            h, ed = np.histogram(lt, bins=np.linspace(lo - 0.5, hi + 0.5, 9))
            c = 0.5 * (ed[1:] + ed[:-1]); ok = h > 0
            s = -np.polyfit(c[ok], np.log(h[ok]), 1)[0] if ok.sum() >= 3 else None
            for l in ("M7dc", "M7d", "M6dc", "M6d"):
                pm = mc & (Xc[f"{l}_calo"] >= TB[i]) & (Xc[f"{l}_calo"] < TB[i + 1]) & (Xc["sat"] == 0) & (Xc["Tf"] > 0)
                if pm.sum() >= 20 and s is not None:
                    lr = np.log(Xc["Tf"][pm] / Xc[f"{l}_calo"][pm])
                    p16, p84 = np.percentile(lr, [16, 84]); sig = (p84 - p16) / 2
                    row[f"shrink_{l}_{bl}"] = dict(spectrum_slope_s=float(s), sigma_log_cal=float(sig),
                                                   predicted_truth_binned_shift=float(-s * sig ** 2), n_cal=int(pm.sum()))
        mech[PL[P]] = row
    # (5) diagnostic, deployable correction fitted on CALIBRATION events: pred-binned median truth/pred
    cal = {}
    for P in range(3):
        for l in ("M7dc", "M7d", "M6dc", "M6d"):
            mp = f"{l}_calo"
            edges, fac = pred_calibration(Xc, mp, P)
            m0 = (plane == P)
            b = np.clip(np.searchsorted(edges, X[mp][m0]) - 1, 0, 19)
            cq = np.where(X[mp][m0] > 8000, X[mp][m0] * fac[b], X[mp][m0])
            row = dict(factors=fac.tolist())
            for i, bl in enumerate(TBL):
                mm = (X["Tf"][m0] >= TB[i]) & (X["Tf"][m0] < TB[i + 1]) & (X["sat"][m0] == 0)
                row[bl] = dict(before=ratio_stats(X[mp][m0][mm] / X["Tf"][m0][mm]), after=ratio_stats(cq[mm] / X["Tf"][m0][mm]))
            cal[f"{PL[P]}_{l}"] = row
    mech["cal_pred_binned_correction"] = cal
    res["mechanism"] = mech
    return res


# ----------------------------------------------------------------------------------------- Q1c
def q1c(X, labs):
    res = {}
    plane = X["g"] % 3
    m0 = X["Tf"] >= BR
    idx = np.flatnonzero(m0)
    key = (((X["ev"][idx].astype(np.int64) * 6 + X["g"][idx].astype(np.int64)) * 1024 + X["fw"][idx].astype(np.int64)) * 512
           + X["ft"][idx].astype(np.int64))
    o = np.argsort(key); key, idx = key[o], idx[o]
    def pairs(dw, dt):
        k2 = key + dw * 512 + dt
        pos = np.clip(np.searchsorted(key, k2), 0, len(key) - 1)
        ok = key[pos] == k2
        return idx[ok], idx[pos[ok]]
    lags = {"wire": [(d, 0) for d in range(1, 9)], "time": [(0, d) for d in range(1, 9)], "diag": [(1, 1), (2, 2)]}
    PR = {(dw, dt): pairs(dw, dt) for v in lags.values() for (dw, dt) in v}
    for mp in ("M7dc_calo", "M6dc_calo", "M7d_calo", "M6d_calo", "M7dc32_calo"):
        if mp not in X:
            continue
        r = rel(X[mp], X["Tf"])
        row = {}
        for P in list(range(3)) + ["all"]:
            sub = {}
            for name, L in lags.items():
                cs = []
                for (dw, dt) in L:
                    a, b = PR[(dw, dt)]
                    if P != "all":
                        k = plane[a] == P; a, b = a[k], b[k]
                    k = (X["sat"][a] == 0) & (X["sat"][b] == 0); a, b = a[k], b[k]
                    cs.append(dict(lag=[dw, dt], n=int(len(a)), corr=float(np.corrcoef(r[a], r[b])[0, 1]),
                                   spearman=spearman(r[a], r[b]) if len(a) < 3e6 else None))
                sub[name] = cs
                c = np.array([x["corr"] for x in cs if x["lag"][0] == 0 or x["lag"][1] == 0])
                if name in ("wire", "time"):
                    d = np.arange(1, len(c) + 1)
                    tgt = np.exp(-1)
                    full = np.r_[1.0, c]; dd = np.r_[0, d]
                    j = np.flatnonzero(full < tgt)
                    sub[f"{name}_1e_lag"] = (float(dd[j[0] - 1] + (full[j[0] - 1] - tgt) / (full[j[0] - 1] - full[j[0]]))
                                             if len(j) else None)
                    # exponential length from lags 1..4 (shape beyond the nugget at lag 0)
                    k = c[:4] > 0.01
                    sub[f"{name}_exp_len_lag1to4"] = float(-1 / np.polyfit(d[:4][k], np.log(c[:4][k]), 1)[0]) if k.sum() >= 2 else None
            # same-particle vs different-particle neighbours (lag 1 along wire / time)
            for name, (dw, dt) in (("wire1", (1, 0)), ("time1", (0, 1))):
                a, b = PR[(dw, dt)]
                if P != "all":
                    k = plane[a] == P; a, b = a[k], b[k]
                k = (X["sat"][a] == 0) & (X["sat"][b] == 0); a, b = a[k], b[k]
                same = (X["maj_trk"][a] == X["maj_trk"][b]) & (X["maj_share"][a] >= 0.9) & (X["maj_share"][b] >= 0.9) & (X["maj_trk"][a] >= 0)
                sub[f"{name}_same_particle"] = dict(n=int(same.sum()), corr=float(np.corrcoef(r[a][same], r[b][same])[0, 1]))
                sub[f"{name}_mixed_or_other"] = dict(n=int((~same).sum()), corr=float(np.corrcoef(r[a][~same], r[b][~same])[0, 1]))
            row[PL[P] if P != "all" else "all"] = sub
        res[f"corr_{mp}"] = row
    # ---- shared component: 2x2 tiles with >= 3 bright unsaturated cells; s = tile mean of r (M7dc calo)
    for mp in ("M7dc_calo", "M6dc_calo"):
        r = rel(X[mp], X["Tf"])
        k = m0 & (X["sat"] == 0)
        ii = np.flatnonzero(k)
        tk = (((X["ev"][ii].astype(np.int64) * 6 + X["g"][ii].astype(np.int64)) * 1024 + X["fw"][ii].astype(np.int64) // 2) * 512
              + X["ft"][ii].astype(np.int64) // 2)
        u, inv, cnt = np.unique(tk, return_inverse=True, return_counts=True)
        sel = cnt[inv] >= 3
        ii, inv = ii[sel], inv[sel]
        u2, inv2 = np.unique(inv, return_inverse=True)
        n_ = np.bincount(inv2)
        s = np.bincount(inv2, weights=r[ii]) / n_
        within = r[ii] - s[inv2]
        out = dict(n_tiles=int(len(s)), n_cells=int(len(ii)), var_r=float(r[ii].var()), var_tile_mean_per_cell=float((s[inv2] - r[ii].mean()).var()),
                   var_within=float(within.var()))
        out["shared_share"] = out["var_tile_mean_per_cell"] / out["var_r"]
        # expected shared share if cells were independent: E[var(mean of n)] = var / n
        out["shared_share_if_independent"] = float(np.mean(1.0 / n_[inv2]))
        # tile covariates
        sm = lambda v: np.bincount(inv2, weights=np.asarray(v, np.float64)[ii]) / n_
        cov_ = dict(log_pred_charge=np.log10(np.maximum(np.bincount(inv2, weights=X[mp][ii].astype(np.float64)), 1)),
                    ncoef_tile=np.bincount(inv2, weights=X["ncoef"][ii].astype(np.float64)),
                    ncoef5_mean=sm(X["ncoef5"]),
                    ncoef_per_charge=np.bincount(inv2, weights=X["ncoef5"][ii].astype(np.float64)) / np.maximum(np.bincount(inv2, weights=X[mp][ii].astype(np.float64)), 1) * 1e5,
                    plane=sm(plane), time_pos_ft=sm(X["ft"]), drift_cm=sm(X["drift_cm"]),
                    pdens_log=np.log10(1 + sm(X[mp.replace("_calo", "_pd")])),
                    p_mean=sm(X[mp.replace("_calo", "_p")]),
                    truth_maj_share=sm(X["maj_share"]), truth_n10=sm(X["n10"]), truth_dedx=sm(X["dedx"]),
                    truth_log_charge=np.log10(np.bincount(inv2, weights=X["Tf"][ii].astype(np.float64))))
        cv = {}
        for name, v in cov_.items():
            b = v.astype(int) if name == "plane" else deciles(v)
            cv[name] = dict(eta2=eta2(s, b), spearman=spearman(s, v), deploy_time=not name.startswith(("truth_", "drift")))
        out["covariates"] = cv
        # per-plane shared share
        tp = np.round(cov_["plane"]).astype(int)
        out["by_plane"] = {PL[P]: dict(var_tile_mean=float(s[tp == P].var()), mean_tile_r=float(s[tp == P].mean())) for P in range(3)}
        res[f"shared_{mp}"] = out
    return res


# ----------------------------------------------------------------------------------------- Q2
FB = [(102, 500), (500, 1000), (1000, 2000)]


def q2_block(X, m, l):
    calo, pc, Tf = X[f"{l}_calo"][m].astype(np.float64), X[f"{l}_pcal"][m].astype(np.float64), X["Tf"][m].astype(np.float64)
    if len(calo) < 20:
        return None
    ch = Tf > 25
    S_ch, U = Tf[ch].sum(), Tf[(~ch)].sum()
    f_true, f_hat = ch.mean(), pc.mean()
    m_true = Tf[ch].mean() if ch.any() else np.nan
    m_hat = calo.sum() / pc.sum()
    chat = calo / np.maximum(pc, 1e-12)                      # conditional estimate (hurdle: qJ; non-hurdle: qJ / p)
    q = X[f"{l}_q"][m].astype(np.float64); y = np.log1p(q / Q0); yt = np.log1p(Tf / Q0)
    out = dict(n=int(m.sum()), truth_over_pred=float(Tf.sum() / calo.sum()),
               presence_true=float(f_true), presence_pcal=float(f_hat), presence_factor=float(f_true / f_hat),
               cond_true_mean=float(m_true), cond_hat_pcal_weighted=float(m_hat), cond_factor=float(m_true / m_hat),
               sub25_charge=float(U), sub25_factor=float(1 + U / S_ch) if S_ch > 0 else None,
               frac_cells_0_lt_Tf_le25=float(((Tf > 0) & ~ch).mean()), frac_cells_Tf0=float((Tf <= 0).mean()),
               cond_true_over_hat_on_charged=float(Tf[ch].mean() / chat[ch].mean()) if ch.any() else None,
               mean_chat_on_uncharged=float(chat[~ch].mean()) if (~ch).any() else None,
               pcal_on_charged=float(pc[ch].mean()) if ch.any() else None, pcal_on_uncharged=float(pc[~ch].mean()) if (~ch).any() else None,
               p_raw_mean=float(X[f"{l}_p"][m].mean()))
    out["identity_check"] = out["presence_factor"] * out["cond_factor"] * (out["sub25_factor"] or np.nan) / out["truth_over_pred"]
    if ch.sum() > 20:
        r = yt[ch] - y[ch]
        qJ = X[f"{l}_qJ"][m][ch].astype(np.float64)
        out["log_resid_charged"] = dict(mean=float(r.mean()), var=float(r.var()),
                                        jensen_applied=float(qJ.mean() / (Q0 * np.expm1(y[ch])).mean()),
                                        jensen_needed=float(Tf[ch].mean() / (Q0 * np.expm1(y[ch])).mean()),
                                        lognormal_pred_factor=float(np.mean(np.exp(r))), q_raw_mean=float(q[ch].mean()))
    # own-label presence for M6 (hits truth > 0)
    out["presence_own_label_hits"] = float((X["Th"][m] > 0).mean())
    return out


def q2(X, labs, tag):
    res = {}
    plane = X["g"] % 3
    for l in labs:
        if l.endswith("32"):
            continue
        for lo, hi in FB:
            mb = (X[f"{l}_calo"] >= lo) & (X[f"{l}_calo"] < hi)
            res[f"{l}_{lo}-{hi}_all"] = q2_block(X, mb, l)
            db = np.clip(np.searchsorted(DEDGE, X[f"{l}_pd"], side="left") - 1, 0, len(DLAB) - 1)
            for P in range(3):
                res[f"{l}_{lo}-{hi}_{PL[P]}"] = q2_block(X, mb & (plane == P), l)
            for d, dl in enumerate(DLAB):
                res[f"{l}_{lo}-{hi}_pdens_{dl}"] = q2_block(X, mb & (db == d), l)
    return res


# ----------------------------------------------------------------------------------------- Q3
CB = np.array([0, 102, 500, 1e3, 2e3, 4e3, 8e3, 16e3, 64e3, 256e3, np.inf])


def ltv(T, C):
    T = np.asarray(T, np.float64); C = np.asarray(C, np.float64); R = T - C
    vT = T.var()
    return dict(n=int(len(T)), var_T=float(vT), var_C_over_var_T=float(C.var() / vT), var_res_over_var_T=float(R.var() / vT),
                two_cov_over_var_T=float(2 * np.cov(C, R, bias=True)[0, 1] / vT), corr_C_res=float(np.corrcoef(C, R)[0, 1]),
                mean_T=float(T.mean()), mean_C=float(C.mean()))


def q3(X, Pa, labs):
    res = dict(ltv_cells={}, ltv_particles={}, hist={}, dedx={})
    plane = X["g"] % 3
    for l in ("M7dc", "M7d", "M6dc", "M6d"):
        mp = f"{l}_calo"
        for P in list(range(3)) + ["all"]:
            mP = np.ones(len(plane), bool) if P == "all" else plane == P
            m = mP & (X["cov"] > 0)
            row = dict(overall=ltv(X["Tf"][m], X[mp][m]))
            for i in range(len(CB) - 1):
                mb = m & (X[mp] >= CB[i]) & (X[mp] < CB[i + 1])
                if mb.sum() > 50:
                    row[f"{CB[i]:g}-{CB[i + 1]:g}"] = ltv(X["Tf"][mb], X[mp][mb])
            mc = m & (X["Tf"] >= 1000)
            row["charged_ge1k_same_cells"] = ltv(X["Tf"][mc], X[mp][mc])
            res["ltv_cells"][f"{l}_{PL[P] if P != 'all' else 'all'}"] = row
            mpp = (Pa["E"] >= 1.0) & ((Pa["g"] % 3 == P) if P != "all" else True)
            res["ltv_particles"][f"{l}_{PL[P] if P != 'all' else 'all'}"] = ltv(Pa["q"][mpp], Pa[mp][mpp])
    # ---- per-cell charge distributions, cells above 1 k e- (each map's own cells, and the truth's cells)
    qs = [1, 5, 16, 50, 84, 95, 99, 99.9]
    for P in range(3):
        mP = plane == P
        T = X["Tf"][mP & (X["Tf"] >= 1000)]
        row = dict(Tf=dict(n=int(len(T)), log10_quantiles=np.percentile(np.log10(T), qs).tolist()))
        for l in ("M7dc", "M7d", "M6dc", "M6d"):
            for k in ("q", "gated", "calo", "clean"):
                v = X[f"{l}_{k}"][mP]
                own = v[v >= 1000]
                same = X[f"{l}_{k}"][mP & (X["Tf"] >= 1000)]
                same = np.maximum(same, 1.0)
                row[f"{l}_{k}"] = dict(n_own=int(len(own)), n_own_over_n_truth=float(len(own) / len(T)),
                                       log10_quantiles_own=np.percentile(np.log10(own), qs).tolist(), w1_log10_own=w1_log(own, T),
                                       log10_quantiles_same_cells=np.percentile(np.log10(same), qs).tolist(),
                                       w1_log10_same_cells=w1_log(same, T),
                                       std_log10_own=float(np.log10(own).std()))
        row["Tf"]["std_log10"] = float(np.log10(T).std())
        res["hist"][PL[P]] = row
    # ---- dE/dx-like: single-particle MIP-like cells on MIP-like tracks
    key_p = {(int(e), int(g), int(t)): i for i, (e, g, t) in enumerate(zip(Pa["ev"], Pa["g"], Pa["trk_row"]))}
    sel = (X["maj_share"] >= 0.9) & (X["maj_trk"] >= 0) & (X["Tf"] >= 1000) & (X["sat"] == 0)
    ii = np.flatnonzero(sel)
    pi = np.array([key_p.get((int(e), int(g), int(t)), -1) for e, g, t in zip(X["ev"][ii], X["g"][ii], X["maj_trk"][ii])])
    ok = pi >= 0
    ii, pi = ii[ok], pi[ok]
    mip = (Pa["dedx_med"][pi] >= 1.5) & (Pa["dedx_med"][pi] <= 2.5) & (Pa["bragg_frac"][pi] < 0.05) & (Pa["length_mm"][pi] >= 100)
    ii, pi = ii[mip], pi[mip]
    up, inv, cnt = np.unique(pi, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 10
    ii, inv = ii[keep], inv[keep]
    up2, inv2 = np.unique(inv, return_inverse=True)
    def med_by(v):
        o = np.lexsort((v, inv2)); vs = v[o]; gs = inv2[o]
        st = np.searchsorted(gs, np.arange(len(up2))); en = np.searchsorted(gs, np.arange(len(up2)), side="right")
        return np.array([np.median(vs[a:b]) for a, b in zip(st, en)]), st, en, vs
    def trunc_mean(v, frac=0.6):
        o = np.lexsort((v, inv2)); vs = v[o]; gs = inv2[o]
        st = np.searchsorted(gs, np.arange(len(up2))); en = np.searchsorted(gs, np.arange(len(up2)), side="right")
        return np.array([vs[a:a + max(1, int(round(frac * (b - a))))].mean() for a, b in zip(st, en)])
    Tf = X["Tf"][ii].astype(np.float64)
    medT = med_by(Tf)[0]
    pln = (X["g"][ii] % 3).astype(int)
    trk_plane = np.array([int(Pa["g"][up[u]]) % 3 for u in up2])
    for l in ("M7dc", "M7d", "M6dc", "M6d"):
        for k in ("calo", "q", "clean"):
            C = X[f"{l}_{k}"][ii].astype(np.float64)
            medC = med_by(C)[0]
            tmT, tmC = trunc_mean(Tf), trunc_mean(C)
            for P in range(3):
                m = pln == P; mt = trk_plane == P
                xT = Tf[m] / medT[inv2[m]]; xC = C[m] / medT[inv2[m]]; xCs = C[m] / np.maximum(medC[inv2[m]], 1)
                pq = lambda x: dict(zip(["p1", "p10", "p16", "p50", "p84", "p90", "p99"], np.percentile(x, [1, 10, 16, 50, 84, 90, 99]).tolist()))
                hw = lambda x: float((np.percentile(x, 84) - np.percentile(x, 16)) / 2)
                rt = tmC[mt] / tmT[mt]
                res["dedx"][f"{l}_{k}_{PL[P]}"] = dict(
                    n_cells=int(m.sum()), n_tracks=int(mt.sum()),
                    cell_truth=dict(hw68=hw(xT), **pq(xT)), cell_pred_truthnorm=dict(hw68=hw(xC), **pq(xC)),
                    cell_pred_selfnorm=dict(hw68=hw(xCs), **pq(xCs)),
                    hw68_ratio_pred_over_truth=hw(xC) / hw(xT),
                    track_truncmean60_ratio=dict(median=float(np.median(rt)), hw68=hw(rt)),
                    track_truncmean60_spread=dict(truth_hw68_rel=hw(tmT[mt] / np.median(tmT[mt])), pred_hw68_rel=hw(tmC[mt] / np.median(tmC[mt]))),
                    track_mean_ratio=dict(median=float(np.median(trunc_mean(C, 1.0)[mt] / trunc_mean(Tf, 1.0)[mt]))))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default=os.path.join(M7, "cells"), help="test records (smoke tests only: another dir)")
    a = ap.parse_args()
    X, Tl, Pa, labs = load(a.cells)
    Xc, _, _, clabs = load(os.path.join(M7, "cal_cells"))
    print("loaded", len(X["Tf"]), "test cells", labs, len(Xc["Tf"]), "cal cells", flush=True)
    floor_events = sorted(int(os.path.basename(f)[2:5]) for f in glob.glob(os.path.join(C3, "bright", "floor", "ev*.npz")))
    res = dict(models=labs, n_test_events=int(len(np.unique(X["ev"]))), n_cal_events=int(len(np.unique(Xc["ev"]))),
               floor_events=floor_events)
    res["q1a"] = q1a(X, Tl, Pa, labs, floor_events); print("q1a", flush=True)
    res["q1b"] = q1b(X, Xc, labs); print("q1b", flush=True)
    res["q1c"] = q1c(X, labs); print("q1c", flush=True)
    res["q2_test"] = q2(X, labs, "test"); res["q2_cal"] = q2(Xc, clabs, "cal"); print("q2", flush=True)
    res["q3"] = q3(X, Pa, labs); print("q3", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
