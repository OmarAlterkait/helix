#!/usr/bin/env python3
"""Bright-charge error budget from bright_build.py (+ floor.py) records.

    scripts/helix_run.sh python baselines/uboone_sp/causes/bright_analyze.py --out baselines/uboone_sp/causes/results

Bright cell: complete truth Tf >= 16 k e- (the hits truth Th alongside). Relative error
r = pred / truth - 1, clipped to [-1, 1]. "hw68" = half the 16-84% spread of r; "var" = variance
of r; "bias" = median r (and mean r for the MSE decomposition MSE = mean^2 + var).
Explained share of a factor = between-bin variance of r over the factor's bins / total variance
(eta^2), i.e. the part of the per-cell variance a per-bin correction could remove.
"""
import argparse
import glob
import json
import os

import numpy as np

C3 = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/bright"
PL = "UVY"
BR = 16000.0


def load(d):
    cols = json.load(open(os.path.join(d, "columns.json")))
    R, T, P, names = [], [], [], None
    for f in sorted(glob.glob(os.path.join(d, "ev*.npz"))):
        z = np.load(f)
        nm = [str(x) for x in z["names"]]
        if names is None:
            names = nm
        if nm != names:
            continue
        R.append(z["rec"]); T.append(z["tiles"]); P.append(z["parts"])
    rec = np.concatenate(R); til = np.concatenate(T); par = np.concatenate([p for p in P if len(p)])
    X = {c: rec[:, i] for i, c in enumerate(cols["rec_cols"])}
    X.update({n: rec[:, len(cols["rec_cols"]) + j] for j, n in enumerate(names)})
    Tl = {c: til[:, i] for i, c in enumerate(cols["tile_cols"])}
    Tl.update({n: til[:, len(cols["tile_cols"]) + j] for j, n in enumerate(names)})
    Pa = {c: par[:, i] for i, c in enumerate(cols["part_cols"])}
    Pa.update({n: par[:, len(cols["part_cols"]) + j] for j, n in enumerate(names)})
    return X, Tl, Pa, names


def rel(pred, tru):
    return np.clip(pred / np.maximum(tru, 1e-9) - 1.0, -1.0, 1.0)


def stats(r):
    if len(r) < 10:
        return None
    p16, p50, p84 = np.percentile(r, [16, 50, 84])
    return dict(n=int(len(r)), median=float(p50), hw68=float((p84 - p16) / 2), mean=float(r.mean()), var=float(r.var()),
                mse=float((r ** 2).mean()))


def eta2(r, b):
    """Between-bin variance share of r over integer bins b."""
    if len(r) < 20:
        return None
    tot = r.var()
    u, inv = np.unique(b, return_inverse=True)
    mu = np.bincount(inv, weights=r) / np.bincount(inv)
    return float(((mu[inv] - r.mean()) ** 2).mean() / max(tot, 1e-30))


def binned(r, x, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        s = stats(r[m])
        if s:
            out.append(dict(lo=float(lo), hi=float(hi), **s))
    return out


def calibrate(Xc, Xt, m, plane, truth="Tf"):
    """Diagnostic calibration fitted on CALIBRATION events: per plane, the median of truth/pred in
    20 log bins of the prediction (bright region), applied multiplicatively on test."""
    edges = np.logspace(np.log10(8000), np.log10(4e6), 21)
    sc = (Xc["g"] % 3 == plane) & (Xc[m] > 8000) & (Xc[truth] > 0)
    ratio = Xc[truth][sc] / Xc[m][sc]
    b = np.clip(np.searchsorted(edges, Xc[m][sc]) - 1, 0, 19)
    fac = np.array([np.median(ratio[b == i]) if (b == i).sum() >= 30 else np.nan for i in range(20)])
    good = np.isfinite(fac)
    fac = np.interp(np.arange(20), np.flatnonzero(good), fac[good]) if good.any() else np.ones(20)
    st = (Xt["g"] % 3 == plane)
    bt = np.clip(np.searchsorted(edges, Xt[m]) - 1, 0, 19)
    return np.where(st, Xt[m] * fac[bt], Xt[m]), fac.tolist()


def floor_maps(evs):
    """Floor fine-cell maps (floor.py) for the given events -> dict (ev, g) -> {key: map}."""
    F = {}
    for ev in evs:
        f = os.path.join(C3, "floor", f"ev{ev:03d}.npz")
        if os.path.exists(f):
            F[ev] = np.load(f)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X, Tl, Pa, names = load(os.path.join(C3, "cells"))
    Xc, _, _, cnames = load(os.path.join(C3, "cal_cells"))
    models = [n for n in names if not n.endswith("_raw")]
    res = dict(models=models, n_test_events=int(len(np.unique(X["ev"]))), n_cal_events=int(len(np.unique(Xc["ev"]))))
    plane = X["g"] % 3
    for truth in ("Tf", "Th"):
        bright = X[truth] >= BR
        R = {}
        for P in range(3):
            m0 = bright & (plane == P)
            row = {}
            for n in models:
                row[n] = stats(rel(X[n][m0], X[truth][m0]))
            R[PL[P]] = row
        res[f"cells_{truth}"] = R
    # ---------------- 0. truth definition: hits truth vs complete truth, and the model against each
    tdef = {}
    for P in range(3):
        m0 = (X["Tf"] >= BR) & (plane == P)
        rth = rel(X["Th"][m0], X["Tf"][m0])
        row = dict(hits_vs_complete=stats(rth), hits_vs_complete_by_drift=binned(rth, X["drift_cm"][m0], np.linspace(0, 216, 9)))
        for n in ("M6d_bf16", "M6dc_bf16", "M4n_bf16", "cls_tuned"):
            rm = rel(X[n][m0], X["Tf"][m0]); rmh = rel(X[n][m0], np.maximum(X["Th"][m0], 1.0))
            row[n] = dict(vs_complete=stats(rm), vs_hits=stats(rmh),
                          bias_share_truthdef=float(np.median(rth) / np.median(rm)) if np.median(rm) != 0 else None,
                          mse_share_truthdef=float(1 - (rmh ** 2).mean() / (rm ** 2).mean()),
                          var_share_truthdef=float(1 - rmh.var() / rm.var()),
                          vs_hits_by_drift=binned(rmh, X["drift_cm"][m0], np.linspace(0, 216, 9)))
        tdef[PL[P]] = row
    pm = (Pa["E"] >= 1.0)
    for P in range(3):
        mp = pm & (Pa["g"] % 3 == P)
        rth = rel(Pa["Th_share"][mp], Pa["q"][mp])
        for n in ("M6d_bf16", "M6dc_bf16", "cls_tuned"):
            rm = rel(Pa[n][mp], Pa["q"][mp]); rmh = rel(Pa[n][mp], np.maximum(Pa["Th_share"][mp], 1.0))
            tdef[PL[P]][f"particle_{n}"] = dict(vs_complete=stats(rm), vs_hits=stats(rmh), hits_vs_complete=stats(rth),
                                                mse_share_truthdef=float(1 - (rmh ** 2).mean() / (rm ** 2).mean()))
    res["truth_definition"] = tdef
    # ---------------- 1. precision
    prec = {}
    for P in range(3):
        m0 = (X["Tf"] >= BR) & (plane == P)
        for t in ("M6d", "M6dc"):
            r16, r32h, r32 = (rel(X[f"{t}_{k}"][m0], X["Tf"][m0]) for k in ("bf16", "head32", "fp32"))
            d = np.log1p(X[f"{t}_fp32"][m0] / 2043.258) - np.log1p(X[f"{t}_bf16"][m0] / 2043.258)
            prec[f"{PL[P]}_{t}"] = dict(bf16=stats(r16), head32=stats(r32h), fp32=stats(r32),
                                        var_share_removed_fp32=float(1 - r32.var() / r16.var()),
                                        mse_share_removed_fp32=float(1 - (r32 ** 2).mean() / (r16 ** 2).mean()),
                                        rms_log_diff_fp32_vs_bf16=float(np.sqrt((d ** 2).mean())),
                                        corr_err_bf16_fp32=float(np.corrcoef(r16, r32)[0, 1]))
    res["precision"] = prec
    # ---------------- 2. systematic scale
    sysm = {}
    for P in range(3):
        m0 = (X["Tf"] >= BR) & (plane == P)
        for n in ("M6d_bf16", "M6dc_bf16", "M4n_bf16", "cls_tuned"):
            r = rel(X[n][m0], X["Tf"][m0])
            lq = np.log10(X["Tf"][m0])
            e_q = np.clip(((lq - np.log10(BR)) / 0.1).astype(int), 0, 40)
            e_d = np.clip((X["drift_cm"][m0] / 216 * 8).astype(int), 0, 7)
            e_t = np.select([X["ft"][m0] < 10, X["ft"][m0] > 260], [0, 2], 1)
            e_w = np.clip((X["fw"][m0] / (X["fw"].max() + 1) * 10).astype(int), 0, 9)
            row = dict(by_charge=binned(r, X["Tf"][m0], np.array([16e3, 32e3, 64e3, 128e3, 256e3, 512e3, 1e6, 2e6, 4e6, 1e8])),
                       by_drift=binned(r, X["drift_cm"][m0], np.linspace(0, 216, 9)),
                       by_readout_pos=binned(r, X["ft"][m0], np.array([0, 10, 30, 240, 260, 272])),
                       eta2_charge=eta2(r, e_q), eta2_drift=eta2(r, e_d), eta2_edges=eta2(r, e_t), eta2_wirepos=eta2(r, e_w))
            if n in cnames:
                # oracle-drift diagnostic: per (prediction bin, drift bin) median correction from calibration events
                ed = np.linspace(0, 216, 7); eq = np.logspace(np.log10(8000), np.log10(4e6), 13)
                sc = (Xc["g"] % 3 == P) & (Xc[n] > 8000)
                bc = np.clip(np.searchsorted(eq, Xc[n][sc]) - 1, 0, 11) * 6 + np.clip(np.searchsorted(ed, Xc["drift_cm"][sc]) - 1, 0, 5)
                rat = Xc["Tf"][sc] / Xc[n][sc]
                fac2 = np.array([np.median(rat[bc == i]) if (bc == i).sum() >= 30 else np.nan for i in range(72)])
                fac2 = np.where(np.isfinite(fac2), fac2, np.nanmedian(fac2))
                bt = np.clip(np.searchsorted(eq, X[n][m0]) - 1, 0, 11) * 6 + np.clip(np.searchsorted(ed, X["drift_cm"][m0]) - 1, 0, 5)
                rcd = rel(X[n][m0] * fac2[bt], X["Tf"][m0])
                row["calibrated_q_drift"] = stats(rcd)
                row["mse_share_removed_by_cal_q_drift"] = float(1 - (rcd ** 2).mean() / (r ** 2).mean())
                cal, fac = calibrate(Xc, X, n, P)
                rc = rel(cal[m0], X["Tf"][m0])
                row["calibrated"] = stats(rc); row["cal_factors"] = fac
                row["mse_share_removed_by_cal"] = float(1 - (rc ** 2).mean() / (r ** 2).mean())
                row["var_share_removed_by_cal"] = float(1 - rc.var() / r.var())
            sysm[f"{PL[P]}_{n}"] = row
    res["systematic"] = sysm
    # ---------------- 3. topology (cells) + 4. saturation (cells)
    top = {}
    for P in range(3):
        m0 = (X["Tf"] >= BR) & (plane == P)
        for n in ("M6d_bf16", "M6dc_bf16", "cls_tuned"):
            r = rel(X[n][m0], X["Tf"][m0])
            e_dx = np.digitize(X["dedx"][m0], [1.8, 2.2, 3.0, 4.0, 6.0, 10.0])
            e_sh = np.digitize(X["maj_share"][m0], [0.5, 0.7, 0.9, 0.99])
            e_n = np.clip(X["n10"][m0], 1, 4).astype(int)
            sat = X["sat"][m0] > 0
            row = dict(eta2_dedx=eta2(r, e_dx), eta2_majshare=eta2(r, e_sh), eta2_n10=eta2(r, e_n),
                       by_dedx=binned(r, X["dedx"][m0], np.array([0, 1.8, 2.2, 3.0, 4.0, 6.0, 10.0, 1e3])),
                       by_majshare=binned(r, X["maj_share"][m0], np.array([0, 0.5, 0.7, 0.9, 0.99, 1.01])),
                       sat_frac_cells=float(sat.mean()), sat=stats(r[sat]), unsat=stats(r[~sat]), eta2_sat=eta2(r, sat.astype(int)),
                       mse_share_sat=float((r[sat] ** 2).sum() / max((r ** 2).sum(), 1e-30)))
            # saturated vs unsaturated cells of equal charge: compare within charge bins >= 256 k
            hiq = X["Tf"][m0] >= 256e3
            row["sat_hiq"] = stats(r[sat & hiq]); row["unsat_hiq"] = stats(r[~sat & hiq])
            top[f"{PL[P]}_{n}"] = row
    res["topology_cells"] = top
    # particles
    part = {}
    pplane = Pa["g"] % 3
    for P in range(3):
        mp = (pplane == P) & (Pa["E"] >= 1.0)
        for n in ("M6d_bf16", "M6dc_bf16", "M6d_fp32", "M6dc_fp32", "cls_tuned"):
            r = rel(Pa[n][mp], Pa["q"][mp])
            rh = rel(Pa["Th_share"][mp], Pa["q"][mp])
            row = dict(all=stats(r), hits_truth_vs_complete=stats(rh),
                       by_th_drift=binned(r, Pa["th_drift"][mp], np.array([0, 20, 40, 60, 75, 85, 90.1])),
                       by_th_wire=binned(r, Pa["th_wire"][mp], np.array([0, 20, 40, 60, 75, 90.1])),
                       by_bragg=binned(r, Pa["bragg_frac"][mp], np.array([0, 0.05, 0.2, 0.5, 1.01])),
                       by_overlap=binned(r, Pa["overlap"][mp], np.array([0, 0.05, 0.2, 0.5, 1.01])),
                       by_E=binned(r, Pa["E"][mp], np.array([1, 3, 10, 30, 100, 1e5])),
                       by_sat=binned(r, Pa["sat_frac"][mp], np.array([0, 1e-9, 0.01, 1.01])),
                       eta2_th_drift=eta2(r, np.digitize(Pa["th_drift"][mp], [20, 40, 60, 75, 85])),
                       eta2_th_wire=eta2(r, np.digitize(Pa["th_wire"][mp], [20, 40, 60, 75])),
                       eta2_bragg=eta2(r, np.digitize(Pa["bragg_frac"][mp], [0.05, 0.2, 0.5])),
                       eta2_overlap=eta2(r, np.digitize(Pa["overlap"][mp], [0.05, 0.2, 0.5])),
                       eta2_E=eta2(r, np.digitize(Pa["E"][mp], [3, 10, 30, 100])),
                       eta2_sat=eta2(r, (Pa["sat_frac"][mp] > 0).astype(int)))
            part[f"{PL[P]}_{n}"] = row
    res["particles"] = part
    # ---------------- 6. grid: cell vs 2x2 / 4x4 / 16x8 tiles
    grid = {}
    for P in range(3):
        for lev, lab in ((0, "2x2"), (1, "4x4"), (2, "16x8")):
            mt = (Tl["level"] == lev) & (Tl["g"] % 3 == P) & (Tl["Tf"] >= 64e3)
            grid[f"{PL[P]}_{lab}"] = {n: stats(rel(Tl[n][mt], Tl["Tf"][mt])) for n in ("M6d_bf16", "M6dc_bf16", "M6d_fp32", "cls_tuned")}
        # boundary transfer: within 2x2 tiles of bright cells, share of the summed squared cell error that cancels
        m0 = (X["Tf"] >= BR) & (plane == P)
        key = (X["ev"][m0] * 6 + X["g"][m0]) * 1e7 + (X["fw"][m0] // 2) * 1e3 + X["ft"][m0] // 2
        for n in ("M6d_bf16", "M6dc_bf16"):
            e = X[n][m0] - X["Tf"][m0]
            u, inv = np.unique(key, return_inverse=True)
            s1 = np.bincount(inv, weights=e); s2 = np.bincount(inv, weights=e ** 2)
            grid[f"{PL[P]}_{n}_cancel_2x2"] = float(1 - (s1 ** 2).sum() / s2.sum())
    res["grid"] = grid
    # ---------------- 5. floor (events with floor.py output)
    evs = sorted(set(int(e) for e in X["ev"]))
    F = floor_maps(evs)
    flo = {}
    if F:
        fe = np.isin(X["ev"], list(F))
        keys_inputs = ("syn", "clean", "noisy", "corpus")
        invs = ("oracle", "tik0.0001", "tik0.001", "tik0.01")
        for P in range(3):
            cells, tiles = {}, {}
            for ev, z in F.items():
                for g in (P, P + 3):
                    T = z[f"truth_g{g}"]
                    mb = T >= BR
                    cells.setdefault("Tf", []).append(T[mb])
                    for k in keys_inputs:
                        for iv in invs:
                            cells.setdefault(f"{k}_{iv}", []).append(z[f"{k}_{iv}_g{g}"][mb])
                    for ta, tb, lab in ((2, 2, "2x2"), (4, 4, "4x4"), (16, 8, "16x8")):
                        nw_, nt_ = (T.shape[0] // ta) * ta, (T.shape[1] // tb) * tb
                        tl = lambda Y: Y[:nw_, :nt_].reshape(nw_ // ta, ta, nt_ // tb, tb).sum((1, 3)).ravel()
                        tt = tl(T); mt = tt >= 64e3
                        tiles.setdefault(f"{lab}_Tf", []).append(tt[mt])
                        for k in keys_inputs:
                            for iv in invs:
                                tiles.setdefault(f"{lab}_{k}_{iv}", []).append(tl(z[f"{k}_{iv}_g{g}"])[mt])
            Tc = np.concatenate(cells["Tf"])
            row = {f"cell_{k}": stats(rel(np.concatenate(v), Tc)) for k, v in cells.items() if k != "Tf"}
            for lab in ("2x2", "4x4", "16x8"):
                Tt = np.concatenate(tiles[f"{lab}_Tf"])
                for k in keys_inputs:
                    for iv in invs:
                        row[f"{lab}_{k}_{iv}"] = stats(rel(np.concatenate(tiles[f"{lab}_{k}_{iv}"]), Tt))
            # the models on the same floor events
            m0 = fe & (X["Tf"] >= BR) & (plane == P)
            for n in ("M6d_bf16", "M6dc_bf16", "M6d_fp32", "cls_tuned"):
                row[f"cell_model_{n}"] = stats(rel(X[n][m0], X["Tf"][m0]))
                for lev, lab in ((0, "2x2"), (1, "4x4"), (2, "16x8")):
                    mt = (Tl["level"] == lev) & (Tl["g"] % 3 == P) & (Tl["Tf"] >= 64e3)
                    row[f"{lab}_model_{n}_allevents"] = stats(rel(Tl[n][mt], Tl["Tf"][mt]))
            flo[PL[P]] = row
        res["floor_events"] = sorted(F)
    res["floor"] = flo
    json.dump(res, open(os.path.join(a.out, "bright_budget.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(a.out, "bright_budget.json"))


if __name__ == "__main__":
    main()
