#!/usr/bin/env python3
"""Deployable post-hoc checks for m7_diagnostics.md, every parameter fitted on CALIBRATION events only and
scored on the test events (writes m7_diagnostics_post.json).

    scripts/helix_run.sh python baselines/uboone_sp/causes/m7diag_post.py \
        --out baselines/uboone_sp/causes/results/m7_diagnostics_post.json

  phase   per-plane multiplicative correction of calo by the cell's position inside its 16-wire token
          (fw mod 8), from the median of truth/calo over bright predicted cells (calo >= 16 k) on cal events
  blend   per plane and predicted-charge bin, calo_b = w * calo + (1 - w) * tik, tik = Fourier Tikhonov
          inversion of the CORPUS input (floor.py --inputs corpus); lambda and w chosen on the calibration
          events (m7diag/cal_floor) by the hw68 of truth/pred over bright cells; scored on test floor events
          ev260-307 (bright/floor). A diagnostic of complementary information, not a proposed product.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m7diag_analyze import load, rel, stats, M7, C3, PL, BR  # noqa: E402

LAMS = ("tik0.0001", "tik0.001", "tik0.01")
QB = np.array([16e3, 64e3, 256e3, np.inf]); QBL = ["16-64k", "64-256k", ">=256k"]


def attach_floor(X, fdir, evs):
    """Floor maps (corpus input, every lambda) at the record cells with Tf >= 8 k e- -> dict lam -> array."""
    sel = np.flatnonzero((X["Tf"] >= 8000) & np.isin(X["ev"], evs))
    out = {l: np.full(len(sel), np.nan) for l in LAMS}
    tt = np.full(len(sel), np.nan)
    for ev in evs:
        f = os.path.join(fdir, f"ev{ev:03d}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        for g in range(6):
            k = (X["ev"][sel] == ev) & (X["g"][sel] == g)
            if not k.any() or f"truth_g{g}" not in z.files:
                continue
            T = z[f"truth_g{g}"]
            fw, ft = X["fw"][sel][k].astype(int), X["ft"][sel][k].astype(int)
            inb = (fw < T.shape[0]) & (ft < T.shape[1])
            idx = np.flatnonzero(k)[inb]
            tt[idx] = T[fw[inb], ft[inb]]
            for l in LAMS:
                out[l][idx] = z[f"corpus_{l}_g{g}"][fw[inb], ft[inb]]
    return sel, out, tt


def phase(X, Xc, mp="M7dc_calo"):
    res = {}
    for P in range(3):
        sc = (Xc["g"] % 3 == P) & (Xc[mp] >= BR) & (Xc["Tf"] > 0)
        fac = np.array([np.median(Xc["Tf"][sc & (Xc["fw"] % 8 == j)] / Xc[mp][sc & (Xc["fw"] % 8 == j)]) for j in range(8)])
        st = (X["g"] % 3 == P)
        corr = np.where(st & (X[mp] >= BR), X[mp] * fac[(X["fw"] % 8).astype(int)], X[mp])
        m0 = st & (X["Tf"] >= BR)
        r0, r1 = rel(X[mp][m0], X["Tf"][m0]), rel(corr[m0], X["Tf"][m0])
        res[PL[P]] = dict(factors_cal=fac.round(5).tolist(), before=stats(r0), after=stats(r1),
                          var_share_removed=float(1 - r1.var() / r0.var()))
    return res


def blend(X, Xc, mp="M7dc_calo"):
    res = {}
    cal_evs = sorted(set(int(e) for e in Xc["ev"]))
    sc, Fc, tc = attach_floor(Xc, os.path.join(M7, "cal_floor"), cal_evs)
    st, Ft, tt = attach_floor(X, os.path.join(C3, "bright", "floor"), list(range(260, 308)))
    res["n_cal_events_with_floor"] = int(sum(os.path.exists(os.path.join(M7, "cal_floor", f"ev{e:03d}.npz")) for e in cal_evs))
    res["cal_truth_match_maxrel"] = float(np.nanmax(np.abs(tc - Xc["Tf"][sc]) / Xc["Tf"][sc]))
    for P in range(3):
        row = {}
        mc = (Xc["g"][sc] % 3 == P) & (Xc["Tf"][sc] >= BR) & np.isfinite(Fc[LAMS[0]])
        mt = (X["g"][st] % 3 == P) & (X["Tf"][st] >= BR) & np.isfinite(Ft[LAMS[0]])
        Tc_, Tt_ = Xc["Tf"][sc][mc].astype(np.float64), X["Tf"][st][mt].astype(np.float64)
        Cc_, Ct_ = Xc[mp][sc][mc].astype(np.float64), X[mp][st][mt].astype(np.float64)
        lam = min(LAMS, key=lambda l: stats(rel(Fc[l][mc], Tc_))["hw68"])
        Lc, Lt = Fc[lam][mc], Ft[lam][mt]
        row["lambda_cal"] = lam
        row["cal_floor_hw68"] = stats(rel(Lc, Tc_))["hw68"]
        bc = np.clip(np.searchsorted(QB, Cc_, side="right") - 1, 0, 2)
        bt = np.clip(np.searchsorted(QB, Ct_, side="right") - 1, 0, 2)
        ws = []
        for i in range(3):
            k = bc == i
            if k.sum() < 50:
                ws.append(1.0); continue
            ws.append(float(min(np.round(np.arange(0, 1.01, 0.05), 2), key=lambda w: stats(rel(w * Cc_[k] + (1 - w) * Lc[k], Tc_[k]))["hw68"])))
        ws = np.array(ws)
        row["w_model_by_pred_bin_cal"] = dict(zip(QBL, ws.tolist()))
        Bt = ws[bt] * Ct_ + (1 - ws[bt]) * Lt
        row["test_model"] = stats(rel(Ct_, Tt_)); row["test_floor"] = stats(rel(Lt, Tt_)); row["test_blend"] = stats(rel(Bt, Tt_))
        row["test_by_truth_bin"] = []
        for i in range(3):
            k = (Tt_ >= QB[i]) & (Tt_ < QB[i + 1])
            row["test_by_truth_bin"].append(dict(bin=QBL[i], n=int(k.sum()), model=stats(rel(Ct_[k], Tt_[k]))["hw68"],
                                                 floor=stats(rel(Lt[k], Tt_[k]))["hw68"], blend=stats(rel(Bt[k], Tt_[k]))["hw68"],
                                                 blend_median=stats(rel(Bt[k], Tt_[k]))["median"]))
        res[PL[P]] = row
    return res


def topshare(X, mp="M7dc_calo"):
    """Share of bright cells and of the bright-cell MSE of r carried by Tf >= 1 M e- (and by saturated cells)."""
    out = {}
    for P in range(3):
        m0 = (X["g"] % 3 == P) & (X["Tf"] >= BR)
        r = rel(X[mp][m0], X["Tf"][m0]); top = X["Tf"][m0] >= 1e6; sat = X["sat"][m0] > 0
        out[PL[P]] = dict(frac_cells_ge1M=float(top.mean()), mse_share_ge1M=float((r[top] ** 2).sum() / (r ** 2).sum()),
                          frac_cells_sat=float(sat.mean()), mse_share_sat=float((r[sat] ** 2).sum() / (r ** 2).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X, _, _, labs = load(os.path.join(M7, "cells"))
    Xc, _, _, _ = load(os.path.join(M7, "cal_cells"))
    res = dict(phase=phase(X, Xc), phase_M6dc=phase(X, Xc, "M6dc_calo"), topshare=topshare(X))
    print("phase", json.dumps(res["phase"])[:2000], flush=True)
    res["blend"] = blend(X, Xc)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
