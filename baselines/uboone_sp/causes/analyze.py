#!/usr/bin/env python3
"""Hypotheses F, B, C from the per-cell evidence of build.py (all 128 test events).

    scripts/helix_run.sh python baselines/uboone_sp/causes/analyze.py \
        --cells /global/cfs/cdirs/m5238/users/oalter/helix_runs/causes/cells --out baselines/uboone_sp/causes/results

Gated map = raw charge where presence > 0.5 (eval_denoise "_gated"); density = hits-truth
charge in the +-8-cell box, cell excluded; strata as dense/analyze.py (iso = 0, <=10k = (0, 1e4], ...). Weighted records (1 or 100).
"""
import argparse
import glob
import json
import os

import numpy as np

Q0 = 2043.258
DB = ((-1, 0.5, "iso"), (0.5, 1e4, "<=10k"), (1e4, 1e5, "10k-100k"), (1e5, 1e6, "100k-1M"), (1e6, 4e6, "1M-4M"), (4e6, 1e12, ">4M"))
AB = ((0, 3.0, "<3"), (3.0, 4.0, "3-4"), (4.0, 5.0, "4-5"), (5.0, 6.0, "5-6"), (6.0, 8.0, "6-8"), (8.0, 12.0, "8-12"), (12.0, 1e9, ">12"))


def load(d):
    cols = json.load(open(os.path.join(d, "columns.json")))["cols"]
    R, T, info = [], [], []
    for f in sorted(glob.glob(os.path.join(d, "ev*.npz"))):
        z = np.load(f)
        R.append(z["rec"]); T.append(z["tiles"]); info.append(json.loads(str(z["info"])))
    R = np.concatenate(R)
    X = {c: R[:, i].astype(np.float64) for i, c in enumerate(cols)}
    for t in ("M6d", "M6dc"):
        X[f"{t}_g"] = np.where(X[f"{t}_p"] > 0.5, X[f"{t}_q"], 0.0)
    return X, np.concatenate(T), info


def frac(m, w, sel):
    s = w[sel].sum()
    return float((w[sel & m]).sum() / s) if s > 0 else float("nan")


def hyp_F(X):
    """Dense haze: do the 'false' cells (Th = 0, pred > t) hold charge in the complete truth?"""
    w, Th, Tf, dq = X["w"], X["Th"], X["Tf"], X["dq"]
    emp = Th == 0
    out = {}
    for lo, hi, lab in DB:
        db = emp & (dq > lo) & (dq <= hi)
        row = dict(n_empty=float(w[db].sum()),
                   base_Tf_ge25=frac(Tf >= 25, w, db), base_Tf_ge100=frac(Tf >= 100, w, db))
        for m in ("M6d_g", "M6dc_g", "cls_q"):
            for t, tl in ((100.0, "100"), (Q0, "1Q0")):
                fs = db & (X[m] > t)
                if w[fs].sum() == 0:
                    continue
                P = X[m][fs]; F = Tf[fs]; ww = w[fs]
                # rescored against the complete truth: cells that are still empty, and the remaining false fraction
                cemp = db & (Tf < 25)
                row[f"{m}>{tl}"] = dict(
                    frac_false=frac(X[m] > t, w, db),
                    n=float(ww.sum()),
                    holds_Tf_gt0=float((ww * (F > 0)).sum() / ww.sum()),
                    holds_Tf_ge25=float((ww * (F >= 25)).sum() / ww.sum()),
                    holds_Tf_ge100=float((ww * (F >= 100)).sum() / ww.sum()),
                    holds_Tf_ge_half_pred=float((ww * (F >= 0.5 * P)).sum() / ww.sum()),
                    median_Tf=float(np.median(F)), median_pred=float(np.median(P)),
                    pred_explained=float((ww * np.minimum(P, F)).sum() / (ww * P).sum()),
                    frac_false_vs_complete=frac(X[m] > t, w, cemp))
        out[lab] = row
    return out


def tiles_lin(T):
    """32 x 128 regions: sum ratio pred / truth, binned by hits truth and by complete truth."""
    p, Th, Tf = T[:, 0], T[:, 1], T[:, 2]
    out = {}
    for ref, lab in ((Th, "hits"), (Tf, "complete")):
        for lo, hi in ((1e3, 4e3), (4e3, 16e3), (16e3, 64e3)):
            m = (ref > lo) & (ref <= hi)
            out[f"{lab}_{lo / 1e3:g}-{hi / 1e3:g}k"] = dict(
                n=int(m.sum()), **{nm: float(T[m, j].sum() / ref[m].sum()) for j, nm in ((3, "M6d_g"), (4, "M6dc_g"), (5, "cls_q"))},
                complete_over_hits=float(Tf[m].sum() / Th[m].sum()))
    for lab, m in (("empty_hits", Th == 0), ("empty_complete", Tf < 25)):
        out[lab] = dict(n=int(m.sum()), **{nm: float((T[m, j] > Q0).mean()) for j, nm in ((3, "M6d_g"), (4, "M6dc_g"), (5, "cls_q"))})
    return out


def hyp_B(X):
    """Sparse false charge: lone noise coefficient vs context prior vs truth incompleteness."""
    w, Th, Tf, dq = X["w"], X["Th"], X["Tf"], X["dq"]
    out = {}
    for lo, hi, lab in DB[:3]:
        sp = (dq > lo) & (dq <= hi)
        res = {}
        for m in ("M6d_g", "M6dc_g", "cls_q"):
            fs = sp & (Th == 0) & (X[m] > 100)
            ww = w[fs]
            if ww.sum() == 0:
                continue
            truthinc = Tf[fs] >= 25
            kept_own = X["Kmax"][fs] > 0
            kept_3 = X["Kmax3"][fs] > 0
            near = X["dist_h"][fs] <= 2
            cat = np.select([truthinc, kept_own, kept_3, near], ["truth_incomplete", "kept_coeff_in_cell",
                                                                 "kept_coeff_adjacent", "no_coeff_near_charge"], "no_coeff_far")
            dh, df, k3 = X["dist_h"][fs], X["dist_f"][fs], X["Kmax3"][fs]
            # no complete-truth charge (>= 25 e-) within 2 cells => a kept coefficient in the 3x3 cannot be the
            # deposit's own signal coefficient (response + band footprint <= 1 cell): a lone NOISE coefficient
            cat2 = np.select([truthinc, (df <= 2) & (k3 > 0), df <= 2, k3 > 0, df <= 8],
                             ["truth_incomplete", "spill_next_to_true_charge", "spill_next_to_true_charge",
                              "lone_noise_coefficient", "context_no_coefficient_near_activity"], "no_coefficient_far")
            sharecat = {c: dict(cells=float(ww[cat2 == c].sum() / ww.sum()),
                                charge=float((ww * X[m][fs])[cat2 == c].sum() / (ww * X[m][fs]).sum()))
                        for c in ("truth_incomplete", "spill_next_to_true_charge", "lone_noise_coefficient",
                                  "context_no_coefficient_near_activity", "no_coefficient_far")}
            lone_amp = k3[cat2 == "lone_noise_coefficient"]
            res[m + "_causal"] = dict(sharecat, lone_amp_median=float(np.median(lone_amp)) if len(lone_amp) else None,
                                      spill_with_coeff_3x3=float(ww[(cat2 == "spill_next_to_true_charge") & (k3 > 0)].sum() / max(ww[cat2 == "spill_next_to_true_charge"].sum(), 1e-9)))
            res[m] = dict(n=float(ww.sum()), frac_false=frac(X[m] > 100, w, sp & (Th == 0)),
                          **{c: float(ww[cat == c].sum() / ww.sum()) for c in
                             ("truth_incomplete", "kept_coeff_in_cell", "kept_coeff_adjacent", "no_coeff_near_charge", "no_coeff_far")},
                          median_Kmax3=float(np.median(X["Kmax3"][fs])), median_dist_h=float(np.median(X["dist_h"][fs])),
                          frac_dist_le1=float(ww[X["dist_h"][fs] <= 1].sum() / ww.sum()),
                          frac_dist_le4=float(ww[X["dist_h"][fs] <= 4].sum() / ww.sum()))
        # true-positive faint cells for comparison
        tp = sp & (Th > 0) & (Th < Q0) & (X["M6d_g"] > 100)
        res["truepos_faint_M6d"] = dict(n=float(w[tp].sum()), median_Kmax3=float(np.median(X["Kmax3"][tp])) if tp.any() else None,
                                        frac_kept_in_cell=frac(X["Kmax"] > 0, w, tp), frac_kept_3=frac(X["Kmax3"] > 0, w, tp))
        # response to a lone coefficient: truly empty cells (Th = 0, Tf < 25, no charge within 2 cells) holding a
        # kept coefficient of amplitude a, vs faint charged cells with the same amplitude
        lone = sp & (Th == 0) & (Tf < 25) & (X["dist_f"] >= 3)
        chg = sp & (Th > 0) & (Th < Q0)
        cal = {}
        for alo, ahi, al in AB:
            a_ = (X["Kmax"] > alo) & (X["Kmax"] <= ahi)
            n_e, n_c = w[lone & a_].sum(), w[chg & a_].sum()
            if n_e + n_c < 20:
                continue
            cal[al] = dict(n_lone=float(n_e), n_faint_charged=float(n_c),
                           lone_pred_frac_M6d=frac(X["M6d_g"] > 100, w, lone & a_), lone_mean_p_M6d=float(np.average(X["M6d_p"][lone & a_], weights=w[lone & a_])) if n_e else None,
                           charged_pred_frac_M6d=frac(X["M6d_g"] > 100, w, chg & a_),
                           charged_mean_p_M6d=float(np.average(X["M6d_p"][chg & a_], weights=w[chg & a_])) if n_c else None,
                           lone_pred_frac_cls=frac(X["cls_q"] > 100, w, lone & a_),
                           posterior_charged=float(n_c / (n_c + n_e)))
        res["lone_coefficient_calibration"] = cal
        # context prior: empty cells with NO kept coefficient within 3x3, by distance to charge
        ctx = {}
        noc = sp & (Th == 0) & (Tf < 25) & (X["Kmax3"] == 0)
        for d0, d1 in ((1, 1), (2, 2), (3, 4), (5, 8), (9, 999)):
            m_ = noc & (X["dist_h"] >= d0) & (X["dist_h"] <= d1)
            ctx[f"{d0}-{d1}"] = dict(n=float(w[m_].sum()), M6d_pred_frac=frac(X["M6d_g"] > 100, w, m_),
                                     M6d_mean_p=float(np.average(X["M6d_p"][m_], weights=w[m_])) if w[m_].sum() else None,
                                     cls_pred_frac=frac(X["cls_q"] > 100, w, m_))
        res["context_no_coeff"] = ctx
        out[lab] = res
    return out


def hyp_C(X):
    """Gating zeroes true charge: unrecoverable input vs presence mis-calibration vs threshold."""
    w, Th, dq = X["w"], X["Th"], X["dq"]
    out = {}
    chg = Th > 0
    total = (w * Th)[chg].sum()
    for m in ("M6d", "M6dc"):
        z = chg & (X[f"{m}_p"] <= 0.5)
        res = dict(zeroed_charge_frac=float((w * Th)[z].sum() / total))
        cats = np.select([X["cov"] == 0, X["Kmax3"] == 0, X["Kmax"] == 0, X[f"{m}_p"] > 0.3],
                         ["uncovered", "no_kept_coeff_3x3", "kept_adjacent_only", "kept_in_cell_p_0.3-0.5"], "kept_in_cell_p<=0.3")
        sub = np.select([X["Smax3"] < 1.0, X["Smax3"] < 3.35], ["clean_signal<1sigma", "clean_signal_1-3.35sigma"], "clean_signal>=3.35sigma")
        for lo, hi, lab in (((-1, 1e12, "all"),) + DB):
            db = z & (dq > lo) & (dq <= hi)
            q = (w * Th)[db]
            if q.sum() == 0:
                continue
            row = dict(zeroed_frac_of_bin_charge=float(q.sum() / (w * Th)[chg & (dq > lo) & (dq <= hi)].sum()))
            for c in ("uncovered", "no_kept_coeff_3x3", "kept_adjacent_only", "kept_in_cell_p_0.3-0.5", "kept_in_cell_p<=0.3"):
                row[c] = float(q[cats[db] == c].sum() / q.sum())
            for c in ("clean_signal<1sigma", "clean_signal_1-3.35sigma", "clean_signal>=3.35sigma"):
                row[c] = float(q[sub[db] == c].sum() / q.sum())
            # input-limited share: no kept coefficient within 3x3 (or uncovered)
            row["input_absent"] = row["uncovered"] + row["no_kept_coeff_3x3"]
            ia = np.isin(cats[db], ["uncovered", "no_kept_coeff_3x3"])
            for c in ("clean_signal<1sigma", "clean_signal_1-3.35sigma", "clean_signal>=3.35sigma"):
                row["input_absent&" + c] = float(q[ia & (sub[db] == c)].sum() / q.sum())
            kc = np.isin(cats[db], ["kept_in_cell_p_0.3-0.5", "kept_in_cell_p<=0.3"])
            km = X["Kmax"][db]
            row["kept_in_cell_Kmax_lt5"] = float(q[kc & (km < 5)].sum() / q.sum())
            row["kept_in_cell_Kmax_ge5"] = float(q[kc & (km >= 5)].sum() / q.sum())
            row["median_cell_charge"] = float(np.median(Th[db]))
            res[lab] = row
        # by cell charge and plane, 1-2k e-
        for P in range(3):
            sel = chg & (X["plane"] == P) & (Th > 1000) & (Th <= 2000)
            zz = sel & (X[f"{m}_p"] <= 0.5)
            res[f"plane{P}_1-2k"] = dict(zeroed_frac=frac(X[f"{m}_p"] <= 0.5, w, sel),
                                         of_which_input_absent=frac((X["cov"] == 0) | (X["Kmax3"] == 0), w, zz),
                                         of_which_clean_below_1sigma=frac(X["Smax3"] < 1.0, w, zz))
        # reliability: presence bins vs observed charged fraction, per density
        rel = {}
        for lo, hi, lab in DB:
            db = (dq > lo) & (dq <= hi)
            rows = []
            for a in np.linspace(0, 0.9, 10):
                b_ = db & (X[f"{m}_p"] > a) & (X[f"{m}_p"] <= a + 0.1)
                if w[b_].sum() < 50:
                    continue
                rows.append([round(a + 0.05, 2), float(np.average(X[f"{m}_p"][b_], weights=w[b_])), frac(chg, w, b_), float(w[b_].sum())])
            rel[lab] = rows
        res["reliability"] = rel
        # threshold curve: charge zeroed and empty cells admitted vs presence threshold
        curve = []
        emp_all = ~chg
        for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            zc = float((w * Th)[chg & (X[f"{m}_p"] <= t)].sum() / total)
            fa = float(w[emp_all & (X[f"{m}_p"] > t) & (X[f"{m}_q"] > 100)].sum() / w[emp_all].sum())
            curve.append(dict(t=t, charge_zeroed=zc, empty_cells_gt100=fa))
        res["threshold_curve"] = curve
        out[m] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X, T, info = load(a.cells)
    chk = [v for inf in info for k, v in inf.items() if k.startswith("g")]
    ttot = sum(sum(v for k, v in inf.items() if k.endswith("_q_total")) for inf in info)
    res = dict(n_events=len(info),
               machinery=dict(thr_vs_hits_rel_l2_median=float(np.median([c["thr_vs_hits_rel_l2"] for c in chk])),
                              thr_vs_hits_rel_l2_p95=float(np.percentile([c["thr_vs_hits_rel_l2"] for c in chk], 95)),
                              complete_over_hits=float(sum(c["Tf"] for c in chk) / sum(c["Th"] for c in chk)),
                              thr_over_hits=float(sum(c["Tthr"] for c in chk) / sum(c["Th"] for c in chk)),
                              deposit_charge_unplaced_frac=float(sum(sum(v for k, v in inf.items() if k.endswith("_unplaced")) for inf in info) / ttot),
                              deposit_charge_track_t0_frac=float(sum(sum(v for k, v in inf.items() if k.endswith("_track_t0")) for inf in info) / ttot)),
               F=hyp_F(X), F_regions=tiles_lin(T), B=hyp_B(X), C=hyp_C(X))
    json.dump(res, open(os.path.join(a.out, "causes_FBC.json"), "w"), indent=1, default=float)
    print(json.dumps(res["machinery"], indent=1))


if __name__ == "__main__":
    main()
