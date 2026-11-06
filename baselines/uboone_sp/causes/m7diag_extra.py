#!/usr/bin/env python3
"""Follow-up checks for m7_diagnostics.md (run after m7diag_analyze.py; writes m7_diagnostics_extra.json).

    scripts/helix_run.sh python baselines/uboone_sp/causes/m7diag_extra.py \
        --out baselines/uboone_sp/causes/results/m7_diagnostics_extra.json

  lags      error correlation of bright cells along wire / time out to 24 cells (token period = 8 cells
            along the wire: a token spans 16 wires) -- M7dc calo, M6dc calo
  cancel    bright_analyze's 2 x 2 'cancellation' 1 - sum(sum e)^2 / sum e^2 on absolute errors, as in
            bright_budget.md, raw and with the plane's median relative bias removed (e - med(r) * Tf)
  presence  Q2 with an exact, binning-robust split: T/P = [sum 1[ch] chat / sum p_cal chat] x
            [sum_ch Tf / sum_ch chat] x [1 + sub-25 charge / sum_ch Tf]; plus the reliability of p_cal
            inside the faint calo bins (test and calibration events)
  column    dE/dx-like per (track, 2-wire column) sums of single-particle cells, truth vs calo
  floorcomb Y-plane model vs Tikhonov floor on the floor events: error correlation and the hw68 of their
            average (complementary information test)
  labels    complete truth of the records vs the corpus complete truth the products were fitted with
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m7diag_analyze import load, rel, stats, ratio_stats, M7, C3, PL, BR, Q0  # noqa: E402


def lags(X):
    res = {}
    m0 = (X["Tf"] >= BR) & (X["sat"] == 0)
    idx = np.flatnonzero(m0)
    key = (((X["ev"][idx].astype(np.int64) * 6 + X["g"][idx].astype(np.int64)) * 1024 + X["fw"][idx].astype(np.int64)) * 512
           + X["ft"][idx].astype(np.int64))
    o = np.argsort(key); key, idx = key[o], idx[o]
    plane = X["g"] % 3
    for mp in ("M7dc_calo", "M6dc_calo"):
        r = rel(X[mp], X["Tf"])
        for name, step in (("wire", 512), ("time", 1)):
            for P in range(3):
                cs = []
                for d in range(1, 25):
                    k2 = key + d * step
                    pos = np.clip(np.searchsorted(key, k2), 0, len(key) - 1)
                    ok = key[pos] == k2
                    a, b = idx[ok], idx[pos[ok]]
                    k = plane[a] == P
                    a, b = a[k], b[k]
                    cs.append(round(float(np.corrcoef(r[a], r[b])[0, 1]), 4) if len(a) > 100 else None)
                res[f"{mp}_{name}_{PL[P]}"] = cs
        # token phase: lag-8 wire correlation for cells at each position inside the 8-cell token period
        for P in range(3):
            k2 = key + 8 * 512
            pos = np.clip(np.searchsorted(key, k2), 0, len(key) - 1)
            ok = key[pos] == k2
            a, b = idx[ok], idx[pos[ok]]
            k = plane[a] == P
            a, b = a[k], b[k]
            ph = (X["fw"][a].astype(int)) % 8
            res[f"{mp}_wire8_by_phase_{PL[P]}"] = [round(float(np.corrcoef(r[a][ph == j], r[b][ph == j])[0, 1]), 4) for j in range(8)]
        # mean relative error by position inside the token (fw mod 8) and per 8-cell time block (ft mod 8)
        for P in range(3):
            k = m0 & (plane == P)
            res[f"{mp}_median_r_by_fw_mod8_{PL[P]}"] = [round(float(np.median(r[k & (X['fw'] % 8 == j)])), 5) for j in range(8)]
            res[f"{mp}_hw68_by_fw_mod8_{PL[P]}"] = [round(float(stats(r[k & (X['fw'] % 8 == j)])["hw68"]), 5) for j in range(8)]
            res[f"{mp}_median_r_by_ft_mod8_{PL[P]}"] = [round(float(np.median(r[k & (X['ft'] % 8 == j)])), 5) for j in range(8)]
    return res


def cancel(X):
    res = {}
    plane = X["g"] % 3
    for P in range(3):
        m0 = (X["Tf"] >= BR) & (plane == P)
        key = (X["ev"][m0].astype(np.int64) * 6 + X["g"][m0].astype(np.int64)) * 10 ** 7 + (X["fw"][m0].astype(np.int64) // 2) * 1000 + X["ft"][m0].astype(np.int64) // 2
        u, inv = np.unique(key, return_inverse=True)
        for mp in ("M7dc_calo", "M6dc_calo", "M7d_calo", "M6d_calo", "M6dc_gated", "M6d_gated"):
            T = X["Tf"][m0].astype(np.float64); C = X[mp][m0].astype(np.float64)
            out = {}
            for lab, e in (("raw", C - T), ("debiased", C - T * (1 + np.median(C / T - 1)))):
                s1 = np.bincount(inv, weights=e); s2 = np.bincount(inv, weights=e ** 2)
                out[lab] = float(1 - (s1 ** 2).sum() / s2.sum())
            res[f"{PL[P]}_{mp}"] = out
    return res


def presence(X, labs):
    res = {}
    plane = X["g"] % 3
    for l in labs:
        if l.endswith("32"):
            continue
        for lo, hi in ((102, 500), (500, 1000)):
            for P in list(range(3)) + ["all"]:
                m = (X[f"{l}_calo"] >= lo) & (X[f"{l}_calo"] < hi)
                if P != "all":
                    m &= plane == P
                calo = X[f"{l}_calo"][m].astype(np.float64); pc = X[f"{l}_pcal"][m].astype(np.float64)
                Tf = X["Tf"][m].astype(np.float64); p = X[f"{l}_p"][m].astype(np.float64)
                chat = calo / np.maximum(pc, 1e-12)
                ch = Tf > 25
                row = dict(n=int(m.sum()), truth_over_pred=float(Tf.sum() / calo.sum()),
                           presence_w=float((ch * chat).sum() / (pc * chat).sum()),
                           cond_on_charged=float(Tf[ch].sum() / chat[ch].sum()),
                           sub25=float(1 + Tf[~ch].sum() / Tf[ch].sum()))
                row["identity"] = row["presence_w"] * row["cond_on_charged"] * row["sub25"] / row["truth_over_pred"]
                # reliability of p_cal inside the bin
                e = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
                b = np.clip(np.searchsorted(e, pc, side="right") - 1, 0, 9)
                row["reliability"] = [dict(bin=f"{e[i]:.1f}-{min(e[i + 1], 1):.1f}", n=int((b == i).sum()),
                                           mean_pcal=float(pc[b == i].mean()) if (b == i).any() else None,
                                           frac_charged=float(ch[b == i].mean()) if (b == i).any() else None) for i in range(10)]
                # what the raw (uncorrected) presence would give: odds-ratio check
                row["mean_p_raw"] = float(p.mean()); row["mean_pcal"] = float(pc.mean()); row["frac_charged"] = float(ch.mean())
                res[f"{l}_{lo}-{hi}_{PL[P] if P != 'all' else 'all'}"] = row
    return res


def column(X, Pa):
    """Per (track, 2-wire column): sums over the column's single-particle cells (maj_share >= 0.9)."""
    res = {}
    key_p = {(int(e), int(g), int(t)): i for i, (e, g, t) in enumerate(zip(Pa["ev"], Pa["g"], Pa["trk_row"]))}
    sel = (X["maj_share"] >= 0.9) & (X["maj_trk"] >= 0) & (X["Tf"] > 0)
    ii = np.flatnonzero(sel)
    pi = np.array([key_p.get((int(e), int(g), int(t)), -1) for e, g, t in zip(X["ev"][ii], X["g"][ii], X["maj_trk"][ii])])
    ok = pi >= 0
    ii, pi = ii[ok], pi[ok]
    mip = ((Pa["dedx_med"][pi] >= 1.5) & (Pa["dedx_med"][pi] <= 2.5) & (Pa["bragg_frac"][pi] < 0.05) & (Pa["length_mm"][pi] >= 100)
           & (Pa["th_drift"][pi] < 60) & (Pa["sat_frac"][pi] == 0) & (Pa["overlap"][pi] < 0.2))
    ii, pi = ii[mip], pi[mip]
    ck = pi.astype(np.int64) * 4096 + X["fw"][ii].astype(np.int64)
    u, inv = np.unique(ck, return_inverse=True)
    trk = u // 4096
    for l, k in (("M7dc", "calo"), ("M7d", "calo"), ("M6dc", "calo"), ("M6d", "calo"), ("M7dc", "q"), ("M7dc32", "calo")):
        if f"{l}_{k}" not in X:
            continue
        Tc = np.bincount(inv, weights=X["Tf"][ii].astype(np.float64))
        Cc = np.bincount(inv, weights=X[f"{l}_{k}"][ii].astype(np.float64))
        ut, ti, nt = np.unique(trk, return_inverse=True, return_counts=True)
        keep = (nt[ti] >= 10) & (Tc >= 5000)
        Tc_, Cc_, ti_ = Tc[keep], Cc[keep], ti[keep]
        # drop each track's two end columns (partial path): keep columns strictly inside the track's fw range
        colfw = (u % 4096)[keep]
        lo = np.full(len(ut), 10 ** 9); hi = np.full(len(ut), -1)
        np.minimum.at(lo, ti_, colfw); np.maximum.at(hi, ti_, colfw)
        inner = (colfw > lo[ti_]) & (colfw < hi[ti_])
        Tc_, Cc_, ti_ = Tc_[inner], Cc_[inner], ti_[inner]
        med = np.zeros(len(ut))
        o = np.lexsort((Tc_, ti_)); gs = ti_[o]; vs = Tc_[o]
        st = np.searchsorted(gs, np.arange(len(ut))); en = np.searchsorted(gs, np.arange(len(ut)), side="right")
        for j, (a, b) in enumerate(zip(st, en)):
            med[j] = np.median(vs[a:b]) if b > a else np.nan
        xT, xC = Tc_ / med[ti_], Cc_ / med[ti_]
        pl = (Pa["g"][ut[ti_]] % 3).astype(int)
        for P in range(3):
            m = pl == P
            if m.sum() < 100:
                continue
            q = lambda x: dict(zip(["p1", "p5", "p16", "p50", "p84", "p95", "p99"], np.round(np.percentile(x, [1, 5, 16, 50, 84, 95, 99]), 4).tolist()))
            hw = lambda x: float((np.percentile(x, 84) - np.percentile(x, 16)) / 2)
            hT, hC = hw(xT[m]), hw(xC[m])
            # 60% truncated mean per track (lowest 60% of columns), truth vs pred
            tmr = []
            for j in np.unique(ti_[m]):
                mm = ti_ == j
                a_, b_ = np.sort(Tc_[mm]), np.sort(Cc_[mm])
                n6 = max(1, int(round(0.6 * len(a_))))
                tmr.append(b_[:n6].mean() / a_[:n6].mean())
            tmr = np.array(tmr)
            res[f"{l}_{k}_{PL[P]}"] = dict(n_columns=int(m.sum()), n_tracks=int(len(np.unique(ti_[m]))),
                                          truth=dict(hw68=hT, **q(xT[m])), pred=dict(hw68=hC, **q(xC[m])),
                                          hw68_ratio=hC / hT, narrower_in_quadrature=float(np.sqrt(max(hT ** 2 - hC ** 2, 0))),
                                          col_ratio=ratio_stats(Cc_[m] / Tc_[m]),
                                          trunc60_ratio=dict(median=float(np.median(tmr)), hw68=hw(tmr)))
    return res


def floorcomb(X):
    res = {}
    F = {}
    for ev in range(260, 308):
        f = os.path.join(C3, "bright", "floor", f"ev{ev:03d}.npz")
        if os.path.exists(f):
            F[ev] = np.load(f)
    plane = X["g"] % 3
    for P in range(3):
        m0 = (X["Tf"] >= BR) & (plane == P) & np.isin(X["ev"], list(F))
        ii = np.flatnonzero(m0)
        lam = {0: "tik0.0001", 1: "tik0.001", 2: "tik0.001"}[P]
        for inp in ("corpus", "noisy"):
            fl = np.zeros(len(ii)); tt = np.zeros(len(ii))
            for ev, z in F.items():
                for g in (P, P + 3):
                    k = (X["ev"][ii] == ev) & (X["g"][ii] == g)
                    if not k.any():
                        continue
                    M = z[f"{inp}_{lam}_g{g}"]; T = z[f"truth_g{g}"]
                    fw, ft = X["fw"][ii][k].astype(int), X["ft"][ii][k].astype(int)
                    inb = (fw < M.shape[0]) & (ft < M.shape[1])
                    v = np.full(k.sum(), np.nan); t = np.full(k.sum(), np.nan)
                    v[inb] = M[fw[inb], ft[inb]]; t[inb] = T[fw[inb], ft[inb]]
                    fl[k] = v; tt[k] = t
            ok = np.isfinite(fl)
            T = X["Tf"][ii][ok].astype(np.float64)
            row = dict(n=int(ok.sum()), truth_match_maxrel=float(np.nanmax(np.abs(tt[ok] - T) / T)), floor=stats(rel(fl[ok], T)))
            for mp in ("M7dc_calo", "M7dc32_calo", "M6dc_calo", "M7d_calo"):
                C = X[mp][ii][ok].astype(np.float64)
                rm, rf = rel(C, T), rel(fl[ok], T)
                row[mp] = dict(model=stats(rm), corr_err_model_floor=float(np.corrcoef(rm, rf)[0, 1]),
                               average=stats(rel(0.5 * (C + fl[ok]), T)))
                # best fixed-weight blend (diagnostic only; weights on a 0.1 grid, on these test events: an ORACLE)
                best = min(((w, stats(rel(w * C + (1 - w) * fl[ok], T))["hw68"]) for w in np.round(np.arange(0, 1.01, 0.1), 1)), key=lambda x: x[1])
                row[mp]["oracle_blend_w_model"] = float(best[0]); row[mp]["oracle_blend_hw68"] = float(best[1])
                # by charge
                row[mp]["by_charge"] = []
                for lo, hi in ((16e3, 64e3), (64e3, 256e3), (256e3, 1e9)):
                    k = (T >= lo) & (T < hi)
                    row[mp]["by_charge"].append(dict(lo=lo, hi=hi, n=int(k.sum()), model_hw68=stats(rm[k])["hw68"], floor_hw68=stats(rf[k])["hw68"]))
            res[f"{PL[P]}_{inp}_{lam}"] = row
    return res


def labels(X):
    """Records' complete truth vs the corpus complete truth (denoise_truth_complete) on two test events."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "scripts"))
    import denoise_products as DP
    out = {}
    shapes = DP._shapes()
    for ev in (260, 300):
        shard, event, pos = DP._event_inputs(ev, None)
        ck, cq = DP._corpus_truth(DP.COMPLETE, shard, event)
        Tc = DP._maps(ck, cq, shapes)
        m = X["ev"] == ev
        d = []
        lab_agree = []
        for g in range(6):
            k = m & (X["g"] == g)
            fw, ft = X["fw"][k].astype(int), X["ft"][k].astype(int)
            sh = Tc[g].shape
            inb = (fw < sh[0]) & (ft < sh[1])
            a = Tc[g][fw[inb], ft[inb]]; b = X["Tf"][k][inb].astype(np.float64)
            big = b > 1000
            d.append(np.median(np.abs(a[big] - b[big]) / b[big]))
            lab_agree.append(float(((a > 25) == (b > 25)).mean()))
        out[f"ev{ev}"] = dict(median_rel_diff_Tf_gt1k=[float(x) for x in d], label_agreement_q_gt25=lab_agree)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X, Tl, Pa, labs = load(os.path.join(M7, "cells"))
    Xc, _, _, clabs = load(os.path.join(M7, "cal_cells"))
    res = {}
    res["labels"] = labels(X); print("labels", res["labels"], flush=True)
    res["lags"] = lags(X); print("lags", flush=True)
    res["cancel"] = cancel(X); print("cancel", flush=True)
    res["presence_test"] = presence(X, labs); res["presence_cal"] = presence(Xc, clabs); print("presence", flush=True)
    res["column"] = column(X, Pa); print("column", flush=True)
    res["floorcomb"] = floorcomb(X); print("floorcomb", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
