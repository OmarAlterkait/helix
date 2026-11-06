#!/usr/bin/env python3
"""Causes of the denoiser's failings, hypotheses A (faint isolated deposits), D
(induction vs collection) and E (decoder window gain not seen per cell).

    scripts/helix_run.sh python baselines/uboone_sp/dense/causes_ade.py \
        --out baselines/uboone_sp/dense/results/causes_ade.json

Window scores (max over the window's 2x16 cells, exactly eval_denoise / v2/score.py):
  cls      classical tuned best statistic (tuned_any: U zlf06, V zor06a2, Y q_s0.67_f0),
           classical noise draw (seed 1000+ev), full maps from dense/cls_maps.py
  M6d      M6d presence on the stored corpus (dense3/model; reproduces eval windows exactly)
  M6dcn    M6d presence on the corpus RE-BUILT from the classical noise draw (causes2/model_cn)
  M4n      M4n presence (dense2/model), M6dc presence (dense3/model)
Thresholds: 1% per plane on the near-activity charge-free windows (bgn), the
benchmark's "near pp"; far = on the floor's bg windows.
Per-deposit evidence (dense/snr_deposits.py): matched-filter statistic on the
classical input (gate), on the same realisation thresholded like the corpus (k1),
and on the stored corpus (corp), all in units of the gated-noise sigma (the
statistic's null std on the unthresholded input); known-location decisions use
each input's own pooled per-plane null (99th percentile).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, os.path.dirname(os.path.dirname(BASE)))
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"
ANNOT = "/pscratch/sd/o/oalter/helix_work/denoise/noise_vs_hits.npz"
CFS = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
CLS = (f"{CFS}/dense2/cls_fix", "/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/cls_fit")
EVS = list(range(260, 388))
SNRB = [(0, 1e-9, "nosig"), (1e-9, 3, "0-3"), (3, 5, "3-5"), (5, 7, "5-7"), (7, 10, "7-10"), (10, 15, "10-15"), (15, 1e9, ">15")]
ZB = [(-1e9, 2, "<2"), (2, 3, "2-3"), (3, 4, "3-4"), (4, 5, "4-5"), (5, 6, "5-6"), (6, 8, "6-8"), (8, 12, "8-12"), (12, 1e9, ">12")]
B = 200


def load_windows():
    from helix.probe.resolution import unkey
    from sp.pipeline import lookup
    A = np.load(ANNOT)
    ann = {k: (float(s), float(pk), float(fo), float(q)) for k, s, pk, fo, q in
           zip(zip(A["ev"].tolist(), A["g"].tolist(), A["w0"].tolist(), A["t0"].tolist()), A["snr"], A["peak"], A["foreign"], A["q"])}
    TN = os.path.exists(f"{CFS}/causes2/model_tn/ev387.npz") and os.path.exists(f"{CFS}/causes2/cls_tn/ev387_g5.npz")
    rows, S, CELLS = [], {k: [] for k in ("cls", "M6d", "M6dcn", "M4n", "M6dc") + (("M6dtn", "clstn") if TN else ())}, []
    for ev in EVS:
        z = np.load(os.path.join(TRUTH, f"ev{ev:03d}.npz"), allow_pickle=True)
        zn = np.load(os.path.join(NEAR, f"ev{ev:03d}.npz"), allow_pickle=True)
        F = {g: np.load(next(os.path.join(d, f"ev{ev:03d}_g{g}.npz") for d in CLS if os.path.exists(os.path.join(d, f"ev{ev:03d}_g{g}.npz"))))
             for g in range(6)}
        M3 = np.load(f"{CFS}/dense3/model/ev{ev:03d}.npz"); M2 = np.load(f"{CFS}/dense2/model/ev{ev:03d}.npz")
        MC = np.load(f"{CFS}/causes2/model_cn/ev{ev:03d}.npz")
        if TN:
            MT = np.load(f"{CFS}/causes2/model_tn/ev{ev:03d}.npz")
            FT_ = {g: np.load(f"{CFS}/causes2/cls_tn/ev{ev:03d}_g{g}.npz")["tuned_any"] for g in range(6)}
        T = {g: F[g]["truth"] for g in range(6)}
        for src in (z, zn):
            mm = json.loads(str(src["meta"])); keys = src["wkey"]; wid = src["wid"]
            if not len(mm):
                continue
            g_, fw, ft = unkey(keys)
            to = int(M3["toff"])

            def mlook(M, tag):
                out = np.zeros(len(keys))
                for gg in range(6):
                    X = M[f"p_{tag}_g{gg}"]
                    m = (g_ == gg) & (fw < X.shape[0]) & (ft + to >= 0) & (ft + to < X.shape[1])
                    out[m] = X[fw[m], ft[m] + to]
                return out
            vals = dict(cls=lookup({gg: F[gg]["tuned_any"] for gg in range(6)}, keys),
                        M6d=mlook(M3, "M6d"), M6dc=mlook(M3, "M6dc"), M4n=mlook(M2, "M4n"), M6dcn=mlook(MC, "M6d"))
            if TN:
                vals["M6dtn"] = mlook(MT, "M6d"); vals["clstn"] = lookup(FT_, keys)
            tru = lookup(T, keys)
            starts = np.searchsorted(wid, np.arange(len(mm))); ends = np.append(starts[1:], len(wid))
            for k, v in vals.items():
                s = np.maximum.reduceat(v, starts) if len(v) else np.zeros(len(mm))
                s[ends == starts] = 0
                S[k].append(s)
            for i, m_ in enumerate(mm):
                r = dict(ev=ev, kind=m_["kind"], g=m_["g"], p=m_["g"] % 3, w0=m_["w0"], t0=m_["t0"])
                if m_["kind"] == "iso":
                    r["E"] = float(m_["E"][0])
                    a = ann.get((ev, m_["g"], m_["w0"], m_["t0"]))
                    if a:
                        r["snr"], r["peak"], r["foreign"], r["q"] = a
                rows.append(r)
                CELLS.append({k: vals[k][starts[i]:ends[i]] for k in ("M6d", "M4n")} | {"T": tru[starts[i]:ends[i]]})
    S = {k: np.concatenate(v) for k, v in S.items()}
    return rows, S, CELLS


def attach_snr(rows):
    D = {}
    for f in sorted(glob.glob(f"{CFS}/causes2/snr/ev*.json")):
        for r in json.load(open(f)):
            D[(r["ev"], r["g"], r["w0"], r["t0"])] = r
    # pooled per-plane null quantiles per input (stat units)
    nulls = {(p, s): [] for p in range(3) for s in ("gate", "k1", "corp")}
    sd = {}
    for r in D.values():
        if "s_gate" in r:
            for s in ("gate", "k1", "corp"):
                nulls[(r["g"] % 3, s)] += r[f"null_{s}"]
    thr = {k: float(np.quantile(v, 0.99)) for k, v in nulls.items()}
    for p in range(3):
        sd[p] = float(np.std(nulls[(p, "gate")]))
    for r in rows:
        d = D.get((r["ev"], r["g"], r["w0"], r["t0"])) if r["kind"] == "iso" else None
        if d and "s_gate" in d:
            p = r["p"]
            for s in ("gate", "k1", "corp"):
                r[f"z_{s}"] = d[f"s_{s}"] / sd[p]
                r[f"kl_{s}"] = d[f"s_{s}"] > thr[(p, s)]
            r["resp_k1"] = d["resp_k1"] / sd[p]; r["resp_gate"] = d["resp_gate"] / sd[p]
        elif d:
            for s in ("gate", "k1", "corp"):
                r[f"z_{s}"] = 0.0; r[f"kl_{s}"] = False
    return thr, sd


def thresholds(R, S, kind="bgn"):
    th = {}
    for k, s in S.items():
        for p in range(3):
            m = (R["kind"] == kind) & (R["p"] == p)
            th[(k, p)] = float(np.quantile(s[m], 0.99))
    return th


def det(R, S, th):
    return {k: np.array([s[i] > th[(k, R["p"][i])] for i in range(len(s))]) for k, s in S.items()}


def boot_eff(x, ev, sel, W, evs):
    """efficiency of boolean x over sel, central + 16-84 over event bootstrap weights W (B, nev)."""
    if sel.sum() == 0:
        return [np.nan, np.nan, np.nan, 0]
    e = np.searchsorted(evs, ev[sel])
    num = np.bincount(e, weights=x[sel].astype(float), minlength=len(evs)); den = np.bincount(e, minlength=len(evs)).astype(float)
    b = (W @ num) / np.maximum(W @ den, 1)
    return [float(num.sum() / den.sum()), float(np.percentile(b, 16)), float(np.percentile(b, 84)), int(sel.sum())]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, S, CELLS = load_windows()
    thr_kl, sd = attach_snr(rows)
    R = {k: np.array([r.get(k, np.nan) for r in rows]) for k in ("ev", "p", "g", "E", "snr", "peak", "foreign", "q",
                                                                    "z_gate", "z_k1", "z_corp", "resp_k1", "resp_gate")}
    R["kind"] = np.array([r["kind"] for r in rows])
    for s in ("gate", "k1", "corp"):
        R[f"kl_{s}"] = np.array([bool(r.get(f"kl_{s}", False)) for r in rows])
    th = thresholds(R, S, "bgn"); thf = thresholds(R, S, "bg")
    D = det(R, S, th); Df = det(R, S, thf)
    for s in ("gate", "k1", "corp"):
        D[f"KL_{s}"] = R[f"kl_{s}"]
    evs = np.array(EVS)
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(evs), len(evs)), minlength=len(evs)) for _ in range(B)]).astype(float)
    iso = R["kind"] == "iso"
    E = R["E"]; eb = iso & (E > 0.1) & (E <= 0.2)
    clean = iso & (R["foreign"] <= 0.25 * R["q"])
    hassnr = iso & np.isfinite(R["z_gate"])
    res = dict(meta=dict(n_iso=int(iso.sum()), n_eb=int(eb.sum()), thresholds={f"{k}|{'UVY'[p]}": v for (k, p), v in th.items()},
                         kl_null99={f"{'UVY'[p]}|{s}": v for (p, s), v in thr_kl.items()}, gate_null_sd={"UVY"[p]: v for p, v in sd.items()}))
    meths = ("cls", "M6d", "M6dcn", "M4n", "M6dc", "KL_gate", "KL_k1", "KL_corp") + tuple(k for k in ("M6dtn", "clstn") if k in S)
    # ---------------- headline per plane
    hd = {}
    for m in meths:
        for pl in (0, 1, 2, 3):
            sel = eb & ((R["p"] == pl) if pl < 3 else True)
            hd[f"{m}|{'UVYa'[pl]}|near"] = boot_eff(D[m], R["ev"], sel, W, evs)
            if m in Df:
                hd[f"{m}|{'UVYa'[pl]}|far"] = boot_eff(Df[m], R["ev"], sel, W, evs)
            hd[f"{m}|{'UVYa'[pl]}|near_clean"] = boot_eff(D[m], R["ev"], sel & clean, W, evs)
    res["headline_0.1-0.2"] = hd
    # ---------------- A/D: efficiency vs analytic SNR, per plane, clean / all
    snr_tab = {}
    for m in meths:
        for pl in (0, 1, 2, 3):
            for cl, csel in (("clean", clean), ("all", iso)):
                for lo, hi, lab in SNRB:
                    sel = csel & ((R["p"] == pl) if pl < 3 else True) & (R["snr"] >= lo) & (R["snr"] < hi) & np.isfinite(R["snr"])
                    if lab == "nosig":
                        sel = csel & ((R["p"] == pl) if pl < 3 else True) & ((R["peak"] <= 0) | (R["snr"] <= 0))
                    if sel.sum() >= 10:
                        snr_tab[f"{m}|{'UVYa'[pl]}|{cl}|{lab}"] = boot_eff(D[m], R["ev"], sel, W, evs)
    res["eff_vs_snr"] = snr_tab
    # ---------------- A: efficiency vs evidence on each input (classical draw gate / k1, stored corpus)
    ztab = {}
    for zname in ("z_gate", "z_k1", "z_corp"):
        for m in ("cls", "M6d", "M6dcn", "KL_gate", "KL_k1", "KL_corp"):
            for pl in (0, 1, 2, 3):
                for cl, csel in (("clean", clean), ("all", iso)):
                    for lo, hi, lab in ZB:
                        sel = csel & hassnr & ((R["p"] == pl) if pl < 3 else True) & (R[zname] >= lo) & (R[zname] < hi)
                        if sel.sum() >= 10:
                            ztab[f"{zname}|{m}|{'UVYa'[pl]}|{cl}|{lab}"] = boot_eff(D[m], R["ev"], sel, W, evs)
    res["eff_vs_evidence"] = ztab
    # ---------------- A: information decomposition (clean & all, 0.1-0.2 MeV and SNR-defined faint set)
    from scipy.stats import binned_statistic
    dec = {}
    edges = np.array([-1e9, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 1e9])

    def f_of(z, x, sel):
        """efficiency of detections x as a step function of evidence z over sel (bin means)."""
        st, _, _ = binned_statistic(z[sel], x[sel].astype(float), "mean", bins=edges)
        return np.nan_to_num(st, nan=1.0)
    for pl in (0, 1, 2, 3):
        for cl, csel in (("clean", clean), ("all", iso)):
            for setn, ssel in (("0.1-0.2", eb), ("snr3-10", iso & (R["snr"] >= 3) & (R["snr"] < 10))):
                sel = csel & ssel & hassnr & ((R["p"] == pl) if pl < 3 else True)
                if sel.sum() < 30:
                    continue
                Ec, Em, Emn = D["cls"][sel].mean(), D["M6d"][sel].mean(), D["M6dcn"][sel].mean()
                pl_ = (R["p"] == pl) if pl < 3 else np.ones(len(R["p"]), bool)
                base = csel & hassnr & pl_                     # curves from all deposits of the plane set
                fM = f_of(R["z_k1"], D["M6dcn"], base)          # model eff vs ITS evidence (k1, same draw)
                fC = f_of(R["z_gate"], D["cls"], base)          # classical eff vs ITS evidence (gate)
                ib = np.clip(np.digitize(R["z_gate"][sel], edges) - 1, 0, len(edges) - 2)
                ik = np.clip(np.digitize(R["z_k1"][sel], edges) - 1, 0, len(edges) - 2)
                M_full = fM[ib].mean()                         # model's evidence->eff curve fed full-waveform evidence
                C_corp = fC[ik].mean()                         # classical's curve fed corpus evidence
                gap = Ec - Emn
                dec[f"{'UVYa'[pl]}|{cl}|{setn}"] = dict(
                    n=int(sel.sum()), cls=float(Ec), M6d_stored=float(Em), M6d_cn=float(Emn),
                    KL_gate=float(D["KL_gate"][sel].mean()), KL_k1=float(D["KL_k1"][sel].mean()), KL_corp=float(D["KL_corp"][sel].mean()),
                    model_curve_on_full_evidence=float(M_full), cls_curve_on_corpus_evidence=float(C_corp),
                    gap_cls_minus_M6dcn=float(gap),
                    info_share_modelcurve=float((M_full - Emn) / gap) if abs(gap) > 1e-9 else None,
                    info_share_clscurve=float((Ec - C_corp) / gap) if abs(gap) > 1e-9 else None,
                    median_z_gate=float(np.median(R["z_gate"][sel])), median_z_k1=float(np.median(R["z_k1"][sel])),
                    median_z_corp=float(np.median(R["z_corp"][sel])), median_snr=float(np.nanmedian(R["snr"][sel])),
                    frac_k1_zero=float(np.mean(R["z_k1"][sel] == 0)), frac_corp_zero=float(np.mean(R["z_corp"][sel] == 0)))
    res["info_decomposition"] = dec
    # ---------------- A(ii): strong-evidence misses
    sel = iso & hassnr & R["kl_corp"] & (R["z_corp"] >= 6)
    miss = sel & ~D["M6d"]
    hit = sel & D["M6d"]
    ms = dict(n_strong=int(sel.sum()), n_miss=int(miss.sum()), miss_rate=float(miss.sum() / max(sel.sum(), 1)))
    for nm, mk in (("miss", miss), ("hit", hit)):
        ms[nm] = dict(plane=[int(((R["p"] == p) & mk).sum()) for p in range(3)],
                      foreign_over_q_median=float(np.median((R["foreign"] / R["q"])[mk])) if mk.any() else None,
                      frac_clean=float(np.mean(clean[mk])) if mk.any() else None,
                      E_median=float(np.median(E[mk])) if mk.any() else None,
                      snr_median=float(np.nanmedian(R["snr"][mk])) if mk.any() else None,
                      far_detected=float(np.mean(Df["M6d"][mk])) if mk.any() else None,
                      cls_detected=float(np.mean(D["cls"][mk])) if mk.any() else None,
                      M6dcn_detected=float(np.mean(D["M6dcn"][mk])) if mk.any() else None)
    # per plane miss rate and miss rate per E bin
    ms["miss_rate_by_plane"] = [float(miss[R["p"] == p].sum() / max(sel[R["p"] == p].sum(), 1)) for p in range(3)]
    ms["miss_rate_by_E"] = {lab: float(miss[(E > lo) & (E <= hi)].sum() / max(sel[(E > lo) & (E <= hi)].sum(), 1))
                            for lo, hi, lab in ((0, 0.1, "<0.1"), (0.1, 0.2, "0.1-0.2"), (0.2, 0.5, "0.2-0.5"), (0.5, 1e9, ">0.5"))}
    # score margin of misses relative to threshold, and threshold-limited share (detected at the far threshold)
    sc = S["M6d"]
    ms["miss_score_over_thr_median"] = float(np.median([sc[i] / th[("M6d", R["p"][i])] for i in np.nonzero(miss)[0]])) if miss.any() else None
    res["strong_evidence_misses"] = ms
    res["_miss_list"] = [dict(ev=int(R["ev"][i]), g=int(R["g"][i]), w0=int(rows[i]["w0"]), t0=int(rows[i]["t0"]), E=float(E[i]),
                              snr=float(R["snr"][i]), z_corp=float(R["z_corp"][i]), z_gate=float(R["z_gate"][i]),
                              foreign_over_q=float(R["foreign"][i] / R["q"][i]), score=float(sc[i]), thr=th[("M6d", R["p"][i])],
                              cls=bool(D["cls"][i]), M6dcn=bool(D["M6dcn"][i]), far=bool(Df["M6d"][i]))
                         for i in np.nonzero(miss)[0]]
    # ---------------- D: matched-SNR plane gap
    dg = {}
    for m in meths:
        for cl, csel in (("clean", clean), ("all", iso)):
            for setn, ssel in (("0.1-0.2", eb),):
                for zname, zb in (("snr", SNRB[1:]), ("z_corp", ZB), ("z_gate", ZB)):
                    eff = {}
                    for p in range(3):
                        sp = csel & ssel & (R["p"] == p) & np.isfinite(R[zname])
                        eff[p] = (sp, D[m][sp].mean() if sp.any() else np.nan)
                    Ey = eff[2][1]
                    for p in (0, 1):
                        sp = eff[p][0]
                        # Y efficiency re-weighted to plane p's evidence distribution
                        num = den = 0.0
                        for lo, hi, lab in zb:
                            np_ = (sp & (R[zname] >= lo) & (R[zname] < hi)).sum()
                            sy = eff[2][0] & (R[zname] >= lo) & (R[zname] < hi)
                            if np_ and sy.sum() >= 5:
                                num += np_ * D[m][sy].mean(); den += np_
                            elif np_:
                                num += np_ * D[m][sp & (R[zname] >= lo) & (R[zname] < hi)].mean(); den += np_   # no Y support: no gap
                        Ey_rw = num / den if den else np.nan
                        raw = Ey - eff[p][1]
                        dg[f"{m}|{cl}|{setn}|{zname}|{'UV'[p]}"] = dict(
                            eff_plane=float(eff[p][1]), eff_Y=float(Ey), eff_Y_at_plane_dist=float(Ey_rw), raw_gap=float(raw),
                            residual_gap=float(Ey_rw - eff[p][1]),
                            share_explained=float(1 - (Ey_rw - eff[p][1]) / raw) if abs(raw) > 1e-9 else None,
                            median_evidence_plane=float(np.nanmedian(R[zname][sp])), median_evidence_Y=float(np.nanmedian(R[zname][eff[2][0]])))
    res["plane_gap"] = dg
    zs = {}
    for p in range(3):
        for lo, hi, lab in ((0, 0.1, "<0.1"), (0.1, 0.2, "0.1-0.2")):
            s_ = iso & (R["p"] == p) & (E > lo) & (E <= hi)
            zs[f"{'UVY'[p]}|{lab}"] = dict(n=int(s_.sum()), nosignal=float(np.mean((R["peak"][s_] <= 0))),
                                           median_snr=float(np.nanmedian(R["snr"][s_])),
                                           median_z_gate=float(np.nanmedian(R["z_gate"][s_])),
                                           median_z_k1=float(np.nanmedian(R["z_k1"][s_])), median_z_corp=float(np.nanmedian(R["z_corp"][s_])),
                                           frac_corp_zero=float(np.nanmean(R["z_corp"][s_] == 0)),
                                           survive_k1=float(np.nanmedian((R["resp_k1"] / R["resp_gate"])[s_ & (R["resp_gate"] > 1)])))
    res["plane_physics"] = zs
    # ---------------- E: window-level vs cell-level for M4n vs M6d presence
    e = {}
    isoT = [i for i in range(len(rows)) if rows[i]["kind"] == "iso"]
    bgnT = [i for i in range(len(rows)) if rows[i]["kind"] == "bgn"]
    for m in ("M4n", "M6d"):
        for p in range(3):
            t = th[(m, p)]
            # empty cells of bgn windows: per-cell FPR at the window threshold, and n_eff
            ce = np.concatenate([CELLS[i][m][CELLS[i]["T"] <= 0] for i in bgnT if rows[i]["p"] == p])
            fcell = float(np.mean(ce > t))
            ncell = np.array([np.sum(CELLS[i][m][CELLS[i]["T"] <= 0] > t) for i in bgnT if rows[i]["p"] == p])
            n_eff = float(np.log(0.99) / np.log(1 - fcell)) if 0 < fcell < 1 else None
            # iso windows: number of cells above threshold, detected fraction, charged vs empty cells above
            sel_iso = [i for i in isoT if rows[i]["p"] == p and 0.1 < rows[i]["E"] <= 0.2]
            nab = np.array([np.sum(CELLS[i][m] > t) for i in sel_iso])
            nab_ch = np.array([np.sum((CELLS[i][m] > t) & (CELLS[i]["T"] > 0)) for i in sel_iso])
            mx_is_charged = np.array([CELLS[i]["T"][np.argmax(CELLS[i][m])] > 0 for i in sel_iso])
            e[f"{m}|{'UVY'[p]}"] = dict(window_thr=t, cell_fpr_at_window_thr=fcell, n_cells_bgn=int(len(ce)), n_eff_independent=n_eff,
                                       bgn_windows_with_k_cells_above=[float(np.mean(ncell == k)) for k in range(4)] + [float(np.mean(ncell >= 4))],
                                       iso_eff=float(np.mean(nab > 0)), iso_mean_cells_above_if_detected=float(nab[nab > 0].mean()) if (nab > 0).any() else None,
                                       iso_frac_detected_by_charged_cell=float(np.mean(nab_ch[nab > 0] > 0)) if (nab > 0).any() else None,
                                       iso_argmax_on_charged=float(np.mean(mx_is_charged)))
            # efficiency at matched CELL-level FPR (empty cells in bgn windows), several FPRs
            cs = np.sort(ce)[::-1]
            for f in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
                tf = cs[min(int(f * len(cs)), len(cs) - 1)]
                wfpr = float(np.mean([CELLS[i][m].max() > tf for i in bgnT if rows[i]["p"] == p]))
                ie = float(np.mean([CELLS[i][m].max() > tf for i in sel_iso]))
                e[f"{m}|{'UVY'[p]}|cellfpr{f:g}"] = dict(thr=float(tf), window_fpr=wfpr, iso_window_eff=ie)
    # window-max distributions (bgn) upper quantiles
    for m in ("M4n", "M6d"):
        for p in range(3):
            s = S[m][(R["kind"] == "bgn") & (R["p"] == p)]
            e[f"{m}|{'UVY'[p]}|bgn_max_quantiles_90_99_999"] = [float(x) for x in np.quantile(s, (0.9, 0.99, 0.999))]
            si = S[m][eb & (R["p"] == p)]
            e[f"{m}|{'UVY'[p]}|iso0.1-0.2_max_quantiles_10_50_90"] = [float(x) for x in np.quantile(si, (0.1, 0.5, 0.9))]
    # paired: deposits detected by one but not the other
    for p in range(3):
        s_ = eb & (R["p"] == p)
        e[f"paired|{'UVY'[p]}"] = dict(both=float(np.mean(D["M6d"][s_] & D["M4n"][s_])), only_M6d=float(np.mean(D["M6d"][s_] & ~D["M4n"][s_])),
                                     only_M4n=float(np.mean(~D["M6d"][s_] & D["M4n"][s_])))
    res["E"] = e
    # cell-level ROC at low FPR by density (dense3 records: every charged cell, density-weighted empty sample)
    C = json.load(open(f"{CFS}/dense3/rec/columns.json"))
    fs = sorted(glob.glob(f"{CFS}/dense3/rec/ev*.npz"))
    SG = np.concatenate([np.load(f)["sig"] for f in fs]); EM = np.concatenate([np.load(f)["emp"] for f in fs])
    sc_, ec_ = C["sig_cols"], C["emp_cols"]
    DBn = [(-1, 0.5, "iso"), (0.5, 1e4, "<=10k"), (1e4, 1e5, "10k-100k"), (1e5, 1e6, "100k-1M"), (1e6, 4e6, "1M-4M"), (4e6, 1e12, ">4M")]
    roc = {}
    for m in ("cls_tuned_any", "M4n_p", "M6d_p", "M6dc_p"):
        for lo, hi, lab in DBn:
            for f in (1e-4, 1e-3, 1e-2):
                hits = {q: [0, 0] for q in ("1-2k", "2-4k", "all")}
                for p in range(3):
                    me = (EM[:, 0] == p) & (EM[:, 2] > lo) & (EM[:, 2] <= hi)
                    x, w = EM[me, ec_.index(m)], EM[me, 1]
                    if w.sum() < 10 / f:
                        continue
                    o = np.argsort(x)[::-1]; cw = np.cumsum(w[o]) / w.sum()
                    t = x[o][min(np.searchsorted(cw, f), len(o) - 1)]
                    ms_ = (SG[:, 0] == p) & (SG[:, 2] > lo) & (SG[:, 2] <= hi)
                    for q, (a_, b_) in (("1-2k", (1e3, 2e3)), ("2-4k", (2e3, 4e3)), ("all", (0, 1e12))):
                        mq = ms_ & (SG[:, 1] > a_) & (SG[:, 1] <= b_)
                        hits[q][0] += int((SG[mq, sc_.index(m)] > t).sum()); hits[q][1] += int(mq.sum())
                for q, (h, n) in hits.items():
                    if n >= 30:
                        roc[f"{m}|{lab}|{q}|fpr{f:g}"] = [h / n, n]
    res["E_cell_roc"] = roc
    # ---------------- (iii) noise realisation: paired differences on identical deposits
    pr = {}
    pairs = [("M6d", "M6dcn"), ("cls", "M6dcn"), ("cls", "M6d")] + ([("M6d", "M6dtn"), ("M6dcn", "M6dtn"), ("cls", "clstn"), ("clstn", "M6dtn")] if "M6dtn" in S else [])
    for a_, b_ in pairs:
        for pl in (0, 1, 2, 3):
            for setn, ssel in (("0.1-0.2", eb), ("0.1-0.2_clean", eb & clean), ("snr3-10", iso & (R["snr"] >= 3) & (R["snr"] < 10))):
                sel = ssel & ((R["p"] == pl) if pl < 3 else True)
                d_ = D[a_][sel].astype(float) - D[b_][sel].astype(float)
                e_ = np.searchsorted(evs, R["ev"][sel])
                num = np.bincount(e_, weights=d_, minlength=len(evs)); den = np.bincount(e_, minlength=len(evs)).astype(float)
                bb = (W @ num) / np.maximum(W @ den, 1)
                pr[f"{a_}-{b_}|{'UVYa'[pl]}|{setn}"] = [float(d_.mean()), float(np.percentile(bb, 16)), float(np.percentile(bb, 84)), int(sel.sum())]
    res["paired_diff"] = pr
    # ---------------- A: per-deposit attribution of the classical-only detections
    at = {}
    for mod, zk, kl in (("M6d", "z_corp", "kl_corp"), ("M6dcn", "z_k1", "kl_k1")):
        for pl in (0, 1, 2, 3):
            for setn, ssel in (("0.1-0.2", eb), ("snr3-10", iso & (R["snr"] >= 3) & (R["snr"] < 10)), ("snr3-10_clean", clean & (R["snr"] >= 3) & (R["snr"] < 10))):
                sel = ssel & hassnr & ((R["p"] == pl) if pl < 3 else True)
                c_only = sel & D["cls"] & ~D[mod]; m_only = sel & ~D["cls"] & D[mod]
                n = max(sel.sum(), 1)
                at[f"{mod}|{'UVYa'[pl]}|{setn}"] = dict(
                    n=int(sel.sum()), cls_only=float(c_only.sum() / n), model_only=float(m_only.sum() / n),
                    cls_only_no_evidence=float((c_only & ~R[kl]).sum() / n),                 # model input fails known-location
                    cls_only_weak=float((c_only & R[kl] & (R[zk] < 4)).sum() / n),
                    cls_only_strong=float((c_only & R[kl] & (R[zk] >= 4)).sum() / n),
                    model_only_no_full_evidence=float((m_only & ~R["kl_gate"]).sum() / n))
    res["attribution"] = at
    # ---------------- E: exact decomposition of the M6d - M4n window gain
    ed = {}
    for p in range(3):
        bg_ = [i for i in bgnT if rows[i]["p"] == p]
        si = [i for i in isoT if rows[i]["p"] == p and 0.1 < rows[i]["E"] <= 0.2]
        out = {}
        cellsE = {m: np.sort(np.concatenate([CELLS[i][m][CELLS[i]["T"] <= 0] for i in bg_]))[::-1] for m in ("M4n", "M6d")}

        def at_fpr(m, f):
            cs = cellsE[m]; return cs[min(int(round(f * len(cs))), len(cs) - 1)]

        def effs_(m, t):
            ch = np.array([np.any((CELLS[i][m] > t) & (CELLS[i]["T"] > 0)) for i in si])
            an = np.array([np.any(CELLS[i][m] > t) for i in si])
            return float(an.mean()), float(ch.mean()), float((an & ~ch).mean())
        t4, t6 = th[("M4n", p)], th[("M6d", p)]
        f4 = float(np.mean(cellsE["M4n"] > t4)); f6 = float(np.mean(cellsE["M6d"] > t6))
        t6_at_f4 = at_fpr("M6d", f4)
        e4, c4, x4 = effs_("M4n", t4); e6, c6, x6 = effs_("M6d", t6); e6f, c6f, x6f = effs_("M6d", t6_at_f4)
        out.update(gain=e6 - e4, M4n_eff=e4, M6d_eff=e6, M4n_cell_fpr=f4, M6d_cell_fpr=f6,
                   corr_part=e6 - e6f, rank_charged_part=c6f - c4, spread_empty_only_part=x6f - x4,
                   M4n_detect_charged=c4, M4n_detect_empty_only=x4, M6d_detect_charged_at_f4=c6f, M6d_detect_empty_only_at_f4=x6f)
        for f in (1e-4, 1e-3, 1e-2):
            out[f"charged_cell_detection_at_cellfpr{f:g}"] = dict(M4n=effs_("M4n", at_fpr("M4n", f))[1], M6d=effs_("M6d", at_fpr("M6d", f))[1])
        ed["UVY"[p]] = out
    res["E_decomposition"] = ed
    np.savez(f"{CFS}/causes2/windows.npz", **{k: v for k, v in R.items()}, **{f"S_{k}": v for k, v in S.items()},
             **{f"D_{k}": v for k, v in D.items()}, **{f"Df_{k}": v for k, v in Df.items()},
             w0=np.array([r["w0"] for r in rows]), t0=np.array([r["t0"] for r in rows]))
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
