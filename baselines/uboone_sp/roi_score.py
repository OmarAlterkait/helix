#!/usr/bin/env python3
"""Score the ROI threshold sweep (roi_sweep.py) and the model's ROI-level behaviour.

    scripts/helix_run.sh python baselines/uboone_sp/roi_score.py \
        --sweep /pscratch/sd/o/oalter/helix_work/uboone_sp/roi_sweep \
        --model-cells /pscratch/sd/o/oalter/helix_work/uboone_sp/model_m4n --out baselines/uboone_sp/results

Window scores are exactly eval_denoise's: helix.probe.resolution.window_stats
(max over the window's 2x16 cells of the predicted charge) on the floor windows
(truth_v2) plus the near-activity windows (truth_v2_near), iso windows annotated
with snr/clean from noise_vs_hits.npz (as eval_denoise --annot), then scalars /
near_scalars.

Per setting:
  ROI operating point  FPR = fraction of far (bg) / near (bgn) noise windows with
                       ANY ROI charge (score > 0); efficiency = fraction of iso
                       windows with any ROI charge
  ROC                  thresholding the window score (efficiency vs FPR)
  eff@1% FPR           at the 99th percentile of the bg (far) or bgn (near)
                       scores, per energy bin and per SNR bin; per plane with
                       each plane's own threshold
"Best" over settings is chosen ON THESE TEST EVENTS (optimistic; labelled).

The model (M4n_ft_full_near, presence-scored): its own near-scored windows file
(dn_near_M4n_ft_full_near_pres_trained_windows.json) for every window metric, and
its CPU-replayed per-cell outputs (model_presence.py) for the ROI-level
decomposition (cells with presence > threshold as its 'ROI').
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"
ANNOT = "/pscratch/sd/o/oalter/helix_work/denoise/noise_vs_hits.npz"
MODEL_W = "/pscratch/sd/o/oalter/helix_work/denoise/results/dn_near_M4n_ft_full_near_pres_trained_windows.json"
EB = ("0-0.1", "0.1-0.2", "0.2-0.5", "0.5-1", "1-1e+09")
SNRB = ((0, 1e-9, "nosignal"), (1e-9, 3, "1e-09-3"), (3, 5, "3-5"), (5, 7, "5-7"), (7, 10, "7-10"), (10, 15, "10-15"),
        (15, 1e9, "15-inf"))


def metrics(W):
    """Window-level numbers for one window list (window_stats records + kind/g/snr/clean)."""
    from helix.probe.resolution import near_scalars, scalars
    s = scalars(np.zeros(2), np.array([0.0, 1.0]), [], W, 1.0)
    out = {k: v for k, v in s.items() if k.startswith(("floor_eff", "floor_auc"))}
    out.update(near_scalars(W))
    bg = np.array([w["score"] for w in W if w["kind"] == "bg"])
    bn = np.array([w["score"] for w in W if w["kind"] == "bgn"])
    iso = [w for w in W if w["kind"] == "iso"]
    si = np.array([w["score"] for w in iso])
    out["roi_fpr_far"] = float(np.mean(bg > 0)); out["roi_fpr_near"] = float(np.mean(bn > 0))
    out["roi_eff"] = float(np.mean(si > 0))
    for i, nm in enumerate(EB):
        m = np.array([w.get("eb") == i for w in iso])
        if m.any():
            out[f"roi_eff_{nm}"] = float(np.mean(si[m] > 0))
    # ROC by thresholding the score
    thr = np.unique(np.concatenate([[0.0], np.quantile(np.concatenate([bg, bn, si]), np.linspace(0, 1, 401))]))
    ebs = np.array([w.get("eb", -1) for w in iso])
    for lab, m in (("all", np.ones(len(si), bool)), ("E0-0.1", ebs == 0), ("E0.1-0.2", ebs == 1), ("E0.2-0.5", ebs == 2)):
        out[f"roc_far_{lab}"] = [[float(np.mean(bg > t)), float(np.mean(si[m] > t)), float(t)] for t in thr]
        out[f"roc_near_{lab}"] = [[float(np.mean(bn > t)), float(np.mean(si[m] > t)), float(t)] for t in thr]
    for p in range(3):
        Wp = [w for w in W if w["g"] % 3 == p]
        sp = scalars(np.zeros(2), np.array([0.0, 1.0]), [], Wp, 1.0)
        npp = near_scalars(Wp)
        out[f"plane_{'UVY'[p]}"] = dict({k: v for k, v in sp.items() if k.startswith("floor_eff1pct")},
                                        **{k: v for k, v in npp.items() if k.startswith(("near_eff1pct", "near_fpr", "snr_eff1pct"))},
                                        roi_fpr_far=float(np.mean([w["score"] > 0 for w in Wp if w["kind"] == "bg"])),
                                        roi_fpr_near=float(np.mean([w["score"] > 0 for w in Wp if w["kind"] == "bgn"])),
                                        roi_eff=float(np.mean([w["score"] > 0 for w in Wp if w["kind"] == "iso"])))
    return out


def one(args):
    name, W = args
    return name, metrics(W)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--model-cells", default="/pscratch/sd/o/oalter/helix_work/uboone_sp/model_m4n")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from helix.data.denoise import DenoiseEvents
    from helix.probe import resolution as R
    import roi_sweep as RS
    cfg = DenoiseEvents("", [], "", items=[]).cfg
    toff = {i: cfg.toff[i] for i in range(3)}

    evs = sorted({int(os.path.basename(f)[2:5]) for f in glob.glob(os.path.join(a.sweep, "ev*_g*.npz"))})
    evs = [e for e in evs if all(os.path.exists(os.path.join(a.sweep, f"ev{e:03d}_g{g}.npz")) for g in range(6))]
    A = np.load(ANNOT)
    ann = {k: (float(s) if pk > 0 else 0.0, bool(fo <= 0.25 * q_))
           for k, s, pk, fo, q_ in zip(zip(A["ev"].tolist(), A["g"].tolist(), A["w0"].tolist(), A["t0"].tolist()),
                                       A["snr"], A["peak"], A["foreign"], A["q"])}
    names = list(RS.NAMES)
    for ev in evs:                                       # settings present in EVERY file (later runs skipped some)
        for g in range(6):
            fl = set(np.load(os.path.join(a.sweep, f"ev{ev:03d}_g{g}.npz")).files)
            names = [n for n in names if f"w_{n}" in fl]
    print("settings scored:", names, flush=True)
    META, WQ, WID, WP = [], [], [], {n: [] for n in names}
    DEC = {n: [] for n in names}; PUR = {n: np.zeros((3, 6)) for n in names}; DMETA = []
    for ev in evs:
        z = np.load(os.path.join(TRUTH, f"ev{ev:03d}.npz"), allow_pickle=True)
        zn = np.load(os.path.join(NEAR, f"ev{ev:03d}.npz"), allow_pickle=True)
        F = {g: np.load(os.path.join(a.sweep, f"ev{ev:03d}_g{g}.npz")) for g in range(6)}
        for src, pre in ((z, "w_"), (zn, "n_")):
            base = len(META)
            mm = json.loads(str(src["meta"]))
            for m_ in mm:
                m_["ev"] = ev
                k_ = (ev, m_["g"], m_["w0"], m_["t0"])
                if m_["kind"] == "iso" and k_ in ann:
                    m_["snr"], m_["clean"] = ann[k_]
            META += mm; WQ.append(src["wq"]); WID.append(src["wid"] + base)
            for n in names:
                WP[n].append(sum(F[g][f"{pre}{n}"].astype(np.float64) for g in range(6)))
        for g in range(6):
            im = F[g]["iso_meta"]
            for j in range(len(im)):
                k_ = (ev, g, int(im[j, 0]), int(im[j, 1]))
                DMETA.append(dict(ev=ev, g=g, E=float(im[j, 2]), snr=ann.get(k_, (np.nan, False))[0],
                                  clean=ann.get(k_, (np.nan, False))[1]))
            for n in names:
                DEC[n].append(F[g][f"iso_{n}"].reshape(-1, 5))
                PUR[n][g % 3] += F[g][f"pur_{n}"]
    wq, wid = np.concatenate(WQ), np.concatenate(WID)
    starts = np.searchsorted(wid, np.arange(len(META))); ends = np.append(starts[1:], len(wid))
    print(f"{len(evs)} events, {len(META)} windows", flush=True)

    jobs = []
    for n in names:
        W = R.window_stats(np.concatenate(WP[n]), wq, META, starts, ends, cfg.pw, cfg.pt, cfg.delta[0], toff)
        for w_, m_ in zip(W, META):
            w_.update(g=m_["g"], **{k: m_[k] for k in ("snr", "clean") if k in m_})
        jobs.append((n, W))
    # post-hoc hybrid: each plane type from the setting that suits it (chosen after seeing the per-plane results)
    byname = dict(jobs)
    for hname, pick in (("hyb_UVfixU_Ynofake", {0: "wc_fixU", 1: "wc_fixU", 2: "wc_nofake"}),
                        ("hyb_UVfixU_Ynofake_s0.83", {0: "wc_fixU", 1: "wc_fixU", 2: "wc_nofake_s0.83"})):
        if all(v in byname for v in pick.values()):
            jobs.append((hname, [byname[pick[m_["g"] % 3]][i] for i, m_ in enumerate(META)]))
    MW = json.load(open(MODEL_W))
    same = len(MW) == len(META) and all((w["ev"], w["g"], w["w0"], w["t0"], w["kind"]) == (m_["ev"], m_["g"], m_["w0"], m_["t0"], m_["kind"])
                                        for w, m_ in zip(MW, META))
    print("model windows identical:", same, flush=True)
    jobs.append(("model", MW))
    res = {}
    with Pool(min(a.workers, len(jobs))) as pool:
        for n, r in pool.imap_unordered(one, jobs):
            res[n] = dict(metrics=r)
    # ---- ROI-level decomposition (classical)
    E_ = np.array([d["E"] for d in DMETA]); P_ = np.array([d["g"] % 3 for d in DMETA])
    S_ = np.array([d["snr"] for d in DMETA]); CL = np.array([bool(d["clean"]) for d in DMETA])
    ebi = np.searchsorted(R.ENERGY_BINS, E_, side="right") - 1
    for n in names:
        D = np.concatenate(DEC[n])                         # qtrue_box, qtrue_in_roi, covered, qreco_box, qreco_pos_box
        dec = {}
        for p in range(4):
            mp = (P_ == p) if p < 3 else np.ones(len(P_), bool)
            lab = "UVY"[p] if p < 3 else "all"
            row = {}
            for i, nm in enumerate(EB[:4]):
                m = mp & (ebi == i) & (D[:, 0] > 0)
                if m.sum() < 5:
                    continue
                cov = D[m, 2] > 0
                row[f"E{nm}"] = dict(n=int(m.sum()), roi_eff=float(cov.mean()),
                                     q_in_roi_frac_if_found=float(np.median(D[m & (D[:, 2] > 0), 1] / D[m & (D[:, 2] > 0), 0])) if cov.any() else None,
                                     reco_over_true_if_found=float(np.median(D[m & (D[:, 2] > 0), 3] / D[m & (D[:, 2] > 0), 0])) if cov.any() else None)
            for lo, hi, nm in SNRB:
                m = mp & CL & (S_ >= lo) & (S_ < hi) & (D[:, 0] > 0)
                if m.sum() < 5:
                    continue
                row[f"SNR{nm}"] = dict(n=int(m.sum()), roi_eff=float((D[m, 2] > 0).mean()))
            dec[lab] = row
        pur = {"UVY"[p]: dict(roi_pixels=float(PUR[n][p, 0]), frac_area_truth_empty=float(PUR[n][p, 1] / max(PUR[n][p, 0], 1)),
                              frac_pos_charge_truth_empty=float(PUR[n][p, 3] / max(PUR[n][p, 2], 1)),
                              frac_truth_charge_in_roi=float(PUR[n][p, 4] / max(PUR[n][p, 5], 1))) for p in range(3)}
        res[n]["decomposition"] = dec; res[n]["purity"] = pur
        res[n]["_dec_rows"] = None
    # charge resolution inside found vs oracle ROIs, per chain (deposits found by the default setting)
    for chain, dflt, orc in (("ex", "ex_s1", "ex_oracle"), ("wc", "wc_s1", "wc_oracle")):
        Df, Do = np.concatenate(DEC[dflt]), np.concatenate(DEC[orc])
        out = {}
        for p in range(4):
            mp = (P_ == p) if p < 3 else np.ones(len(P_), bool)
            for i, nm in enumerate(EB[:4]):
                m = mp & (ebi == i) & (Df[:, 0] > 0) & (Df[:, 2] > 0)
                if m.sum() < 5:
                    continue
                rf, ro = Df[m, 3] / Df[m, 0], Do[m, 3] / Do[m, 0]
                out[f"{'UVY'[p] if p < 3 else 'all'}_E{nm}"] = dict(
                    n=int(m.sum()), found_median=float(np.median(rf)), found_hw68=float((np.percentile(rf, 84) - np.percentile(rf, 16)) / 2),
                    oracle_median=float(np.median(ro)), oracle_hw68=float((np.percentile(ro, 84) - np.percentile(ro, 16)) / 2),
                    truth_in_found_roi=float(np.median(Df[m, 1] / Df[m, 0])))
        res[f"_charge_{chain}"] = out
    # ---- model ROI-level decomposition from per-cell presence
    res["model"]["decomposition"] = model_decomposition(a.model_cells, evs, META, wq, starts, ends, S_ if False else None, ann)
    res["_meta"] = dict(n_events=len(evs), events=evs, n_windows=len(META), settings=names, model_windows_identical=same)
    for n in res:
        if isinstance(res[n], dict):
            res[n].pop("_dec_rows", None)
    with open(os.path.join(a.out, "roi_sweep.json"), "w") as fh:
        json.dump(res, fh, default=float)
    print("wrote", os.path.join(a.out, "roi_sweep.json"))


def model_decomposition(mdir, evs, META, wq, starts, ends, _unused, ann):
    """The model's 'ROI' = cells with presence > t. For t at the model's own far and near
    1%-FPR window thresholds: per iso window, whether any truth cell is in the ROI,
    the fraction of the window's truth charge in ROI cells; purity over every covered
    cell from the per-plane presence histograms."""
    from helix.probe import resolution as R
    files = [os.path.join(mdir, f"ev{e:03d}.npz") for e in evs]
    if not all(os.path.exists(f) for f in files):
        return {"missing": sum(not os.path.exists(f) for f in files)}
    PW = []; HS = np.zeros((6, 100)); HE = np.zeros((6, 100))
    for f in files:
        z = np.load(f)
        PW += [z["pres_w"].astype(np.float64), z["pres_n"].astype(np.float64)]
        HS += z["hist_sig"]; HE += z["hist_emp"]
    pw = np.concatenate(PW)
    sc = np.array([pw[s:e].max() if e > s else 0.0 for s, e in zip(starts, ends)])
    kind = np.array([m["kind"] for m in META])
    out = {}
    edges = np.linspace(0, 1, 101)
    for lab, thr in (("far1pct", np.quantile(sc[kind == "bg"], 0.99)), ("near1pct", np.quantile(sc[kind == "bgn"], 0.99))):
        rows = {}
        for p in range(4):
            sel = [i for i, m in enumerate(META) if m["kind"] == "iso" and (p == 3 or m["g"] % 3 == p)]
            cov, fq, Es, Ss = [], [], [], []
            for i in sel:
                q = wq[starts[i]:ends[i]]; pr = pw[starts[i]:ends[i]]
                if q.sum() <= 0:
                    continue
                inr = pr > thr
                cov.append(bool((inr & (q > 0)).any())); fq.append(float(q[inr].sum() / q.sum()))
                Es.append(float(np.sum(META[i]["E"])))
                k_ = (META[i]["ev"], META[i]["g"], META[i]["w0"], META[i]["t0"])
                Ss.append(ann.get(k_, (np.nan, False))[0] if ann.get(k_, (0, False))[1] else np.nan)
            cov, fq, Es, Ss = map(np.array, (cov, fq, Es, Ss))
            ebi = np.searchsorted(R.ENERGY_BINS, Es, side="right") - 1
            r = {}
            for b, nm in enumerate(EB[:4]):
                m = ebi == b
                if m.sum() >= 5:
                    r[f"E{nm}"] = dict(n=int(m.sum()), roi_eff=float(cov[m].mean()),
                                       q_in_roi_frac_if_found=float(np.median(fq[m & cov])) if (m & cov).any() else None)
            for lo, hi, nm in SNRB:
                m = (Ss >= lo) & (Ss < hi)
                if m.sum() >= 5:
                    r[f"SNR{nm}"] = dict(n=int(m.sum()), roi_eff=float(cov[m].mean()))
            if p < 3:
                k = int(np.searchsorted(edges, thr, side="right")) - 1     # bins at or above the threshold bin
                hs, he = HS[[p, p + 3]].sum(0)[k:].sum(), HE[[p, p + 3]].sum(0)[k:].sum()
                r["purity"] = dict(roi_cells=float(hs + he), frac_cells_truth_empty=float(he / max(hs + he, 1)),
                                   note="covered cells with presence in the threshold's histogram bin or above (1% bins)")
            rows["UVY"[p] if p < 3 else "all"] = r
        out[lab] = dict(threshold=float(thr), rows=rows,
                        window_fpr_far=float(np.mean(sc[kind == "bg"] > thr)), window_fpr_near=float(np.mean(sc[kind == "bgn"] > thr)))
    return out


if __name__ == "__main__":
    main()
