#!/usr/bin/env python3
"""Where does each method win, and why -- on identical cells and windows (Part B).

    scripts/helix_run.sh python baselines/uboone_sp/v2/aspects.py \
        --cls-full /pscratch/sd/o/oalter/helix_work/uboone_sp/v2/cls_full \
        --cls /pscratch/sd/o/oalter/helix_work/uboone_sp/v2/cls \
        --model /pscratch/sd/o/oalter/helix_work/uboone_sp/v2/model \
        --bench baselines/uboone_sp/v2/results/benchmark.json --out baselines/uboone_sp/v2/results

Cell-level (every fine cell of every plane of the 128 test events; the model
predicts 0 on cells no token covers, which is its actual output):
  noise      predicted charge on truth-empty cells by chessboard distance (cells)
             to the nearest truth charge: quantiles, rms, fraction above 0.05 / 0.25 / 1 Q0
  fidelity   truth>0 cells: median pred/true and 68% half-width per true-charge bin;
             total charge recovery per plane (all cells, and truth>0 cells only)
Window-level (eval windows; scores = max over the window's cells):
  roc        far / near ROC per plane for the detection scores
  dip        pair windows: helix.probe.resolution dip by separation bin
  neardist   near-activity noise windows: fraction above the FAR 1% threshold
             (per plane) by distance (cells) from the window to the nearest charge
Charge methods: classical default (MicroBooNE), tuned ROI charge, no-ROI Gaus_wide,
oracle-ROI (reference), model charge heads (M4n, N64). Detection scores: the
same plus the tuned best classical statistic and the model presence heads.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, os.path.dirname(os.path.dirname(BASE)))
Q0 = 2043.258
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"
DBINS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 6))
DLAB = ("1", "2", "3-4", "5-8", "9-16", ">16")
QBINS = (0, 500, 1000, 2000, 4000, 8000, 16000, 64000, 1e9)
HEDGE = np.concatenate([[-1e9, 0.0, 1e-6], np.logspace(0, 6.5, 131), [1e12]])


def maps_for(ev, g, picks, cls_full, model_dir):
    F = np.load(os.path.join(cls_full, f"ev{ev:03d}_g{g}.npz"))
    Mz = np.load(os.path.join(model_dir, f"ev{ev:03d}.npz"))
    p = g % 3
    T = F["full_truth"]
    to = int(Mz["toff"])
    crop = lambda X: X[:T.shape[0], to:to + T.shape[1]]        # model maps extend past the readout edges
    m = {"cls_default": F[f"full_{picks['default'][p]}"], "cls_tuned_q": F[f"full_{picks['tuned_q'][p]}"],
         "cls_noroi": F["full_g"], "cls_oracle_roi": F["full_qor"],
         "M4n_q": crop(Mz[f"q_M4n_g{g}"]), "N64_q": crop(Mz[f"q_N64_g{g}"])}
    return T, m, None, crop(Mz[f"cov_g{g}"])


def per_event(args):
    ev, picks, cls_full, model_dir = args
    from scipy.ndimage import distance_transform_cdt
    out = dict(hist={}, sig=[], tot={}, bgn_dist=[], cov={})
    zn = np.load(os.path.join(NEAR, f"ev{ev:03d}.npz"), allow_pickle=True)
    nmeta = json.loads(str(zn["meta"]))
    Dg = {}
    for g in range(6):
        T, m, det, cov = maps_for(ev, g, picks, cls_full, model_dir)
        p = g % 3
        has = T > 0
        D = distance_transform_cdt(~has, metric="chessboard") if has.any() else np.full(T.shape, 10 ** 6)
        Dg[g] = D
        emp = ~has
        for name, X in m.items():
            for (lo, hi), lab in zip(DBINS, DLAB):
                sel = emp & (D >= lo) & (D <= hi)
                h = np.histogram(X[sel], HEDGE)[0]
                k = (name, p, lab)
                out["hist"][k] = out["hist"].get(k, 0) + h
        idx = np.nonzero(has)
        out["sig"].append(np.stack([np.full(len(idx[0]), p), T[idx]] + [m[n][idx] for n in m] + [cov[idx].astype(float)], 1))
        for n, X in m.items():
            k = (n, p)
            a = out["tot"].get(k, np.zeros(3))
            out["tot"][k] = a + np.array([X.sum(), X[has].sum(), T.sum()])
        c = out["cov"].get(p, np.zeros(4))
        out["cov"][p] = c + np.array([cov[has].sum(), has.sum(), cov[emp].sum(), emp.sum()])
    for mt in nmeta:
        if mt["kind"] != "bgn":
            continue
        g, w0, t0 = mt["g"], mt["w0"], mt["t0"]
        D = Dg[g]
        sub = D[w0 // 2: w0 // 2 + 16, t0 // 16: t0 // 16 + 8]
        out["bgn_dist"].append(int(sub.min()) if sub.size else -1)
    out["names"] = list(m)
    out["sig"] = np.concatenate(out["sig"]).astype(np.float32)
    return ev, out


def hist_quant(h, qs):
    c = np.cumsum(h) / max(h.sum(), 1)
    centers = np.concatenate([[-1.0, 0.0], np.sqrt(HEDGE[2:-2] * HEDGE[3:-1]), [HEDGE[-2]]])
    return [float(centers[min(np.searchsorted(c, q), len(centers) - 1)]) for q in qs]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cls-full", required=True)
    ap.add_argument("--cls", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    B = json.load(open(a.bench))
    picks = {f: {int(k): v for k, v in d.items()} for f, d in B["selection"]["pick"].items()}
    evs = list(range(260, 388))
    with Pool(a.workers) as pool:
        R = dict(pool.map(per_event, [(e, picks, a.cls_full, a.model) for e in evs]))
    names = R[evs[0]]["names"]
    res = {"picks": picks, "noise": {}, "fidelity": {}, "totals": {}, "coverage": {}}
    # ---- noise suppression
    for n in names:
        for p in range(3):
            for lab in DLAB + ("all",):
                labs = DLAB if lab == "all" else (lab,)
                h = sum(R[e]["hist"][(n, p, l_)] for e in evs for l_ in labs)
                N = int(h.sum())
                if not N:
                    continue
                above = lambda thr: float(h[np.searchsorted(HEDGE, thr, side="right"):].sum() / N)
                res["noise"][f"{n}|{'UVY'[p]}|{lab}"] = dict(
                    n=N, q50_90_99_999=hist_quant(h, (0.5, 0.9, 0.99, 0.999)),
                    frac_gt_005Q0=above(0.05 * Q0), frac_gt_025Q0=above(0.25 * Q0), frac_gt_1Q0=above(Q0),
                    count_gt_005Q0=int(round(above(0.05 * Q0) * N)), count_gt_1Q0=int(round(above(Q0) * N)),
                    frac_nonzero=float(1 - h[:2].sum() / N))
    # ---- fidelity
    S = np.concatenate([R[e]["sig"] for e in evs])
    P, Tq, cov = S[:, 0].astype(int), S[:, 1].astype(np.float64), S[:, -1] > 0
    for j, n in enumerate(names):
        X = S[:, 2 + j].astype(np.float64)
        for p in range(4):
            mp = (P == p) if p < 3 else np.ones(len(P), bool)
            rows = {}
            for lo, hi in zip(QBINS[:-1], QBINS[1:]):
                m = mp & (Tq > lo) & (Tq <= hi)
                if m.sum() < 20:
                    continue
                r = X[m] / Tq[m]
                p16, p50, p84 = np.percentile(r, [16, 50, 84])
                rows[f"{lo:g}-{hi:g}"] = dict(n=int(m.sum()), median_ratio=float(p50), hw68=float((p84 - p16) / 2),
                                              frac_zero=float(np.mean(X[m] <= 0)))
            res["fidelity"][f"{n}|{'UVYa'[p]}"] = rows
        for p in range(3):
            t = sum(R[e]["tot"][(n, p)] for e in evs)
            res["totals"][f"{n}|{'UVY'[p]}"] = dict(all_cells_over_truth=float(t[0] / t[2]), truth_cells_over_truth=float(t[1] / t[2]))
    for p in range(3):
        c = sum(R[e]["cov"][p] for e in evs)
        res["coverage"]["UVY"[p]] = dict(signal_cells_covered=float(c[0] / c[1]), empty_cells_covered=float(c[2] / c[3]))
    # ---- window-level aspects
    import score as SC
    A = np.load(SC.ANNOT)
    ann = {k: (float(s) if pk > 0 else 0.0, bool(fo <= 0.25 * q_))
           for k, s, pk, fo, q_ in zip(zip(A["ev"].tolist(), A["g"].tolist(), A["w0"].tolist(), A["t0"].tolist()),
                                       A["snr"], A["peak"], A["foreign"], A["q"])}
    M, wq, st, en, V = SC.load_split(evs, a.cls, a.model, ann)
    T = SC.Tab(M)
    wplane = np.repeat(T.g, en - st)
    comp = {}
    for fam in ("default", "tuned_q", "tuned_any"):
        v = np.zeros(len(wq))
        for p in range(3):
            v[wplane == p] = V[picks[fam][p]][wplane == p]
        comp[f"cls_{fam}"] = v
    comp["cls_noroi"] = V["g"]
    for m_ in SC.MODEL_STATS:
        comp[m_] = V[m_]
    bgnd = np.full(len(M), -1)
    j = 0
    for e in evs:
        dl = R[e]["bgn_dist"]; k = 0
        while j < len(M) and M[j]["ev"] == e:
            if M[j]["kind"] == "bgn":
                bgnd[j] = dl[k]; k += 1
            j += 1
    res["roc"], res["neardist"], res["dip"] = {}, {}, {}
    from helix.data.denoise import DenoiseEvents
    from helix.probe import resolution as RR
    cfg = DenoiseEvents("", [], "", items=[]).cfg
    toff = {i: cfg.toff[i] for i in range(3)}
    for n, v in comp.items():
        s = SC.win_scores(v, st, en)
        for p in range(3):
            mp = T.g == p
            bg, bn = s[mp & (T.kind == "bg")], s[mp & (T.kind == "bgn")]
            for i in (0, 1, 2):
                iso = s[mp & (T.kind == "iso") & (T.eb == i)]
                thr = np.unique(np.quantile(np.concatenate([bg, bn, iso]), np.linspace(0, 1, 301)))
                res["roc"][f"{n}|{'UVY'[p]}|{SC.EBN[i]}"] = dict(
                    far=[[float(np.mean(bg > t)), float(np.mean(iso > t))] for t in thr],
                    near=[[float(np.mean(bn > t)), float(np.mean(iso > t))] for t in thr])
            tf = np.quantile(bg, 0.99)
            for lo, hi in ((1, 2), (3, 4), (5, 8), (9, 16)):
                m = mp & (T.kind == "bgn") & (bgnd >= lo) & (bgnd <= hi)
                if m.sum() >= 10:
                    res["neardist"][f"{n}|{'UVY'[p]}|{lo}-{hi}"] = dict(n=int(m.sum()), fpr_at_far1pct=float(np.mean(s[m] > tf)))
        W = RR.window_stats(v, wq, M, st, en, cfg.pw, cfg.pt, cfg.delta[0], toff)
        for p in range(4):
            for ax, bins in RR.SEP_BINS.items():
                for b in bins:
                    k = f"{ax}<={b}"
                    d = np.array([w["dip"] for w, m_ in zip(W, M) if w.get("sep") == k and (p == 3 or m_["g"] % 3 == p)])
                    dt_ = np.array([w["dip_truth"] for w, m_ in zip(W, M) if w.get("sep") == k and (p == 3 or m_["g"] % 3 == p)])
                    if len(d) >= 5:
                        res["dip"][f"{n}|{'UVYa'[p]}|{k}"] = dict(n=len(d), median_dip=float(np.median(d)),
                                                                  frac_resolved=float(np.mean(d < 0.5)),
                                                                  truth_median_dip=float(np.median(dt_)))
    json.dump(res, open(os.path.join(a.out, "aspects.json"), "w"), default=float)
    np.save("/pscratch/sd/o/oalter/helix_work/uboone_sp/v2/aspects_sigcells.npy", S)
    print("wrote", os.path.join(a.out, "aspects.json"), "names", names)


if __name__ == "__main__":
    main()
