#!/usr/bin/env python3
"""Eval metrics of M6d / M6dc / M4n on the STORED corpus and on fresh noise draws, with
event-bootstrap AND draw-to-draw spread.

    scripts/helix_run.sh python baselines/uboone_sp/dense/draw_metrics.py \
        --draws 0,1,2,3,4 --out baselines/uboone_sp/dense/results/draws.json

Sources: 'stored' = the corpus coefficients (dense3/model for M6d/M6dc, dense2/model for
M4n); 'd<k>' = draws/model_d<k> (model_dump_cn.py --noise tn --seed-offset k: the
corpus's torch-GPU generator path, today's code, fresh realisation).
Detection (eval_denoise / v2/score.py windows, a window scores the max over its cells):
  near pp   1%-FPR threshold per plane on the near-activity charge-free windows, pooled
  near pool one threshold over all planes (eval_denoise's near_eff1pct)
  far pp    threshold on the floor's bg windows
  energy bins <0.1, 0.1-0.2, 0.2-0.5 MeV; SNR bins (clean deposits, noise_vs_hits)
  scores: presence ('p') and the gated charge map ('g' = q where p > 0.5)
Charge (raw charge head 'q' and gated 'g'):
  cells     truth > 16k e-: median pred/true, 68% half-width (all planes; per plane)
  regions   16 x 8 cell tiles with truth > 64k e-: sum ratio, median, half-width
  particles deposited E > 1 MeV, majority cells: median pred/true, half-width
  haze      empty cells within +-8 cells of charge: fraction > 100 e-, > 1 Q0
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, os.path.dirname(os.path.dirname(BASE)))
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"
ANNOT = "/pscratch/sd/o/oalter/helix_work/denoise/noise_vs_hits.npz"
PTRUTH = "/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/ptruth"
CFS = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
CLS = (f"{CFS}/dense2/cls_fix", "/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/cls_fit")
MODELS = ("M6d", "M6dc", "M4n")
EVS = list(range(260, 388))
Q0 = 2043.258
REDGE = np.concatenate([[-1e-9, 1e-9], np.logspace(-3, 2, 1001)])
B = 200


def src_file(src, model, ev):
    if src == "stored":
        return f"{CFS}/{'dense2' if model == 'M4n' else 'dense3'}/model/ev{ev:03d}.npz"
    return f"{CFS}/draws/model_{src}/ev{ev:03d}.npz"


def per_event(args):
    ev, sources = args
    from helix.probe.resolution import unkey
    from scipy.ndimage import maximum_filter
    z = np.load(os.path.join(TRUTH, f"ev{ev:03d}.npz"), allow_pickle=True)
    zn = np.load(os.path.join(NEAR, f"ev{ev:03d}.npz"), allow_pickle=True)
    T = {g: np.load(next(os.path.join(d, f"ev{ev:03d}_g{g}.npz") for d in CLS if os.path.exists(os.path.join(d, f"ev{ev:03d}_g{g}.npz"))))["truth"].astype(np.float64)
         for g in range(6)}
    PT = np.load(os.path.join(PTRUTH, f"ev{ev:03d}.npz"))
    E_of = dict(zip(PT["trk_id"].tolist(), PT["trk_E"].tolist()))
    out = {"win": {}, "cells": {}, "reg": {}, "par": {}, "haze": {}}
    wins = []
    for srcz in (z, zn):
        mm = json.loads(str(srcz["meta"]))
        if not mm:
            continue
        keys, wid = srcz["wkey"], srcz["wid"]
        starts = np.searchsorted(wid, np.arange(len(mm))); ends = np.append(starts[1:], len(wid))
        wins.append((mm, keys, starts, ends))
    meta = [dict(kind=m["kind"], p=m["g"] % 3, g=m["g"], w0=m["w0"], t0=m["t0"], E=float(m["E"][0]) if m["kind"] == "iso" else -1)
            for mm, _, _, _ in wins for m in mm]
    out["meta"] = meta
    for src in sources:
        for model in MODELS:
            f = src_file(src, model, ev)
            if not os.path.exists(f):
                continue
            Mz = np.load(f)
            if f"p_{model}_g0" not in Mz.files:
                continue
            to = int(Mz["toff"])
            Q = {g: Mz[f"q_{model}_g{g}"] for g in range(6)}; P = {g: Mz[f"p_{model}_g{g}"] for g in range(6)}
            tag = f"{src}|{model}"
            sc = {"p": [], "g": []}
            for mm, keys, starts, ends in wins:
                g_, fw, ft = unkey(keys)
                vp = np.zeros(len(keys)); vq = np.zeros(len(keys))
                for gg in range(6):
                    m = (g_ == gg) & (fw < P[gg].shape[0]) & (ft + to >= 0) & (ft + to < P[gg].shape[1])
                    vp[m] = P[gg][fw[m], ft[m] + to]; vq[m] = Q[gg][fw[m], ft[m] + to]
                vg = np.where(vp > 0.5, vq, 0.0)
                for k, v in (("p", vp), ("g", vg)):
                    s = np.maximum.reduceat(v, starts) if len(v) else np.zeros(len(mm))
                    s[ends == starts] = 0
                    sc[k].append(s)
            out["win"][tag] = {k: np.concatenate(v).astype(np.float32) for k, v in sc.items()}
            cells = {"q": [], "g": [], "pl": []}
            reg = {"q": [], "g": [], "T": [], "pl": []}
            par = {"q": [], "g": [], "pl": []}
            hz = np.zeros((2, 3, 3))                       # (q,g) x plane x [n_empty_near, >100 e-, >1Q0]
            for gg in range(6):
                Tg = T[gg]
                crop = lambda X: X[:Tg.shape[0], to:to + Tg.shape[1]].astype(np.float64)
                q, p = crop(Q[gg]), crop(P[gg])
                gq = np.where(p > 0.5, q, 0.0)
                pl = gg % 3
                m16 = Tg > 16e3
                for k, X in (("q", q), ("g", gq)):
                    cells[k].append((X[m16] / Tg[m16]).astype(np.float32))
                cells["pl"].append(np.full(m16.sum(), pl, np.int8))
                nw, nt = (Tg.shape[0] // 16) * 16, (Tg.shape[1] // 8) * 8
                til = lambda X: X[:nw, :nt].reshape(nw // 16, 16, nt // 8, 8).sum((1, 3)).ravel()
                tT = til(Tg); sel = tT > 64e3
                reg["T"].append(tT[sel].astype(np.float32)); reg["pl"].append(np.full(sel.sum(), pl, np.int8))
                for k, X in (("q", q), ("g", gq)):
                    reg[k].append(til(X)[sel].astype(np.float32))
                # particles
                cells_ = PT[f"c_{gg}"]; tmax = PT[f"tmax_{gg}"]
                tid, inv = np.unique(tmax, return_inverse=True)
                Tf = Tg.ravel()[cells_]
                tm = np.bincount(inv, weights=Tf)
                Ep = np.array([E_of.get(t, 0.0) for t in tid])
                keep = (Ep > 1.0) & (tm > Q0)
                for k, X in (("q", q), ("g", gq)):
                    par[k].append((np.bincount(inv, weights=X.ravel()[cells_]) / np.maximum(tm, 1e-9))[keep].astype(np.float32))
                par["pl"].append(np.full(keep.sum(), pl, np.int8))
                near = (maximum_filter((Tg > 0).astype(np.uint8), size=17) > 0) & (Tg <= 0)
                for i, X in enumerate((q, gq)):
                    x = X[near]
                    hz[i, pl] += [len(x), (x > 100).sum(), (x > Q0).sum()]
            out["cells"][tag] = {k: np.concatenate(v) for k, v in cells.items()}
            out["reg"][tag] = {k: np.concatenate(v) for k, v in reg.items()}
            out["par"][tag] = {k: np.concatenate(v) for k, v in par.items()}
            out["haze"][tag] = hz
    return ev, out


def hist_q(H, qs):
    c = np.cumsum(H, -1); tot = c[..., -1:]
    res = []
    for q in qs:
        k = np.minimum((c < q * tot).sum(-1), H.shape[-1] - 1)
        lo, hi = REDGE[k], REDGE[k + 1]
        cb = np.take_along_axis(c, k[..., None], -1)[..., 0]; hb = np.take_along_axis(H, k[..., None], -1)[..., 0]
        f = np.clip((q * tot[..., 0] - (cb - hb)) / np.maximum(hb, 1e-12), 0, 1)
        res.append(lo + f * (hi - lo))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draws", default="0,1,2,3,4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    sources = ["stored"] + [f"d{k}" for k in a.draws.split(",") if k != ""]
    with Pool(a.workers) as pool:
        RES = dict(pool.map(per_event, [(e, sources) for e in EVS]))
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(EVS), len(EVS)), minlength=len(EVS)) for _ in range(B)]).astype(float)
    meta = [m for e in EVS for m in RES[e]["meta"]]
    evw = np.array([i for i, e in enumerate(EVS) for _ in RES[e]["meta"]])
    kind = np.array([m["kind"] for m in meta]); p = np.array([m["p"] for m in meta]); E = np.array([m["E"] for m in meta])
    A = np.load(ANNOT)
    ann = {k: (float(s) if pk > 0 else 0.0, bool(fo <= 0.25 * q_)) for k, s, pk, fo, q_ in
           zip(zip(A["ev"].tolist(), A["g"].tolist(), A["w0"].tolist(), A["t0"].tolist()), A["snr"], A["peak"], A["foreign"], A["q"])}
    snr = np.array([ann.get((EVS[evw[i]], m["g"], m["w0"], m["t0"]), (np.nan, False))[0] if m["kind"] == "iso" else np.nan for i, m in enumerate(meta)])
    cln = np.array([ann.get((EVS[evw[i]], m["g"], m["w0"], m["t0"]), (np.nan, False))[1] if m["kind"] == "iso" else False for i, m in enumerate(meta)])
    tags = sorted(set(t for e in EVS for t in RES[e]["win"]))
    res = {"sources": sources, "metrics": {}}

    def eff_boot(det, sel):
        num = np.bincount(evw[sel], weights=det[sel].astype(float), minlength=len(EVS)); den = np.bincount(evw[sel], minlength=len(EVS)).astype(float)
        b = (W @ num) / np.maximum(W @ den, 1)
        return float(num.sum() / max(den.sum(), 1)), b

    for tag in tags:
        M = {}
        for sk in ("p", "g"):
            s = np.concatenate([RES[e]["win"][tag][sk] for e in EVS])
            thr = {p_: np.quantile(s[(kind == "bgn") & (p == p_)], 0.99) for p_ in range(3)}
            thrf = {p_: np.quantile(s[(kind == "bg") & (p == p_)], 0.99) for p_ in range(3)}
            tpool = np.quantile(s[kind == "bgn"], 0.99)
            dn = s > np.array([thr[x] for x in p]); df = s > np.array([thrf[x] for x in p]); dpool = s > tpool
            for lab, lo, hi in (("<0.1", 0, 0.1), ("0.1-0.2", 0.1, 0.2), ("0.2-0.5", 0.2, 0.5)):
                eb = (kind == "iso") & (E > lo) & (E <= hi)
                for pl in (0, 1, 2, 3):
                    sel = eb & ((p == pl) if pl < 3 else True)
                    M[f"{sk}|near_pp|{lab}|{'UVYa'[pl]}"] = eff_boot(dn, sel)
                    if lab == "0.1-0.2":
                        M[f"{sk}|far_pp|{lab}|{'UVYa'[pl]}"] = eff_boot(df, sel)
                M[f"{sk}|near_pool|{lab}|a"] = eff_boot(dpool, eb)
            for lo, hi, lab in ((3, 5, "3-5"), (5, 7, "5-7"), (7, 10, "7-10"), (10, 15, "10-15")):
                sel = (kind == "iso") & cln & (snr >= lo) & (snr < hi)
                M[f"{sk}|near_pp_snr|{lab}|a"] = eff_boot(dn, sel)
            for pl in range(3):
                M[f"{sk}|near_thr|{'UVY'[pl]}"] = (float(thr[pl]), None)
        for kq in ("q", "g"):
            ev_c = np.concatenate([np.full(len(RES[e]["cells"][tag][kq]), i) for i, e in enumerate(EVS)])
            x = np.concatenate([RES[e]["cells"][tag][kq] for e in EVS]); plc = np.concatenate([RES[e]["cells"][tag]["pl"] for e in EVS])
            for pl in (0, 1, 2, 3):
                sel = (plc == pl) if pl < 3 else np.ones(len(x), bool)
                kb = np.clip(np.searchsorted(REDGE, x[sel], side="right") - 1, 0, len(REDGE) - 2)
                H = np.bincount(ev_c[sel] * (len(REDGE) - 1) + kb, minlength=len(EVS) * (len(REDGE) - 1)).reshape(len(EVS), -1).astype(float)
                med = hist_q(H.sum(0), [0.5])[0]; lo_, hi_ = hist_q(H.sum(0), [0.16, 0.84])
                Hb = W @ H
                M[f"{kq}|cell16k_median|{'UVYa'[pl]}"] = (float(med), hist_q(Hb, [0.5])[0])
                l2, h2 = hist_q(Hb, [0.16, 0.84])
                M[f"{kq}|cell16k_hw68|{'UVYa'[pl]}"] = (float((hi_ - lo_) / 2), (h2 - l2) / 2)
            # regions
            ev_r = np.concatenate([np.full(len(RES[e]["reg"][tag]["T"]), i) for i, e in enumerate(EVS)])
            Tr = np.concatenate([RES[e]["reg"][tag]["T"] for e in EVS]).astype(float); Xr = np.concatenate([RES[e]["reg"][tag][kq] for e in EVS]).astype(float)
            sx, st = np.bincount(ev_r, weights=Xr, minlength=len(EVS)), np.bincount(ev_r, weights=Tr, minlength=len(EVS))
            M[f"{kq}|region64k_sum"] = (float(sx.sum() / st.sum()), (W @ sx) / (W @ st))
            r = Xr / Tr
            kb = np.clip(np.searchsorted(REDGE, r, side="right") - 1, 0, len(REDGE) - 2)
            H = np.bincount(ev_r * (len(REDGE) - 1) + kb, minlength=len(EVS) * (len(REDGE) - 1)).reshape(len(EVS), -1).astype(float)
            lo_, hi_ = hist_q(H.sum(0), [0.16, 0.84]); l2, h2 = hist_q(W @ H, [0.16, 0.84])
            M[f"{kq}|region64k_hw68"] = (float((hi_ - lo_) / 2), (h2 - l2) / 2)
            # particles
            ev_p = np.concatenate([np.full(len(RES[e]["par"][tag][kq]), i) for i, e in enumerate(EVS)])
            xp = np.concatenate([RES[e]["par"][tag][kq] for e in EVS])
            kb = np.clip(np.searchsorted(REDGE, xp, side="right") - 1, 0, len(REDGE) - 2)
            H = np.bincount(ev_p * (len(REDGE) - 1) + kb, minlength=len(EVS) * (len(REDGE) - 1)).reshape(len(EVS), -1).astype(float)
            med = hist_q(H.sum(0), [0.5])[0]; lo_, hi_ = hist_q(H.sum(0), [0.16, 0.84])
            Hb = W @ H; l2, h2 = hist_q(Hb, [0.16, 0.84])
            M[f"{kq}|particle1MeV_median"] = (float(med), hist_q(Hb, [0.5])[0])
            M[f"{kq}|particle1MeV_hw68"] = (float((hi_ - lo_) / 2), (h2 - l2) / 2)
            # haze
            i_ = 0 if kq == "q" else 1
            hz = np.stack([RES[e]["haze"][tag][i_] for e in EVS])     # (nev, 3, 3)
            for pl in (0, 1, 2, 3):
                h = hz[:, pl] if pl < 3 else hz.sum(1)
                M[f"{kq}|haze100|{'UVYa'[pl]}"] = (float(h[:, 1].sum() / h[:, 0].sum()), (W @ h[:, 1]) / (W @ h[:, 0]))
                M[f"{kq}|hazeQ0|{'UVYa'[pl]}"] = (float(h[:, 2].sum() / h[:, 0].sum()), (W @ h[:, 2]) / (W @ h[:, 0]))
        res["metrics"][tag] = {k: dict(central=v[0], boot=(None if v[1] is None else [float(x) for x in np.asarray(v[1])])) for k, v in M.items()}
    # stored vs draws: per model, per metric
    cmp_ = {}
    for model in MODELS:
        st = res["metrics"].get(f"stored|{model}")
        dr = [res["metrics"][f"{s}|{model}"] for s in sources[1:] if f"{s}|{model}" in res["metrics"]]
        if not st or not dr:
            continue
        for k in st:
            sv = st[k]["central"]; dv = np.array([d[k]["central"] for d in dr])
            sb = np.array(st[k]["boot"]) if st[k]["boot"] else None
            db = np.array([d[k]["boot"] for d in dr]) if dr[0][k]["boot"] else None
            row = dict(stored=sv, draws=dv.tolist(), draw_mean=float(dv.mean()), draw_sd=float(dv.std(ddof=1)) if len(dv) > 1 else None,
                       diff=float(sv - dv.mean()))
            if sb is not None:
                # paired event bootstrap of stored - mean(draws) (same events in each resample)
                pb = sb - db.mean(0)
                row["diff_boot16_84"] = [float(np.percentile(pb, 16)), float(np.percentile(pb, 84))]
                row["event_boot_sd_diff"] = float(np.std(pb))
                row["event_boot_sd_single"] = float(np.std(sb))
                # z: stored vs draw distribution; draw-to-draw sd already contains the noise variance of one draw
                sd = row["draw_sd"] if row["draw_sd"] else np.nan
                row["z_vs_draws"] = float((sv - dv.mean()) / (sd * np.sqrt(1 + 1 / len(dv)))) if sd and sd > 0 else None
            cmp_[f"{model}|{k}"] = row
    res["stored_vs_draws"] = cmp_
    json.dump(res, open(a.out, "w"), indent=0)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
