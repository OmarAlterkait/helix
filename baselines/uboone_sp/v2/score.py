#!/usr/bin/env python3
"""Tune the classical detector on TUNING events, freeze it, score it on the TEST
events next to the model -- the corrected benchmark.

    scripts/helix_run.sh python baselines/uboone_sp/v2/score.py \
        --cls /pscratch/sd/o/oalter/helix_work/uboone_sp/v2/cls \
        --model /pscratch/sd/o/oalter/helix_work/uboone_sp/v2/model --out baselines/uboone_sp/v2/results

Windows / truth / annotations / window statistic exactly as scripts/eval_denoise.py
(--near --annot): floor windows (truth_v2) + near-activity noise windows
(truth_v2_near); a window scores the MAX over its 2x16 cells of the method's cell map.

Selection (tuning events ev000-095 only, their own noise): per plane type
(Wire-Cell planes are independent and MicroBooNE configures them per plane), the
candidate maximising J = mean(far, near) 1%-FPR efficiency at 0.1-0.2 MeV with
THAT plane's thresholds. Families:
  default     MicroBooNE configuration, ROI charge (U q_s1_f0 = master, V/Y q_s1_f1)
  tuned_q     best ROI-charge map q_<setting> (a charge output, like the model's)
  tuned_any   best of every statistic (ROI charge, ROI-gated significance,
              continuous significances with no ROI)
Reported on test events (all 128, event bootstrap 200x, 16-84%):
  * eval-standard keys (helix.probe.resolution.scalars / near_scalars: ONE pooled
    threshold over planes) -- only meaningful when the planes' scores share a scale;
  * 'pp' keys: pooled efficiency with a per-plane 1% threshold (every plane at 1%
    FPR; scale-free) -- the fair pooled number for mixed statistics, applied to the
    model as well;
  * per-plane efficiencies, SNR bins, signed centroid offsets (alignment).
Model rows: M4n / le3_ft_N64, charge and presence maps from v2/model_dump.py;
the reproduction of dn_near.jsonl per window is checked and written.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BASE))
sys.path.insert(0, BASE); sys.path.insert(0, ROOT)
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"
ANNOT = "/pscratch/sd/o/oalter/helix_work/denoise/noise_vs_hits.npz"
DNW = "/pscratch/sd/o/oalter/helix_work/denoise/results/dn_near_{}_trained_windows.json"
EBN = ("0-0.1", "0.1-0.2", "0.2-0.5", "0.5-1", "1-1e+09")
SNRB = ((0, 1e-9, "nosignal"), (1e-9, 3, "0-3"), (3, 5, "3-5"), (5, 7, "5-7"), (7, 10, "7-10"), (10, 15, "10-15"),
        (15, 1e9, "15-inf"))
TUNE = list(range(0, 96))
TEST = list(range(260, 388))
MODEL_STATS = ("M4n_q", "M4n_p", "N64_q", "N64_p")


def load_split(evs, cls_dir, model_dir, ann, extra_dirs=(), add_dirs=()):
    """Window table + per-statistic window-cell values for a list of events."""
    from sp.pipeline import lookup
    META, WQ, WID, VALS = [], [], [], {}
    for ev in evs:
        z = np.load(os.path.join(TRUTH, f"ev{ev:03d}.npz"), allow_pickle=True)
        zn = np.load(os.path.join(NEAR, f"ev{ev:03d}.npz"), allow_pickle=True)
        FS = [[np.load(os.path.join(d_, f"ev{ev:03d}_g{g}.npz")) for g in range(6)] for d_ in (cls_dir,) + tuple(add_dirs)]
        X = [dict((tag, [np.load(os.path.join(d_, f"ev{ev:03d}_g{g}.npz")) for g in range(6)]) for tag, d_ in extra_dirs)]
        Mz = np.load(os.path.join(model_dir, f"ev{ev:03d}.npz")) if model_dir else None
        for src, pre in ((z, "w_"), (zn, "n_")):
            base = len(META)
            mm = json.loads(str(src["meta"]))
            for m_ in mm:
                m_["ev"] = ev
                k_ = (ev, m_["g"], m_["w0"], m_["t0"])
                if m_["kind"] == "iso":
                    m_["eb"] = int(np.searchsorted((0.0, 0.1, 0.2, 0.5, 1.0, 1e9), m_["E"][0], side="right") - 1)
                    if k_ in ann:
                        m_["snr"], m_["clean"] = ann[k_]
            META += mm; WQ.append(src["wq"]); WID.append(src["wid"] + base)
            n = len(src["wkey"])
            for F in FS:                    # union over plane files: a plane without a statistic contributes 0
                names = set.union(*[set(k[len(pre):] for k in f.files if k.startswith(pre)) for f in F])
                for k in names:
                    VALS.setdefault(k, []).append(sum(F[g][pre + k].astype(np.float64) if pre + k in F[g].files
                                                      else np.zeros(n) for g in range(6)))
            for tag, FF in X[0].items():
                nm2 = set.union(*[set(k[len(pre):] for k in f.files if k.startswith(pre)) for f in FF])
                for k in nm2:
                    VALS.setdefault(f"{tag}:{k}", []).append(sum(FF[g][pre + k].astype(np.float64) if pre + k in FF[g].files
                                                                 else np.zeros(n) for g in range(6)))
            if Mz is not None:
                for st in MODEL_STATS:
                    tag, kind = st.split("_")
                    maps = {g: Mz[f"{kind}_{tag}_g{g}"] for g in range(6) if f"{kind}_{tag}_g{g}" in Mz.files}
                    if maps:
                        VALS.setdefault(st, []).append(model_lookup(maps, src["wkey"], int(Mz["toff"])) if n else np.zeros(0))
    nwin = len(META)
    VALS = {k: np.concatenate(v) for k, v in VALS.items() if len(v) == 2 * len(evs)}
    wq, wid = np.concatenate(WQ), np.concatenate(WID)
    starts = np.searchsorted(wid, np.arange(nwin)); ends = np.append(starts[1:], len(wid))
    return META, wq, starts, ends, VALS


def model_lookup(maps, keys, toff):
    """Per cell key, maps[g][fw, ft + toff] (model_dump.py layout), 0 outside."""
    from helix.probe.resolution import unkey
    g, fw, ft = unkey(np.asarray(keys, np.int64))
    out = np.zeros(len(g))
    for gg, X in maps.items():
        m = (g == gg) & (fw >= 0) & (fw < X.shape[0]) & (ft + toff >= 0) & (ft + toff < X.shape[1])
        out[m] = X[fw[m], ft[m] + toff]
    return out


def win_scores(v, starts, ends):
    s = np.maximum.reduceat(v, starts)
    s[ends == starts] = 0.0
    return s


class Tab:
    """Vectorised window table."""

    def __init__(self, META):
        self.kind = np.array([m["kind"] for m in META])
        self.g = np.array([m["g"] % 3 for m in META])
        self.ev = np.array([m["ev"] for m in META])
        self.eb = np.array([m.get("eb", -1) for m in META])
        self.snr = np.array([m.get("snr", np.nan) if m.get("clean") else np.nan for m in META], float)


def effs(T, s, sel=None, planes=(0, 1, 2)):
    """Per-plane-threshold efficiencies: dict with far/near, per plane, per energy bin."""
    out = {}
    m_all = np.ones(len(s), bool) if sel is None else sel
    dec = {"far": np.zeros(len(s), bool), "near": np.zeros(len(s), bool)}
    for p in planes:
        mp = m_all & (T.g == p)
        for lab, kd in (("far", "bg"), ("near", "bgn")):
            b = s[mp & (T.kind == kd)]
            thr = np.quantile(b, 0.99) if len(b) else np.inf
            iso = mp & (T.kind == "iso")
            dec[lab][iso] = s[iso] > thr
            for i in range(4):
                m = iso & (T.eb == i)
                if m.any():
                    out[f"{lab}_{'UVY'[p]}_{EBN[i]}"] = float(dec[lab][m].mean())
            for lo, hi, nm in SNRB:
                m = iso & (T.snr >= lo) & (T.snr < hi)
                if m.sum() >= 5:
                    out[f"{lab}_snr_{'UVY'[p]}_{nm}"] = float(dec[lab][m].mean())
    isoall = m_all & (T.kind == "iso")
    for lab in ("far", "near"):
        for i in range(4):
            m = isoall & (T.eb == i)
            if m.any():
                out[f"{lab}_pp_{EBN[i]}"] = float(dec[lab][m].mean())
        for lo, hi, nm in SNRB:
            m = isoall & (T.snr >= lo) & (T.snr < hi)
            if m.sum() >= 5:
                out[f"{lab}_pp_snr_{nm}"] = float(dec[lab][m].mean())
    # eval-standard: one pooled threshold
    for lab, kd in (("far", "bg"), ("near", "bgn")):
        b = s[m_all & (T.kind == kd)]
        thr = np.quantile(b, 0.99)
        for i in range(4):
            m = isoall & (T.eb == i)
            if m.any():
                out[f"{lab}_pool_{EBN[i]}"] = float((s[m] > thr).mean())
        for lo, hi, nm in SNRB:
            m = isoall & (T.snr >= lo) & (T.snr < hi)
            if m.sum() >= 5:
                out[f"{lab}_pool_snr_{nm}"] = float((s[m] > thr).mean())
    tf = np.quantile(s[m_all & (T.kind == "bg")], 0.99)
    out["near_fpr_at_far1pct_pool"] = float(np.mean(s[m_all & (T.kind == "bgn")] > tf))
    nf = []
    for p in planes:
        mp = m_all & (T.g == p)
        tfp = np.quantile(s[mp & (T.kind == "bg")], 0.99)
        nf.append(s[mp & (T.kind == "bgn")] > tfp)
    out["near_fpr_at_far1pct_pp"] = float(np.mean(np.concatenate(nf)))
    return out


def objective(T, s, p, eb=1):
    o = effs(T, s, planes=(p,))
    return 0.5 * (o.get(f"far_{'UVY'[p]}_{EBN[eb]}", 0) + o.get(f"near_{'UVY'[p]}_{EBN[eb]}", 0))


def compose(T, scores, pick):
    s = np.zeros(len(T.g))
    for p in range(3):
        m = T.g == p
        s[m] = scores[pick[p]][m]
    return s


def boot(T, s, nb=200, seed=0):
    point = effs(T, s)
    evs = np.unique(T.ev)
    idx_of = {e: np.nonzero(T.ev == e)[0] for e in evs}
    rng = np.random.default_rng(seed)
    B = []
    for _ in range(nb):
        sm = rng.choice(evs, len(evs), replace=True)
        ii = np.concatenate([idx_of[e] for e in sm])
        Tb = Tab.__new__(Tab)
        for k in ("kind", "g", "ev", "eb", "snr"):
            setattr(Tb, k, getattr(T, k)[ii])
        B.append(effs(Tb, s[ii]))
    out = {}
    for k, v in point.items():
        bv = np.array([b.get(k, np.nan) for b in B], float)
        lo, hi = np.nanpercentile(bv, [16, 84])
        out[k] = [round(v, 4), round(float(lo), 4), round(float(hi), 4)]
    return out


def offsets(META, wq, starts, ends, v, T):
    """Signed centroid offset (prediction - truth), wires / ticks, iso windows, per plane;
    eval_denoise's weighting (max(P - median(P), 0))."""
    gi, gj = np.meshgrid(np.arange(16), np.arange(8), indexing="ij")
    res = {}
    for p in range(3):
        dw, dt = [], []
        for n in np.nonzero((T.kind == "iso") & (T.g == p) & (T.eb >= 2))[0]:
            P = v[starts[n]:ends[n]].reshape(16, 8); Q = wq[starts[n]:ends[n]].reshape(16, 8)
            w = np.maximum(P - np.median(P), 0)
            if w.sum() > 0 and Q.sum() > 0:
                dw.append(((w * gi).sum() / w.sum() - (Q * gi).sum() / Q.sum()) * 2)
                dt.append(((w * gj).sum() / w.sum() - (Q * gj).sum() / Q.sum()) * 16)
        if dw:
            res["UVY"[p]] = dict(n=len(dw), dw_median=float(np.median(dw)), dt_median=float(np.median(dt)),
                                 dw_mean=float(np.mean(dw)), dt_mean=float(np.mean(dt)),
                                 dw_sem=float(np.std(dw) / np.sqrt(len(dw))), dt_sem=float(np.std(dt) / np.sqrt(len(dt))))
    return res


def eval_standard(META, wq, starts, ends, v, nb=200):
    """helix.probe.resolution scalars + near_scalars exactly as eval_denoise (windows only)."""
    from helix.data.denoise import DenoiseEvents
    from helix.probe import resolution as R
    cfg = DenoiseEvents("", [], "", items=[]).cfg
    toff = {i: cfg.toff[i] for i in range(3)}
    W = R.window_stats(v, wq, META, starts, ends, cfg.pw, cfg.pt, cfg.delta[0], toff)
    for w_, m_ in zip(W, META):
        w_.update({k: m_[k] for k in ("snr", "clean") if k in m_})
        w_["g"] = m_["g"]
    keys = lambda d: {k: x for k, x in d.items() if k.startswith(("floor_eff1pct", "floor_auc", "loc_", "sep_dip"))}
    point = keys(R.scalars(np.zeros(2), np.array([0.0, 1.0]), [], W, 1.0)); point.update(R.near_scalars(W))
    evs = sorted(set(w["ev"] for w in W)); wins_of = {e: [w for w in W if w["ev"] == e] for e in evs}
    rng = np.random.default_rng(0); B = []
    for _ in range(nb):
        sm = rng.choice(evs, len(evs), replace=True)
        wb = [w for e in sm for w in wins_of[e]]
        b = keys(R.scalars(np.zeros(2), np.array([0.0, 1.0]), [], wb, 1.0)); b.update(R.near_scalars(wb)); B.append(b)
    out = {}
    for k, x in point.items():
        bv = np.array([b.get(k, np.nan) for b in B], float)
        lo, hi = np.nanpercentile(bv, [16, 84]) if np.isfinite(bv).any() else (np.nan, np.nan)
        out[k] = [round(x, 4), round(float(lo), 4), round(float(hi), 4)]
    return out, W


def _job(args):
    name, v, kind = args
    if kind == "boot":
        return name, kind, boot(GT, win_scores(v, GS, GE))
    if kind == "std":
        r, W = eval_standard(GM, GWQ, GS, GE, v)
        return name, kind, (r, [w["score"] for w in W])
    return name, kind, offsets(GM, GWQ, GS, GE, v, GT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cls", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--extra", default="", help="tag=dir,... extra classical runs on test events (corpus input, example kernel)")
    ap.add_argument("--add", default="", help="comma-separated dirs of extra statistics (classical_extra.py), tune + test")
    ap.add_argument("--picks-from", default="", help="another benchmark.json: also score ITS frozen picks on this data")
    ap.add_argument("--corpus-tune", default="", help="classical.py --input corpus dir with tuning events 0-47: re-select on it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    A = np.load(ANNOT)
    ann = {k: (float(s) if pk > 0 else 0.0, bool(fo <= 0.25 * q_))
           for k, s, pk, fo, q_ in zip(zip(A["ev"].tolist(), A["g"].tolist(), A["w0"].tolist(), A["t0"].tolist()),
                                       A["snr"], A["peak"], A["foreign"], A["q"])}
    extra = [tuple(x.split("=", 1)) for x in a.extra.split(",") if x]

    # ---------------- selection on tuning events
    add = tuple(x for x in a.add.split(",") if x)
    Mt, wqt, st_, et_, Vt = load_split(TUNE, a.cls, a.model, ann, add_dirs=add)
    Tt = Tab(Mt)
    St = {k: win_scores(v, st_, et_) for k, v in Vt.items()}
    sel = {"candidates": {}, "pick": {}}
    cls_names = [k for k in St if k not in MODEL_STATS and k != "qor"]
    for p in range(3):
        rows = []
        for k in cls_names:
            if not np.any(St[k][Tt.g == p]):
                continue
            o = effs(Tt, St[k], planes=(p,))
            pl = "UVY"[p]
            rows.append(dict(stat=k, far=o.get(f"far_{pl}_0.1-0.2", 0), near=o.get(f"near_{pl}_0.1-0.2", 0),
                             far02=o.get(f"far_{pl}_0.2-0.5", 0), near02=o.get(f"near_{pl}_0.2-0.5", 0),
                             far01=o.get(f"far_{pl}_0-0.1", 0), near01=o.get(f"near_{pl}_0-0.1", 0)))
        for r in rows:
            r["J"] = 0.5 * (r["far"] + r["near"])
            r["J_near"] = r["near"]
            r["J_wide"] = np.mean([r[x] for x in ("far01", "near01", "far", "near", "far02", "near02")])
        rows.sort(key=lambda r: (-round(r["J"], 6), r["stat"]))         # deterministic tie-break: name
        sel["candidates"]["UVY"[p]] = rows
        q_rows = [r for r in rows if r["stat"].startswith("q_")]
        n_rows = [r for r in rows if not r["stat"].startswith(("q_", "zr_"))]
        sel["pick"].setdefault("tuned_noroi", {})[p] = n_rows[0]["stat"]
        sel["pick"].setdefault("tuned_any", {})[p] = rows[0]["stat"]
        sel["pick"].setdefault("tuned_q", {})[p] = q_rows[0]["stat"]
        sel["pick"].setdefault("tuned_any_nearobj", {})[p] = min(rows, key=lambda r: (-round(r["J_near"], 6), r["stat"]))["stat"]
        sel["pick"].setdefault("tuned_any_wideobj", {})[p] = min(rows, key=lambda r: (-round(r["J_wide"], 6), r["stat"]))["stat"]
    sel["pick"]["default"] = {0: "q_s1_f0", 1: "q_s1_f1", 2: "q_s1_f1"}
    sel["pick"]["default_nofake"] = {0: "q_s1_f0", 1: "q_s1_f0", 2: "q_s1_f0"}
    for m in MODEL_STATS:
        if m in St:
            sel.setdefault("model_on_tuning", {})[m] = effs(Tt, St[m])
    if a.corpus_tune:                      # the classical chain on the model's (sparsified) information, re-tuned on it
        Mc, _, sc_, ec_, Vc = load_split(list(range(0, 48)), a.corpus_tune, None, ann)
        Tc = Tab(Mc); Sc = {k: win_scores(v, sc_, ec_) for k, v in Vc.items() if k != "qor"}
        pk = {}
        for p in range(3):
            best = min(sorted(Sc), key=lambda k: -round(objective(Tc, Sc[k], p), 6) if np.any(Sc[k][Tc.g == p]) else 1)
            pk[p] = best
        sel["pick"]["corpus_retuned"] = pk
        sel["corpus_retuned_tuning"] = effs(Tc, compose(Tc, Sc, pk))
    print("picks:", json.dumps(sel["pick"]), flush=True)
    for fam in ("default", "tuned_q", "tuned_any"):
        print(fam, "tuning:", {k: round(v, 3) for k, v in effs(Tt, compose(Tt, St, sel["pick"][fam])).items()
                               if k.startswith(("far_pp_0.1", "near_pp_0.1"))}, flush=True)

    # ---------------- test events
    global GM, GWQ, GS, GE, GT
    GM, GWQ, GS, GE, V = load_split(TEST, a.cls, a.model, ann, extra, add)
    GT = Tab(GM)
    rows = {}
    for fam in [f for f in ("default", "default_nofake", "tuned_q", "tuned_any", "tuned_noroi", "tuned_any_nearobj",
                             "tuned_any_wideobj", "corpus_retuned") if f in sel["pick"]]:
        pick = sel["pick"][fam]
        v = np.zeros(len(GWQ))
        # compose per plane at the CELL level (windows are single-plane)
        wplane = np.repeat(GT.g, GE - GS)
        if all(pick[p] in V for p in range(3)):
            for p in range(3):
                v[wplane == p] = V[pick[p]][wplane == p]
            rows[fam] = v
        for tag, _ in extra:
            if all(f"{tag}:{pick[p]}" in V for p in range(3)):
                ve = np.zeros(len(GWQ))
                for p in range(3):
                    ve[wplane == p] = V[f"{tag}:{pick[p]}"][wplane == p]
                rows[f"{fam}@{tag}"] = ve
    if a.picks_from:                       # e.g. the fitted-kernel picks, frozen, on exact-kernel data
        other = json.load(open(a.picks_from))["selection"]["pick"]
        for fam in ("default", "tuned_q", "tuned_any", "tuned_noroi"):
            pk = {int(k): v for k, v in other[fam].items()}
            if all(pk[p] in V for p in range(3)):
                v = np.zeros(len(GWQ))
                for p in range(3):
                    v[wplane == p] = V[pk[p]][wplane == p]
                rows[f"frozen_{fam}"] = v
        sel["picks_from"] = {f: other[f] for f in ("default", "tuned_q", "tuned_any", "tuned_noroi")}
    for k in sorted(V):
        if not k.startswith(("q_", "zr_")) and ":" not in k and k not in MODEL_STATS:
            rows[f"stat_{k}"] = V[k]
    for m in MODEL_STATS:
        if m in V:
            rows[m] = V[m]
    jobs = [(n, v, kd) for n, v in rows.items() for kd in ("boot", "off")]
    jobs += [(n, v, "std") for n, v in rows.items() if n in ("default", "default_nofake", "tuned_q") or n in MODEL_STATS]
    res = {n: {} for n in rows}
    with Pool(min(a.workers, len(jobs))) as pool:
        for n, kd, r in pool.imap_unordered(_job, jobs):
            if kd == "std":
                res[n]["std"], res[n]["_wscores"] = r
            else:
                res[n][kd] = r
            print("done", n, kd, flush=True)
    # model reproduction vs dn_near.jsonl windows (per-window scores)
    repro = {}
    for m, tag in (("M4n_q", "M4n_ft_full_near"), ("M4n_p", "M4n_ft_full_near_pres"), ("N64_q", "le3_ft_N64"), ("N64_p", "le3_ft_N64_pres")):
        f = DNW.format(tag)
        if m in res and "_wscores" in res[m] and os.path.exists(f):
            ref = json.load(open(f))
            same = len(ref) == len(GM) and all((r["g"], r["w0"], r["t0"], r["kind"]) == (x["g"], x["w0"], x["t0"], x["kind"]) for r, x in zip(ref, GM))
            a_, b_ = np.array(res[m]["_wscores"]), np.array([r["score"] for r in ref])
            repro[m] = dict(windows_identical=bool(same), max_abs_diff=float(np.max(np.abs(a_ - b_))),
                            median_rel_diff=float(np.median(np.abs(a_ - b_) / np.maximum(np.abs(b_), 1e-9))),
                            frac_rel_diff_gt_1pct=float(np.mean(np.abs(a_ - b_) > 0.01 * np.maximum(np.abs(b_), 1e-9))))
    for n in res:
        res[n].pop("_wscores", None)
    out = dict(selection=sel, test=res, model_repro=repro,
               meta=dict(tune_events=[TUNE[0], TUNE[-1]], test_events=[TEST[0], TEST[-1]], n_test_windows=len(GM),
                         n_tune_windows=len(Mt)))
    json.dump(out, open(os.path.join(a.out, "benchmark.json"), "w"), default=float, indent=1)
    print("wrote", os.path.join(a.out, "benchmark.json"))


if __name__ == "__main__":
    main()
