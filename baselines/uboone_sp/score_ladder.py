#!/usr/bin/env python3
"""Score the classical ladder exactly as scripts/eval_denoise.py scores the denoiser.

    scripts/helix_run.sh python baselines/uboone_sp/score_ladder.py \
        --preds /pscratch/sd/o/oalter/helix_work/uboone_sp/preds --out baselines/uboone_sp/results

Same events (truth_v2 ev260-ev387), same map rows, same windows, same Q0, same
helix.probe.resolution.window_stats / scalars and the same event bootstrap
(200 resamples, seed 0, 16-84%). The per-cell prediction is the classical
chain's charge in the cell (electrons, >= 0); as for the model, map rows are
scored as log1p(q/Q0) and windows as charge.

Map rows: eval_denoise scores only rows some token covers (the model predicts
nothing elsewhere); ``covered`` reproduces that subset (helix.data.denoise
tokens + helix.probe.resolution.cell_inputs), so map_r is over IDENTICAL rows.
``all`` (every map row) is reported alongside.

Beyond eval_denoise's keys: per-cell charge resolution (map rows with truth
charge, by truth charge bin), total-charge recovery per plane type, and the
example's own metrics (F0 vs filtered truth, per-wire charge).
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

ORDER = ("R0syn", "R0zs", "R0", "R0ex", "R1", "R2", "R2mp", "R3", "R3inc", "R0synnr", "R0zsnr", "R0nr", "R1nr", "R2nr", "R0nrp", "R1nrp", "R2nrp")
QBINS = (0, 1000, 4000, 16000, 64000, 1e12)          # electrons per fine cell
MODEL_RES = "/pscratch/sd/o/oalter/helix_work/denoise/results"


def coverage(job):
    """okm for one event: which map rows a token covers (eval_denoise's ok mask)."""
    f, cache = job
    if os.path.exists(cache):
        return f, np.load(cache)["ok"]
    import h5py
    from helix.data.denoise import DenoiseEvents
    from helix.probe import resolution as R
    ds = DenoiseEvents("", [], "", items=[])
    z = np.load(f, allow_pickle=True)
    shard, event = str(z["shard"]), int(z["event"])
    with h5py.File(shard, "r") as fh:
        pos = int(np.searchsorted(fh["ident"]["event"][:], event))
    _, B, bl = ds.tokens(shard, pos)
    idx, _ = R.cell_inputs(z["mrows"], B["cell_key"], bl, ds.cfg)
    ok = (idx >= 0).any(1)
    np.savez(cache, ok=ok)
    return f, ok


def tokenizer_cfg():
    from helix.data.denoise import DenoiseEvents
    return DenoiseEvents("", [], "", items=[]).cfg


def boot_scalars(args):
    """eval_denoise.score's point estimate + bootstrap, for one prediction."""
    from helix.probe import resolution as R
    tag, p, y, e, pl, wscore, wq, META, starts, ends, cfgd, q30, Q0, nboot = args
    W = R.window_stats(wscore, wq, META, starts, ends, cfgd["pw"], cfgd["pt"], cfgd["delta0"], cfgd["toff"])
    kz = {}
    for k in np.unique(e * 10 + pl):
        m = (e * 10 + pl) == k
        if m.sum() > 20 and y[m].std() > 0 and p[m].std() > 0:
            kz[int(k)] = float(np.arctanh(np.clip(np.corrcoef(p[m], y[m])[0, 1], -0.999, 0.999)))
    q = np.expm1(y) * Q0
    evs = np.unique(e)
    rows_of = {v: np.nonzero(e == v)[0] for v in evs}
    wins_of = {v: [w for w in W if w["ev"] == v] for v in evs}
    z_of = {v: [z_ for k, z_ in kz.items() if k // 10 == v] for v in evs}
    point = R.scalars(p, q, list(kz.values()), W, q30)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(nboot):
        sm = rng.choice(evs, len(evs), replace=True)
        idx = np.concatenate([rows_of[v] for v in sm])
        boots.append(R.scalars(p[idx], q[idx], [z_ for v in sm for z_ in z_of[v]], [w for v in sm for w in wins_of[v]], q30))
    r = {"val_mse": float(np.mean((p - y) ** 2))}
    for k, v in point.items():
        bv = np.array([b_[k] for b_ in boots], float)
        lo, hi = np.nanpercentile(bv, [16, 84]) if np.isfinite(bv).any() else (np.nan, np.nan)
        r[k] = [round(v, 4), round(float(lo), 4), round(float(hi), 4)]
    # diagnostic (NOT an eval_denoise key): floor per plane type, each with its OWN
    # plane's background threshold; and the background windows' score quantiles
    r["per_plane"] = {}
    for pp in range(3):
        Wp = [w for w, m_ in zip(W, META) if int(m_["g"]) % 3 == pp]
        s = R.scalars(np.zeros(2), np.array([0.0, 1.0]), [], Wp, 1.0)
        bgp = np.array([w["score"] for w in Wp if w["kind"] == "bg"])
        r["per_plane"]["UVY"[pp]] = {k: round(v, 4) for k, v in s.items() if k.startswith(("floor_eff", "floor_auc"))}
        r["per_plane"]["UVY"[pp]]["bg_q50_q99"] = [round(float(x), 1) for x in np.quantile(bgp, [0.5, 0.99])]
    bg = np.array([w["score"] for w in W if w["kind"] == "bg"])
    r["bg_frac_zero"] = round(float(np.mean(bg == 0)), 4)
    r["bg_q50_q99"] = [round(float(x), 1) for x in np.quantile(bg, [0.5, 0.99])]
    return tag, r, W


def cell_resolution(qpred, q):
    """Per-cell charge resolution on cells with truth charge, by truth-charge bin:
    median pred/true, 68% half-width of (pred - true)/true, fraction predicted 0."""
    out = {}
    for lo, hi in zip(QBINS[:-1], QBINS[1:]):
        m = (q > lo) & (q <= hi) if lo > 0 else (q > 0) & (q <= hi)
        if m.sum() < 20:
            continue
        r = qpred[m] / q[m]
        d = r - 1
        nm = f"{lo / 1000:g}-{hi / 1000:g}k" if hi < 1e11 else f">{lo / 1000:g}k"
        out[nm] = dict(n=int(m.sum()), median_ratio=round(float(np.median(r)), 4),
                       hw68=round(float((np.percentile(d, 84) - np.percentile(d, 16)) / 2), 4),
                       zero_frac=round(float(np.mean(qpred[m] <= 0)), 4))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", required=True)
    ap.add_argument("--truth", default="/pscratch/sd/o/oalter/helix_work/resolution/truth_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--train-events", type=int, default=260)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model-tag", default="ab_ft1024_win")
    ap.add_argument("--model-cells", default="/pscratch/sd/o/oalter/helix_work/uboone_sp/model",
                    help="model_cells.py output: the model's per-cell predictions (CPU replay)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lo, hi = (int(x) for x in a.events.split("-"))

    files = sorted(glob.glob(os.path.join(a.truth, "ev*.npz")))
    qs = np.concatenate([np.load(f, allow_pickle=True)["mq"] for f in files if int(os.path.basename(f)[2:5]) < a.train_events])
    Q0 = float(np.median(qs[qs > 0]))
    model_rows = [json.loads(l) for l in open(os.path.join(MODEL_RES, "dn.jsonl")) if l.strip()]
    model = next(r for r in model_rows if r["tag"] == a.model_tag)
    print(f"Q0 {Q0} (model trained with {model['Q0']}; equal: {abs(Q0 - model['Q0']) < 1e-6 * Q0})", flush=True)

    pdirs = a.preds.split(",")
    test = [f for f in files if lo <= int(os.path.basename(f)[2:5]) <= hi
            and all(os.path.exists(os.path.join(d, os.path.basename(f))) for d in pdirs)]

    def load_preds(f):
        """Merge every preds dir's file for this event into one dict."""
        out, planes_ = {}, {}
        for d in pdirs:
            pz = np.load(os.path.join(d, os.path.basename(f)))
            for k in pz.files:
                if k == "planes":
                    for g, info in json.loads(str(pz[k])).items():
                        planes_.setdefault(g, {}).update(info)
                else:
                    out[k] = pz[k]
        out["planes"] = planes_
        return out

    first = load_preds(test[0])
    RUNGS = tuple(r for r in ORDER if f"m_{r}" in first)
    have_model = all(os.path.exists(os.path.join(a.model_cells, os.path.basename(f))) for f in test)
    names = RUNGS + (("model_cpu",) if have_model else ())
    print(f"model per-cell predictions for every event: {have_model}", flush=True)
    print(f"{len(test)} events with predictions", flush=True)
    cov_dir = os.path.join(pdirs[0], "coverage"); os.makedirs(cov_dir, exist_ok=True)
    with Pool(a.workers) as pool:
        okm = dict(pool.map(coverage, [(f, os.path.join(cov_dir, os.path.basename(f))) for f in test]))

    from helix.probe import resolution as R
    cfg = tokenizer_cfg()
    cfgd = dict(pw=cfg.pw, pt=cfg.pt, delta0=cfg.delta[0], toff={i: cfg.toff[i] for i in range(3)})

    rows = {s: dict(Y=[], E=[], PL=[], P={r: [] for r in names}, Q={r: [] for r in names}) for s in ("covered", "all")}
    WQ, WID, META, WP = [], [], [], {r: [] for r in names}
    planes, wires = [], {}
    for f in test:
        ei = int(os.path.basename(f)[2:5])
        z = np.load(f, allow_pickle=True)
        pz = load_preds(f)
        y = np.log1p(z["mq"] / Q0).astype(np.float32)
        pg = (R.unkey(z["mrows"])[0] % 3).astype(np.int8)
        # per predictor: score-space map value p, charge per cell q, window charge
        pr = {r: (np.log1p(pz[f"m_{r}"].astype(np.float64) / Q0), pz[f"m_{r}"].astype(np.float64),
                  pz[f"w_{r}"].astype(np.float64)) for r in RUNGS}
        if have_model:
            mz = np.load(os.path.join(a.model_cells, os.path.basename(f)))
            if not np.array_equal(mz["ok_m"], okm[f]):
                raise SystemExit(f"{f}: model coverage differs from the coverage mask")
            pm = mz["p_m"].astype(np.float64)
            pr["model_cpu"] = (pm, np.expm1(np.maximum(pm, 0)) * Q0, np.expm1(np.maximum(mz["p_w"].astype(np.float64), 0)) * Q0)
        for s, m in (("covered", okm[f]), ("all", np.ones(len(y), bool))):
            rows[s]["Y"].append(y[m]); rows[s]["E"].append(np.full(m.sum(), ei)); rows[s]["PL"].append(pg[m])
            for r in names:
                rows[s]["P"][r].append(pr[r][0][m]); rows[s]["Q"][r].append(pr[r][1][m])
        if len(z["wkey"]):
            base = len(META)
            mm = json.loads(str(z["meta"]))
            for m_ in mm:
                m_["ev"] = ei
            META += mm; WQ.append(z["wq"]); WID.append(z["wid"] + base)
            for r in names:
                WP[r].append(pr[r][2])
        for g, info in pz["planes"].items():
            info["g"] = int(g); info["ev"] = ei; planes.append(info)
        for k in pz:
            if k[:1] in "RT" and "_" in k and not k.startswith(("m_", "w_")):
                wires.setdefault(k.split("_")[0] + "_" + str(int(k.split("_")[1]) % 3), []).append(pz[k])

    wq, wid = np.concatenate(WQ), np.concatenate(WID)
    starts = np.searchsorted(wid, np.arange(len(META))); ends = np.append(starts[1:], len(wid))

    # --- the model's windows: same windows, same order? (validates the window pipeline)
    mw = json.load(open(os.path.join(MODEL_RES, f"dn_{a.model_tag}_trained_windows.json")))
    same = len(mw) == len(META) and all((w["ev"], w["g"], w["w0"], w["t0"], w["kind"]) == (m_["ev"], m_["g"], m_["w0"], m_["t0"], m_["kind"])
                                        for w, m_ in zip(mw, META))
    print(f"model windows file: {len(mw)} windows, identical (ev, g, w0, t0, kind) sequence to ours: {same}", flush=True)

    results = {}
    jobs = []
    for s in ("covered", "all"):
        y = np.concatenate(rows[s]["Y"]); e = np.concatenate(rows[s]["E"]); pl = np.concatenate(rows[s]["PL"])
        q = np.expm1(y) * Q0; q30 = np.quantile(q[q > 0], 0.3)
        for r in names:
            qp = np.concatenate(rows[s]["Q"][r])
            p = np.concatenate(rows[s]["P"][r])
            jobs.append((f"{r}|{s}", p, y, e, pl, np.concatenate(WP[r]), wq, META, starts, ends, cfgd, q30, Q0,
                         a.boot if s == "covered" else 0))
            results.setdefault(r, {})[f"cellres_{s}"] = cell_resolution(qp, q)
            results[r][f"cellres_{s}_byplane"] = {"UVY"[k]: cell_resolution(qp[pl == k], q[pl == k]) for k in range(3)}
    with Pool(min(a.workers, len(jobs))) as pool:
        for tag, r, W in pool.imap_unordered(boot_scalars, jobs):
            rung, s = tag.split("|")
            results[rung]["metrics_" + s] = r
            if s == "covered":
                with open(os.path.join(a.out, f"windows_{rung}.json"), "w") as fh:
                    json.dump([dict({k: w[k] for k in ("kind", "score", "eb", "loc", "dip", "edge", "sep") if k in w},
                                    ev=m_.get("ev"), g=m_.get("g"), w0=m_.get("w0"), t0=m_.get("t0"))
                               for w, m_ in zip(W, META)], fh, default=float)
            print(f"{rung:6s} {s:8s} map_r {r['map_r'][0]:.4f} floor_eff1pct <0.1 {r['floor_eff1pct_0-0.1'][0]:.3f} "
                  f"0.1-0.2 {r['floor_eff1pct_0.1-0.2'][0]:.3f} 0.2-0.5 {r['floor_eff1pct_0.2-0.5'][0]:.3f} "
                  f"auc<0.1 {r['floor_auc_0-0.1'][0]:.3f}", flush=True)

    # --- model: its windows rescored with OUR code -> must reproduce dn.jsonl
    from helix.probe.resolution import scalars
    mW = []
    for w in mw:
        mW.append(dict(w))
    chk = scalars(np.zeros(2), np.array([0.0, 1.0]), [], mW, 1.0)
    recheck = {k: (round(chk[k], 4), model["trained"][k][0]) for k in chk
               if k.startswith(("floor_eff", "floor_auc")) and np.isfinite(chk[k])}
    print("model windows rescored by our code vs dn.jsonl:", json.dumps(recheck), flush=True)

    # --- totals and the example's metrics, per plane type
    for r in RUNGS:
        tot = {}
        for k in range(3):
            P = [x for x in planes if x["g"] % 3 == k]
            H = sum(x["hits"] for x in P)
            tot["UVY"[k]] = dict(charge_over_hits=round(sum(x[r]["sum_charge"] for x in P) / H, 4),
                                 cells_over_hits=round(sum(x[r]["sum_cells"] for x in P) / H, 4),
                                 roi_frac=round(float(np.mean([x[r]["roi_frac"] for x in P])), 4),
                                 F0_f=round(float(np.nanmean([x[r]["F0_f"] for x in P])), 4),
                                 corr_f=round(float(np.nanmean([x[r]["corr_f"] for x in P])), 4))
            if f"{r}_{k}" in wires and len(wires[f"{r}_{k}"]) == len(wires[f"T_{k}"]):
                rw = np.concatenate(wires[f"{r}_{k}"]).astype(np.float64)
                tw = np.concatenate(wires[f"T_{k}"]).astype(np.float64)
                for cut in (500, 20000):
                    m = tw > cut
                    fe = (rw[m] - tw[m]) / tw[m]
                    tot["UVY"[k]][f"wire_q>{cut}"] = dict(
                        n=int(m.sum()), bias_mean=round(float(fe.mean()), 4), rms=round(float(np.sqrt((fe ** 2).mean())), 4),
                        median=round(float(np.median(fe)), 4), hw68=round(float((np.percentile(fe, 84) - np.percentile(fe, 16)) / 2), 4))
        results[r]["planes"] = tot
    results["_meta"] = dict(Q0=Q0, n_events=len(test), events=[int(os.path.basename(f)[2:5]) for f in test],
                            n_windows=len(META), model_tag=a.model_tag, model_windows_identical=same,
                            model_window_recheck=recheck,
                            n_rows_covered=int(sum(len(x) for x in rows["covered"]["Y"])),
                            n_rows_all=int(sum(len(x) for x in rows["all"]["Y"])),
                            gate_resid_rms=float(np.mean([x["resid_rms_gate"] for x in planes])),
                            mp_resid_rms=float(np.mean([x["resid_rms_mp"] for x in planes])),
                            truth_in_algo_roi={"UVY"[k]: float(np.mean([x["R3_truth_in_roi"] for x in planes if x["g"] % 3 == k])) for k in range(3)})
    results["model_dnjsonl"] = dict(tag=a.model_tag, metrics_covered=model["trained"])
    with open(os.path.join(a.out, "ladder.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print("wrote", os.path.join(a.out, "ladder.json"))


if __name__ == "__main__":
    main()
