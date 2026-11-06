#!/usr/bin/env python3
"""Near-activity noise windows and false positives, for every rung and the model.

    scripts/helix_run.sh python baselines/uboone_sp/score_near.py \
        --preds /pscratch/sd/o/oalter/helix_work/uboone_sp/preds_near --out baselines/uboone_sp/results

Exactly as scripts/eval_denoise.py --near --annot: the floor's windows plus the
sidecar's near-activity noise windows (kind 'bgn': charge-free, with charge within
16 wires / 128 ticks; truth_v2_near), iso windows annotated with the deposit's
optimal SNR and isolation from noise_vs_hits.npz, then
helix.probe.resolution.window_stats + scalars + near_scalars, 200-resample event
bootstrap (seed 0, 16-84%). The model's numbers come from its own near-scored
windows file (dn_near_debug_ab_ft1024_win_trained_windows.json) through the same
scalars/near_scalars and the same bootstrap.

Also: 'cells > f x Q0 where the truth is 0' summed over all test events, per
plane type, f = 0.05 / 0.25 / 1 -- classical rungs over every cell of every plane,
the model over every token-covered cell (it predicts 0 elsewhere).
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
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"
ANNOT = "/pscratch/sd/o/oalter/helix_work/denoise/noise_vs_hits.npz"
MODEL_W = "/pscratch/sd/o/oalter/helix_work/denoise/results/dn_near_debug_ab_ft1024_win_trained_windows.json"


def stats(args):
    """Point + bootstrap of the floor keys (scalars) and near_scalars over windows W."""
    from helix.probe import resolution as R
    tag, W, nboot = args
    def keys(Ws):
        s = R.scalars(np.zeros(2), np.array([0.0, 1.0]), [], Ws, 1.0)
        out = {k: v for k, v in s.items() if k.startswith(("floor_eff", "floor_auc"))}
        out.update(R.near_scalars(Ws))
        return out
    point = keys(W)
    evs = sorted({w["ev"] for w in W})
    by = {v: [w for w in W if w["ev"] == v] for v in evs}
    rng = np.random.default_rng(0)
    boots = [keys([w for v in rng.choice(evs, len(evs), replace=True) for w in by[v]]) for _ in range(nboot)]
    r = {}
    for k, v in point.items():
        bv = np.array([b.get(k, np.nan) for b in boots], float)
        lo, hi = np.nanpercentile(bv, [16, 84]) if np.isfinite(bv).any() else (np.nan, np.nan)
        r[k] = [round(v, 4), round(float(lo), 4), round(float(hi), 4)]
    bn = np.array([w["score"] for w in W if w["kind"] == "bgn"]); bg = np.array([w["score"] for w in W if w["kind"] == "bg"])
    r["bg_far_q99"] = round(float(np.quantile(bg, 0.99)), 1); r["bg_near_q99"] = round(float(np.quantile(bn, 0.99)), 1)
    return tag, r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", required=True)
    ap.add_argument("--model-fp", default="/pscratch/sd/o/oalter/helix_work/uboone_sp/model_fp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from helix.data.denoise import DenoiseEvents
    from helix.probe import resolution as R
    cfg = DenoiseEvents("", [], "", items=[]).cfg
    toff = {i: cfg.toff[i] for i in range(3)}

    pdirs = a.preds.split(",")
    evs = sorted(int(os.path.basename(f)[2:5]) for f in glob.glob(os.path.join(pdirs[0], "ev*.npz"))
                 if all(os.path.exists(os.path.join(d, os.path.basename(f))) for d in pdirs))

    def load(ev):
        out, planes_ = {}, {}
        for d in pdirs:
            pz = np.load(os.path.join(d, f"ev{ev:03d}.npz"))
            for k in pz.files:
                if k == "planes":
                    for g, info in json.loads(str(pz[k])).items():
                        planes_.setdefault(g, {}).update(info)
                else:
                    out[k] = pz[k]
        out["planes"] = planes_
        return out

    first = load(evs[0])
    rungs = [r for r in ORDER if f"n_{r}" in first]
    A = np.load(ANNOT)
    ann = {k: (float(s) if pk > 0 else 0.0, bool(fo <= 0.25 * q_))
           for k, s, pk, fo, q_ in zip(zip(A["ev"].tolist(), A["g"].tolist(), A["w0"].tolist(), A["t0"].tolist()),
                                       A["snr"], A["peak"], A["foreign"], A["q"])}
    META, WQ, WID, WP = [], [], [], {r: [] for r in rungs}
    planes = []
    for ev in evs:
        pz = load(ev)
        for src, pre in ((TRUTH, "w_"), (NEAR, "n_")):
            z = np.load(os.path.join(src, f"ev{ev:03d}.npz"), allow_pickle=True)
            if not len(z["wkey"]):
                continue
            base = len(META)
            mm = json.loads(str(z["meta"]))
            for m_ in mm:
                m_["ev"] = ev
            META += mm; WQ.append(z["wq"]); WID.append(z["wid"] + base)
            for r in rungs:
                WP[r].append(pz[f"{pre}{r}"].astype(np.float64))
        for g, info in pz["planes"].items():
            info["g"] = int(g); planes.append(info)
    for m_ in META:
        k_ = (m_["ev"], m_["g"], m_["w0"], m_["t0"])
        if m_["kind"] == "iso" and k_ in ann:
            m_["snr"], m_["clean"] = ann[k_]
    wq, wid = np.concatenate(WQ), np.concatenate(WID)
    starts = np.searchsorted(wid, np.arange(len(META))); ends = np.append(starts[1:], len(wid))

    # the model's own near-scored windows must be OUR windows in the same order
    MW = json.load(open(MODEL_W))
    same = len(MW) == len(META) and all((w["ev"], w["g"], w["w0"], w["t0"], w["kind"]) == (m_["ev"], m_["g"], m_["w0"], m_["t0"], m_["kind"])
                                        for w, m_ in zip(MW, META))
    print(f"{len(evs)} events, {len(META)} windows ({sum(m['kind'] == 'bgn' for m in META)} near); model windows identical: {same}", flush=True)

    jobs = []
    for r in rungs:
        W = R.window_stats(np.concatenate(WP[r]), wq, META, starts, ends, cfg.pw, cfg.pt, cfg.delta[0], toff)
        for w_, m_ in zip(W, META):
            w_.update({k_: m_[k_] for k_ in ("snr", "clean") if k_ in m_})
        jobs.append((r, W, a.boot))
    jobs.append(("model", MW, a.boot))
    res = {}
    with Pool(min(a.workers, len(jobs))) as pool:
        for tag, r in pool.imap_unordered(stats, jobs):
            res[tag] = dict(metrics=r)
            print(f"{tag:8s} near_fpr_at_far1pct {r.get('near_fpr_at_far1pct', [np.nan])[0]:.3f} near_eff1pct <0.1 "
                  f"{r.get('near_eff1pct_0-0.1', [np.nan])[0]:.3f} 0.1-0.2 {r.get('near_eff1pct_0.1-0.2', [np.nan])[0]:.3f}", flush=True)

    # false positives: cells above f x Q0 where truth = 0, summed over events, per plane type
    fr = (0.05, 0.25, 1.0)
    for r in rungs:
        res[r]["fp_cells"] = {f"{f:g}Q0": {"UVY"[p]: int(sum(x[r]["fp_cells"][i] for x in planes if x["g"] % 3 == p))
                                           for p in range(3)} for i, f in enumerate(fr)}
        res[r]["n_cells"] = {"UVY"[p]: int(sum(x[r]["n_cells"] for x in planes if x["g"] % 3 == p)) for p in range(3)}
    mf = [np.load(os.path.join(a.model_fp, f"ev{ev:03d}.npz")) for ev in evs if os.path.exists(os.path.join(a.model_fp, f"ev{ev:03d}.npz"))]
    if len(mf) == len(evs):
        fp = sum(m["fp"] for m in mf); nc = sum(m["n_covered"] for m in mf)
        res["model"]["fp_cells"] = {f"{f:g}Q0": {"UVY"[p]: int(fp[i][p] + fp[i][p + 3]) for p in range(3)} for i, f in enumerate(fr)}
        res["model"]["n_cells"] = {"UVY"[p]: int(nc[p] + nc[p + 3]) for p in range(3)}
        res["model"]["n_cells_note"] = "token-covered cells only (the model predicts 0 elsewhere)"
    else:
        print(f"model fp counts for {len(mf)}/{len(evs)} events only -- not reported")
    res["_meta"] = dict(n_events=len(evs), n_windows=len(META), n_near=int(sum(m["kind"] == "bgn" for m in META)),
                        model_windows_identical=same, rungs=rungs)
    with open(os.path.join(a.out, "near.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    # ---- markdown
    names = rungs + ["model"]
    L = [f"Events: {len(evs)}, {len(META)} windows incl. {res['_meta']['n_near']} near-activity noise windows; "
         f"model windows identical to ours: {same}.\n",
         "### Near-activity windows (helix.probe.resolution.near_scalars; 16-84% event bootstrap in brackets)\n",
         "| rung | near_fpr_at_far1pct | near_eff1pct <0.1 | near_eff1pct 0.1-0.2 | near_eff1pct 0.2-0.5 | near_eff1pct 0.5-1 | far / near bg q99 (e-) |",
         "|---|---|---|---|---|---|---|"]
    def v(r, k, ci=True):
        x = res[r]["metrics"].get(k)
        if x is None:
            return "-"
        return f"{x[0]:.3f} ({x[1]:.3f}-{x[2]:.3f})" if ci and x[1] == x[1] else f"{x[0]:.3f}"
    for r in names:
        m = res[r]["metrics"]
        L.append(f"| {r} | {v(r, 'near_fpr_at_far1pct')} | {v(r, 'near_eff1pct_0-0.1')} | {v(r, 'near_eff1pct_0.1-0.2')} | "
                 f"{v(r, 'near_eff1pct_0.2-0.5')} | {v(r, 'near_eff1pct_0.5-1', False)} | {m['bg_far_q99']:.0f} / {m['bg_near_q99']:.0f} |")
    sb = [k[len("snr_eff1pct_far_"):] for k in res["model"]["metrics"] if k.startswith("snr_eff1pct_far_")]
    L += ["\n### Efficiency by deposit SNR (clean isolated deposits), at the far / near 1%-FPR threshold\n",
          "| rung | " + " | ".join(f"SNR {b}" for b in sb) + " |", "|---|" + "---|" * len(sb)]
    for r in names:
        m = res[r]["metrics"]
        L.append(f"| {r} | " + " | ".join(f"{m[f'snr_eff1pct_far_{b}'][0]:.2f} / {m[f'snr_eff1pct_near_{b}'][0]:.2f}"
                                           if f"snr_eff1pct_far_{b}" in m else "-" for b in sb) + " |")
    L += ["\n### Cells above f x Q0 where the truth is 0 (all test events; U, V, Y)\n",
          "| rung | > 0.05 Q0 (102 e-) | > 0.25 Q0 (511 e-) | > 1 Q0 (2043 e-) | cells counted |", "|---|---|---|---|---|"]
    for r in names:
        if "fp_cells" not in res[r]:
            continue
        f = lambda k: ", ".join(f"{res[r]['fp_cells'][k][p]:,}" for p in "UVY")
        nc = ", ".join(f"{res[r]['n_cells'][p]:,}" for p in "UVY") + (" (token-covered)" if r == "model" else "")
        L.append(f"| {r} | {f('0.05Q0')} | {f('0.25Q0')} | {f('1Q0')} | {nc} |")
    open(os.path.join(a.out, "near.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
