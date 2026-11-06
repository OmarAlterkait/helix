#!/usr/bin/env python3
"""Summarise roi_diag.py: stage-by-stage fake-ROI accounting, noise RMS, collection misses.

    scripts/helix_run.sh python baselines/uboone_sp/summarize_roi_diag.py --diag <roi_diag dir> --out results/roi_diag.md
"""
import argparse
import glob
import json
import os

import numpy as np

ANNOT = "/pscratch/sd/o/oalter/helix_work/denoise/noise_vs_hits.npz"
EB = (0.0, 0.1, 0.2, 0.5, 1.0, 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.diag, "ev*.json")))]
    A = np.load(ANNOT)
    snr = {k: (float(s) if pk > 0 else 0.0) for k, s, pk in
           zip(zip(A["ev"].tolist(), A["g"].tolist(), A["w0"].tolist(), A["t0"].tolist()), A["snr"], A["peak"])}
    L = [f"{len(recs)} events (truth_v2 ev{recs[0]['ev']}-ev{recs[-1]['ev']}); R3's input "
         "(digitize(clean + incoherent + coherent) -> gate), the run_ladder noise realisation.\n"]

    # ---- noise RMS
    L += ["### Noise RMS per filtered domain (median over wires; e- per tick)\n",
          "| plane | domain | true noise std (noise-only plane) | robust RMS of noise-only | RMS the example uses (signal+noise, robust) | used / true (p10 / p50 / p90 over wires) | threshold (e-) | threshold / true std | per-sample crossing rate, noise alone (Gaussian expectation) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p, pl in enumerate("UVY"):
        for dom in ("wiener", "tight", "loose", "gauss"):
            rows = [r["planes"][str(g)]["rms"].get(dom) for r in recs for g in (p, p + 3)]
            rows = [x for x in rows if x]
            if not rows:
                continue
            f = lambda k: np.median([x[k] for x in rows])
            u = np.median([x["used_over_true_p10_p50_p90"] for x in rows], 0)
            th = (f"{f('threshold_median_e'):.0f} | {f('threshold_over_true_std'):.2f} | "
                  f"{np.mean([x['noise_crossing_rate_per_sample'] for x in rows]):.2e} "
                  f"({rows[0]['noise_crossing_rate_gauss_expect']:.2e})") if "threshold_median_e" in rows[0] else "- | - | -"
            L.append(f"| {pl} | {dom} | {f('true_std_median'):.0f} | {f('robust_noise_median'):.0f} | {f('used_median_all'):.0f} | "
                     f"{u[0]:.2f} / {u[1]:.2f} / {u[2]:.2f} | {th} |")

    # ---- stage accounting
    L += ["\n### Stage-by-stage: fraction of windows with any ROI pixel (U, V pooled per plane type)\n",
          "| plane | stage | far bg | near bgn | iso <0.1 | iso 0.1-0.2 | iso 0.2-0.5 | iso >0.5 | ROI pixel fraction | ROIs on charge-free wires / event-plane |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p, pl in enumerate("UVY"):
        stages = list(recs[0]["planes"][str(p)]["flags"])
        for s in stages:
            bg, bn, iso, eb, pix, nq = [], [], [], [], [], []
            for r in recs:
                for g in (p, p + 3):
                    P = r["planes"][str(g)]
                    F = P["flags"][s]
                    bg += F["bg"]; bn += F["bgn"]; iso += F["iso"]
                    Es = [x["E"] for x in P["isos"]]
                    # isos list and the iso window flags are in the same order only if every iso had a box;
                    # take E from the truth meta order instead (wins['iso'] order == meta order)
                    eb += Es if len(Es) == len(F["iso"]) else [np.nan] * len(F["iso"])
                    pix.append(F["pixel_frac"]); nq.append(F["n_rois_quiet_wires"])
            iso, eb = np.array(iso, float), np.array(eb)
            cells = []
            for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1e9)):
                m = (eb >= lo) & (eb < hi)
                cells.append(f"{iso[m].mean():.3f}" if m.any() else "-")
            L.append(f"| {pl} | {s} | {np.mean(bg):.3f} | {np.mean(bn):.3f} | " + " | ".join(cells) +
                     f" | {np.mean(pix):.4f} | {np.mean(nq):.0f} |")

    # ---- deposit peak vs threshold
    L += ["\n### Isolated deposits: noise-free peak in the thresholded domain vs the threshold on its wire\n",
          "(induction: tight-LF Wiener output, |value|; collection: Wiener output)\n",
          "| plane | energy bin | n | median peak (e-) | median threshold (e-) | median noise std (e-) | peak/threshold p25 / p50 / p75 | frac peak > threshold |",
          "|---|---|---|---|---|---|---|---|"]
    for p, pl in enumerate("UVY"):
        X = [dict(x, ev=r["ev"], g=g) for r in recs for g in (p, p + 3) for x in r["planes"][str(g)]["isos"]]
        E = np.array([x["E"] for x in X]); pk = np.array([x["peak"] for x in X]); th = np.array([x["thr"] for x in X])
        ns = np.array([x["noise_std"] for x in X])
        for i in range(4):
            m = (E >= EB[i]) & (E < EB[i + 1])
            if m.sum() < 5:
                continue
            r_ = pk[m] / th[m]
            L.append(f"| {pl} | {EB[i]:g}-{EB[i + 1]:g} | {m.sum()} | {np.median(pk[m]):.0f} | {np.median(th[m]):.0f} | {np.median(ns[m]):.0f} | "
                     f"{np.percentile(r_, 25):.2f} / {np.median(r_):.2f} / {np.percentile(r_, 75):.2f} | {np.mean(r_ > 1):.2f} |")
        s_ = np.array([snr.get((x["ev"], x["g"], x["w0"], x["t0"]), np.nan) for x in X])
        for lo, hi in ((0, 1e-9), (1e-9, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 1e9)):
            m = (s_ >= lo) & (s_ < hi)
            if m.sum() >= 5:
                L.append(f"| {pl} | SNR {lo:g}-{hi:g} | {m.sum()} | {np.median(pk[m]):.0f} | {np.median(th[m]):.0f} | {np.median(ns[m]):.0f} | "
                         f"{np.percentile(pk[m] / th[m], 25):.2f} / {np.median(pk[m] / th[m]):.2f} / {np.percentile(pk[m] / th[m], 75):.2f} | {np.mean(pk[m] > th[m]):.2f} |")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
