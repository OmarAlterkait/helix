#!/usr/bin/env python3
"""Figures for the classical ladder.

    scripts/helix_run.sh python baselines/uboone_sp/plot_ladder.py --preds <preds> --model-cells <model> \
        --results baselines/uboone_sp/results --out baselines/uboone_sp/figs [--event 300]

  images_*    truth | R0 | R1 | R2 | R3 | our model on the evaluation's fine grid
              (2 wires x 16 ticks), one plane: full plane and zooms on a faint
              isolated deposit (0.1-0.2 MeV) and on a charge-free noise window.
              Truth, every rung and the model share ONE LogNorm in electrons
              per cell, floored at 0.05 Q0; cells below are drawn in the 'under'
              colour so faint is not confused with empty.
  response    the example's kernel vs the one fitted to our sim
  floor       window score vs deposit energy (R3, R1, model) with each
              predictor's 1%-FPR background threshold
  summary     the ladder's headline metrics next to the model
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

SHOW = ("R0", "R1", "R2", "R3")
LAB = {"R0": "R0  clean + oracle ROI", "R1": "R1  +incoherent noise", "R2": "R2  +coherent, gate removal",
       "R3": "R3  +MicroBooNE ROIs (full chain)", "model": "our model (ab_ft1024_win)", "truth": "TRUE charge (hits)"}


def plane_images(mi, E, bg, cz, floor, cmap, Q0, a, plt, LogNorm, R, fine_cells):
    g = int(mi["g"])
    mb = next((m for m in bg if m["g"] == g), None)
    full = np.load(os.path.join(a.preds, f"full_ev{a.event:03d}_g{g}.npz"))
    T = fine_cells(full["truth"])
    maps = {"truth": T}
    for r in SHOW:
        maps[r] = cz[f"{r}_{g}"]
    mz_path = os.path.join(a.model_cells, f"ev{a.event:03d}.npz")
    if os.path.exists(mz_path) and "cells" in np.load(mz_path).files:
        mz = np.load(mz_path)
        cg, cw, ct = R.unkey(mz["cells"])
        P = np.zeros_like(T)
        on = (cg == g) & (cw < T.shape[0]) & (ct >= 0) & (ct < T.shape[1])
        P[cw[on], ct[on]] = Q0 * np.expm1(np.maximum(mz["cells_p"][on], 0))
        maps["model"] = P
    keys = ["truth"] + list(SHOW) + (["model"] if "model" in maps else [])
    vmax = np.quantile(T[T > 0], 0.999)
    norm = LogNorm(vmin=floor, vmax=vmax)

    def panels(sw, st, title, fn, boxes):
        n = len(keys)
        fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(7 * ((n + 1) // 2), 11), sharex=True, sharey=True)
        ax = ax.ravel()
        ext = [st.start * R.FT, st.stop * R.FT, sw.stop * R.FW, sw.start * R.FW]
        for i, k in enumerate(keys):
            X = maps[k][sw, st]
            im = ax[i].imshow(np.where(X > 0, X, np.nan), aspect="auto", extent=ext, cmap=cmap, norm=norm,
                              interpolation="nearest")
            fp = int(np.sum((maps["truth"][sw, st] == 0) & (X > floor))) if k != "truth" else None
            ax[i].set_title(LAB[k] + (f"\n{fp:,} cells > 0.05 Q0 where truth = 0" if fp is not None else ""), fontsize=11)
            ax[i].set_facecolor("#1b1b2f")
            for (bw, bt, c) in boxes:
                ax[i].add_patch(plt.Rectangle((bt, bw), R.WT, R.WW, fill=False, ec=c, lw=1.5))
        for j in range(len(keys), len(ax)):
            ax[j].axis("off")
        for x in ax[-((n + 1) // 2):]:
            x.set_xlabel("tick")
        for x in ax[:: (n + 1) // 2]:
            x.set_ylabel("wire")
        fig.colorbar(im, ax=ax.tolist(), fraction=0.02, label="charge per cell (e-), shared log scale")
        fig.suptitle(title, fontsize=13)
        fig.savefig(os.path.join(a.out, fn), dpi=100, bbox_inches="tight"); plt.close(fig)
        print("wrote", fn, flush=True)

    nwc, ntc = T.shape
    boxes = [(mi["w0"], mi["t0"], "cyan")] + ([(mb["w0"], mb["t0"], "orange")] if mb else [])
    panels(slice(0, nwc), slice(0, ntc), f"ev{a.event} plane {g} ({'UVY'[g % 3]}), 2-wire x 16-tick cells; cyan: faint "
           f"isolated deposit {np.sum(mi['E']):.3f} MeV, orange: a charge-free window", f"images_ev{a.event}_g{g}_full.png", boxes)
    for m, lab in ((mi, "faint"), (mb, "noise")):
        if m is None:
            continue
        w0, t0 = m["w0"] // R.FW, m["t0"] // R.FT
        sw = slice(max(w0 - 24, 0), min(w0 + R.WW // R.FW + 24, nwc))
        st = slice(max(t0 - 24, 0), min(t0 + R.WT // R.FT + 24, ntc))
        sc = {k: maps[k][w0:w0 + R.WW // R.FW, t0:t0 + R.WT // R.FT].max() for k in keys}
        panels(sw, st, f"ev{a.event} plane {g}: zoom on the {lab} window "
               f"({'E=%.3f MeV' % np.sum(m['E']) if lab == 'faint' else 'no truth within +-16 w / +-128 t'}); "
               "window score (max cell): " + ", ".join(f"{k} {v:.0f}" for k, v in sc.items()),
               f"images_ev{a.event}_g{g}_zoom_{lab}.png", [(m["w0"], m["t0"], "cyan" if lab == "faint" else "orange")])



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", default="/pscratch/sd/o/oalter/helix_work/uboone_sp/preds")
    ap.add_argument("--model-cells", default="/pscratch/sd/o/oalter/helix_work/uboone_sp/model")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    ap.add_argument("--event", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from helix.probe import resolution as R
    from sp.data import Event
    from sp.pipeline import fine_cells
    from sp.response import load_kernel

    res = json.load(open(os.path.join(a.results, "ladder.json")))
    Q0 = res["_meta"]["Q0"]

    # ---------------- images: the first faint isolated deposit on a collection plane and on an induction plane
    E = Event(a.event)
    iso_all = [m for m in E.meta if m["kind"] == "iso" and 0.1 <= float(np.sum(m["E"])) < 0.2]
    bg = [m for m in E.meta if m["kind"] == "bg"]
    picks = [next(m for m in iso_all if m["g"] % 3 == 2), next(m for m in iso_all if m["g"] % 3 != 2)]
    cz = np.load(os.path.join(a.preds, f"cells_ev{a.event:03d}.npz"))
    floor = 0.05 * Q0
    cmap = plt.get_cmap("viridis").copy(); cmap.set_under("#1b1b2f"); cmap.set_bad("#1b1b2f")
    for mi in picks:
        plane_images(mi, E, bg, cz, floor, cmap, Q0, a, plt, LogNorm, R, fine_cells)

    # ---------------- response
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for p, pl in enumerate("UVY"):
        Re, woe, tze = load_kernel(pl, "example")
        Rf, wof, tzf = load_kernel(pl, "fit")
        for dw, c in ((0, "k"), (1, "tab:blue"), (2, "tab:orange")):
            je, jf = int(np.nonzero(woe == dw)[0][0]), int(np.nonzero(wof == dw)[0][0])
            ax[0, p].plot(np.arange(Re.shape[0]) - tze, 1000 * Re[:, je], c=c, ls="--", lw=1, label=f"example, wire {dw:+d}")
            ax[0, p].plot(np.arange(Rf.shape[0]) - tzf, 1000 * Rf[:, jf], c=c, lw=1.5, label=f"fitted, wire {dw:+d}")
        ax[0, p].set_xlim(-40, 40); ax[0, p].set_xlabel("tick relative to the hit's tick")
        ax[0, p].set_ylabel("ADC per 1000 e-"); ax[0, p].set_title(f"{pl}: response per wire offset"); ax[0, p].legend(fontsize=8)
        v = np.abs(Rf).max() * 1000
        im = ax[1, p].imshow(1000 * Rf.T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest",
                             extent=[-tzf - 0.5, Rf.shape[0] - tzf - 0.5, wof[-1] + 0.5, wof[0] - 0.5])
        ax[1, p].set_xlim(-60, 40); ax[1, p].set_xlabel("tick"); ax[1, p].set_ylabel("wire offset")
        ax[1, p].set_title(f"{pl}: fitted 2D kernel (ADC per 1000 e-)")
        fig.colorbar(im, ax=ax[1, p])
    rj = os.path.join(HERE, "response", "residuals.json")
    if os.path.exists(rj):
        s = json.load(open(rj))["summary"]
        fig.suptitle("held-out ||clean - store(hits (*) R)|| / ||clean||:  " + ";  ".join(
            f"{pl} example {s[pl]['resid_example']:.3f}, fitted {s[pl]['resid_fit']:.3f}" for pl in "UVY"))
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "response.png"), dpi=100); plt.close(fig)
    print("wrote response.png")

    # ---------------- floor: window score vs energy
    preds = [r for r in ("R1", "R3", "model_cpu") if os.path.exists(os.path.join(a.results, f"windows_{r}.json"))]
    fig, ax = plt.subplots(1, len(preds), figsize=(6.5 * len(preds), 5.5), sharey=True)
    ax = np.atleast_1d(ax)
    for x, r in zip(ax, preds):
        W = json.load(open(os.path.join(a.results, f"windows_{r}.json")))
        thr = np.quantile([w["score"] for w in W if w["kind"] == "bg"], 0.99)
        # energies from the truth meta; windows_*.json is in the same (event, window) order
        meta = []
        for ev in sorted(set(w["ev"] for w in W)):
            meta += json.loads(str(np.load(os.path.join(res["_meta"].get("truth", "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"),
                                                        f"ev{ev:03d}.npz"), allow_pickle=True)["meta"]))
        assert all((m["g"], m["w0"], m["t0"]) == (w["g"], w["w0"], w["t0"]) for m, w in zip(meta, W))
        e_iso = np.array([float(np.sum(m["E"])) for m, w in zip(meta, W) if w["kind"] == "iso"])
        s_iso = np.array([w["score"] for w in W if w["kind"] == "iso"])
        s_bg = np.array([w["score"] for w in W if w["kind"] == "bg"])
        x.scatter(e_iso, np.maximum(s_iso, 1), s=3, alpha=0.3, label="isolated deposits")
        x.axhline(max(thr, 1), c="r", lw=1, label=f"1%-FPR threshold ({thr:.0f} e-)")
        x.scatter(np.full(len(s_bg), 0.004) * np.exp(np.random.default_rng(0).normal(0, 0.15, len(s_bg))),
                  np.maximum(s_bg, 1), s=3, alpha=0.3, c="gray", label="background windows (at x=0.004)")
        x.set_xscale("log"); x.set_yscale("log"); x.set_xlabel("deposit energy (MeV)")
        x.set_title({"R1": LAB["R1"], "R3": LAB["R3"], "model_cpu": LAB["model"]}[r]); x.legend(fontsize=8)
        for b in (0.1, 0.2, 0.5):
            x.axvline(b, c="k", lw=0.5, ls=":")
    ax[0].set_ylabel("window score = max cell charge (e-), floored at 1")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "floor_scores.png"), dpi=100); plt.close(fig)
    print("wrote floor_scores.png")

    # ---------------- summary
    rows = [r for r in ("R0syn", "R0zs", "R0", "R1", "R2", "R1nr", "R2nr", "R2nrp", "R3") if r in res] + \
        (["model_dnjsonl"] if "model_dnjsonl" in res else [])
    keys2 = [("floor_eff1pct_0-0.1", "eff@1% <0.1"), ("floor_eff1pct_0.1-0.2", "eff@1% 0.1-0.2"),
             ("floor_eff1pct_0.2-0.5", "eff@1% 0.2-0.5"), ("floor_auc_0-0.1", "AUC <0.1"), ("map_r", "map r")]
    fig, ax = plt.subplots(1, len(keys2), figsize=(4.2 * len(keys2), 5.5), sharey=True)
    for x, (k, lab) in zip(ax, keys2):
        v = [res[r]["metrics_covered"][k] for r in rows]
        mid = [t[0] for t in v]; err = [[max(t[0] - t[1], 0) for t in v], [max(t[2] - t[0], 0) for t in v]]
        x.barh(range(len(rows)), mid, xerr=err, color=["tab:gray"] * (len(rows) - 1) + ["tab:green"]
               if rows[-1] == "model_dnjsonl" else "tab:gray")
        x.set_title(lab); x.set_xlim(0, 1.0 if k != "map_r" else 1.0)
        for i, m_ in enumerate(mid):
            x.text(min(m_, 0.98), i, f" {m_:.3f}", va="center", fontsize=8)
            if k.startswith("floor") and rows[i] in ("R0syn", "R0zs", "R0", "R1", "R2"):
                x.patches[i].set_hatch("//"); x.patches[i].set_facecolor("lightgray")
    ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels([r if r != "model_dnjsonl" else "our model" for r in rows]); ax[0].invert_yaxis()
    fig.suptitle("classical ladder vs our model, identical 128 events / windows / rows (bars: 16-84% event bootstrap); "
                 "hatched = oracle-ROI rung: background windows score exactly 0, so floor 'efficiency' is charge survival, not detection",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "summary.png"), dpi=100); plt.close(fig)
    print("wrote summary.png")


if __name__ == "__main__":
    main()
