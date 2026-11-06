#!/usr/bin/env python3
"""Figures for firstp.md from results/firstp.json: presence reliability (raw vs prior-corrected)
and linear-space calibration of the charge estimators in prediction bins (truth / prediction)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0"})
COL = {"raw": "#eb6834", "glob": "#eda100", "pdens": "#2a78d6"}
ECOL = {"raw": "#eb6834", "rawJ": "#eda100", "rawP": "#1baf7a", "rawPJ": "#2a78d6", "gated": "#8a8984", "hurdle": "#1baf7a", "hurdleJ": "#2a78d6"}
QL = ["0-25", "25-102", "102-500", "0.5-1k", "1-2k", "2-4k", "4-8k", "8-16k", "16-64k", ">64k"]


def main():
    R = json.load(open(os.path.join(HERE, "results", "firstp.json")))
    models = R["models"]
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for ax, m in zip(axs.ravel(), models + [None]):
        if m is None:
            ax.axis("off"); continue
        for v in ("raw", "glob", "pdens"):
            r = R[m][f"reliab_{v}"]
            n = np.array(r["n"]); ok = n > 100
            ax.plot(np.array(r["p_mean"])[ok], np.array(r["freq"])[ok], marker="o", ms=3.5, color=COL[v],
                    label=f"{v} (ECE {R[m][f'ece_{v}_own'][0]:.3f})")
        ax.plot([0, 1], [0, 1], color="#52514e", lw=0.8)
        ax.set_title(f"{m}: presence reliability (deployment cells)"); ax.set_xlabel("predicted presence"); ax.set_ylabel("frequency of charged (own label)")
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(HERE, "figs", "firstp_reliability.png"), dpi=120); plt.close(fig)
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    x = np.arange(len(QL))
    for ax, m in zip(axs.ravel(), models + [None]):
        if m is None:
            ax.axis("off"); continue
        for est in ("raw", "rawJ", "rawP", "rawPJ", "gated", "hurdle", "hurdleJ"):
            ks = [f"bypred_cells_ct_{est}_{b}" for b in QL]
            if not any(k in R[m] for k in ks):
                continue
            y = [R[m][k][0] if k in R[m] else np.nan for k in ks]
            ax.plot(x, y, marker="o", ms=3.5, color=ECOL[est], ls="--" if est.endswith("J") else "-", label=est)
        ax.axhline(1, color="#52514e", lw=0.8); ax.set_yscale("log"); ax.set_ylim(0.1, 10)
        ax.set_xticks(x); ax.set_xticklabels(QL, rotation=40)
        ax.set_title(f"{m}: truth / prediction in prediction bins (complete truth)")
        ax.set_xlabel("predicted cell charge (e-)"); ax.legend(fontsize=7)
    fig.savefig(os.path.join(HERE, "figs", "firstp_linear.png"), dpi=120); plt.close(fig)
    print("figs done")


if __name__ == "__main__":
    main()
