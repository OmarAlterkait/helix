#!/usr/bin/env python3
"""figs/aspects_*.png from v2/results/{benchmark,aspects}.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(os.path.dirname(HERE), "figs")
C = {"cls_default": "#8a8a85", "cls_tuned_q": "#eb6834", "cls_tuned_any": "#eda100", "cls_noroi": "#1baf7a",
     "cls_oracle_roi": "#c3c2b7", "M4n_q": "#2a78d6", "M4n_p": "#2a78d6", "N64_q": "#e87ba4", "N64_p": "#e87ba4"}
LAB = {"cls_default": "classical, MicroBooNE default", "cls_tuned_q": "classical, tuned ROI charge",
       "cls_tuned_any": "classical, tuned best statistic", "cls_noroi": "classical, no ROI (Gaus_wide)",
       "cls_oracle_roi": "classical, oracle ROI (reference)", "M4n_q": "model M4n, charge", "M4n_p": "model M4n, presence",
       "N64_q": "model 64-label, charge", "N64_p": "model 64-label, presence"}
BMAP = {"cls_default": "default", "cls_tuned_q": "tuned_q", "cls_tuned_any": "tuned_any", "M4n_p": "M4n_p", "N64_p": "N64_p",
        "M4n_q": "M4n_q"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def main():
    A = json.load(open(os.path.join(RES, "aspects.json")))
    B = json.load(open(os.path.join(RES, "benchmark.json")))
    os.makedirs(FIG, exist_ok=True)
    det = ("cls_default", "cls_tuned_q", "cls_tuned_any", "M4n_p", "N64_p")
    # ROC
    for side in ("far", "near"):
        fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for p, ax in enumerate(axs):
            for n in det:
                r = A["roc"].get(f"{n}|{'UVY'[p]}|0.1-0.2")
                if r:
                    x, y = np.array(r[side]).T
                    ax.plot(x, y, color=C[n], lw=2, ls="--" if n.startswith("N64") else "-", label=LAB[n])
            ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.axvline(0.01, color="#888", lw=0.8, ls=":")
            ax.set_title(f"{'UVY'[p]} plane, 0.1-0.2 MeV deposits"); ax.set_xlabel(f"false-positive rate ({side}-activity noise windows)")
            style(ax)
        axs[0].set_ylabel("efficiency"); axs[0].legend(fontsize=8, frameon=False, loc="upper left")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"aspects_roc_{side}.png"), dpi=130); plt.close(fig)
    # efficiency vs SNR (per plane, near threshold) from the benchmark bootstrap
    snr = ("nosignal", "0-3", "3-5", "5-7", "7-10", "10-15", "15-inf")
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharey=True, sharex=True)
    for i, side in enumerate(("far", "near")):
        for p in range(3):
            ax = axs[i, p]
            for n in det:
                bb = B["test"][BMAP[n]]["boot"]
                ys = [bb.get(f"{side}_snr_{'UVY'[p]}_{s}", [np.nan] * 3) for s in snr]
                y = np.array([v[0] for v in ys]); lo = np.array([v[1] for v in ys]); hi = np.array([v[2] for v in ys])
                xs = np.arange(len(snr))
                ax.errorbar(xs, y, yerr=[y - lo, hi - y], color=C[n], lw=2, marker="o", ms=4, capsize=2,
                            ls="--" if n.startswith("N64") else "-", label=LAB[n])
            ax.set_xticks(range(len(snr))); ax.set_xticklabels(snr, fontsize=8)
            ax.set_title(f"{'UVY'[p]}, threshold from {side}-activity noise"); style(ax)
            if i == 1:
                ax.set_xlabel("deposit optimal SNR (clean, isolated)")
        axs[i, 0].set_ylabel("efficiency at 1% FPR")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "aspects_snr.png"), dpi=130); plt.close(fig)
    # noise vs distance
    chg = ("cls_default", "cls_tuned_q", "cls_noroi", "M4n_q", "N64_q")
    dl = ("1", "2", "3-4", "5-8", "9-16", ">16")
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for i, key in enumerate(("frac_gt_025Q0", "frac_gt_1Q0")):
        for p in range(3):
            ax = axs[i, p]
            for n in chg:
                y = [A["noise"].get(f"{n}|{'UVY'[p]}|{d}", {}).get(key, np.nan) for d in dl]
                ax.plot(range(len(dl)), np.maximum(y, 1e-6), color=C[n], lw=2, marker="o", ms=4,
                        ls="--" if n.startswith("N64") else "-", label=LAB[n])
            ax.set_yscale("log"); ax.set_xticks(range(len(dl))); ax.set_xticklabels(dl)
            ax.set_title(f"{'UVY'[p]}: truth-empty cells above {'0.25' if i == 0 else '1'} Q0"); style(ax)
            if i == 1:
                ax.set_xlabel("distance to nearest charge (cells of 2 wires x 16 ticks)")
        axs[i, 0].set_ylabel("fraction of empty cells")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "aspects_noise.png"), dpi=130); plt.close(fig)
    # fidelity
    chg2 = ("cls_default", "cls_tuned_q", "cls_oracle_roi", "M4n_q", "N64_q")
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for p in range(3):
        rows = {n: A["fidelity"].get(f"{n}|{'UVY'[p]}", {}) for n in chg2}
        bins = list(rows["M4n_q"].keys())
        for n in chg2:
            m = [rows[n].get(b, {}).get("median_ratio", np.nan) for b in bins]
            h = [rows[n].get(b, {}).get("hw68", np.nan) for b in bins]
            axs[0, p].plot(range(len(bins)), m, color=C[n], lw=2, marker="o", ms=4, ls="--" if n.startswith("N64") else "-", label=LAB[n])
            axs[1, p].plot(range(len(bins)), h, color=C[n], lw=2, marker="o", ms=4, ls="--" if n.startswith("N64") else "-")
        for ax in axs[:, p]:
            ax.set_xticks(range(len(bins))); ax.set_xticklabels(bins, rotation=45, fontsize=7); style(ax)
        axs[0, p].axhline(1, color="#888", lw=0.8); axs[0, p].set_title(f"{'UVY'[p]}: per-cell charge, truth > 0 cells")
        axs[1, p].set_xlabel("true cell charge (e-)")
    axs[0, 0].set_ylabel("median predicted / true"); axs[1, 0].set_ylabel("68% half-width of predicted / true")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "aspects_fidelity.png"), dpi=130); plt.close(fig)
    # near-activity robustness + dips
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    dd = ("1-2", "3-4", "5-8", "9-16")
    for n in det:
        y = [np.mean([A["neardist"].get(f"{n}|{pl}|{d}", {}).get("fpr_at_far1pct", np.nan) for pl in "UVY"]) for d in dd]
        axs[0].plot(range(len(dd)), y, color=C[n], lw=2, marker="o", ms=4, ls="--" if n.startswith("N64") else "-", label=LAB[n])
    axs[0].set_xticks(range(len(dd))); axs[0].set_xticklabels(dd); axs[0].axhline(0.01, color="#888", lw=0.8, ls=":")
    axs[0].set_xlabel("distance of a charge-free window to the nearest charge (cells)")
    axs[0].set_ylabel("fraction above the far 1% threshold (mean of planes)"); axs[0].set_yscale("log"); style(axs[0])
    axs[0].legend(fontsize=8, frameon=False)
    sb = ("w<=4", "w<=8", "w<=16", "w<=24", "t<=32", "t<=64", "t<=96")
    PR = json.load(open(os.path.join(RES, "pairs.json")))
    pm = {"cls_tuned_q": "cls_tuned_q", "cls_tuned_any": "cls_tuned_any", "M4n_q": "M4n_q", "M4n_p": "M4n_p", "N64_q": "N64_q"}
    for n, k in pm.items():
        y = [PR.get(f"{k}|all|{s_}", {}).get("resolved_if_both", np.nan) for s_ in sb]
        yb = [PR.get(f"{k}|all|{s_}", {}).get("both", np.nan) for s_ in sb]
        axs[1].plot(range(len(sb)), y, color=C[n], lw=2, marker="o", ms=4, ls="--" if n.endswith("_p") else "-", label=LAB[n])
        axs[1].plot(range(len(sb)), yb, color=C[n], lw=1, marker="x", ms=4, ls=":")
    yt = [PR.get(f"truth|all|{s_}", {}).get("resolved_if_both", np.nan) for s_ in sb]
    axs[1].plot(range(len(sb)), yt, color="#444", lw=1, ls="-", label="truth itself")
    axs[1].set_xticks(range(len(sb))); axs[1].set_xticklabels(sb)
    axs[1].set_ylabel("solid: resolved (valley < 0.5 x weaker peak) | both found\ndotted: both deposits found (far 1% thr)")
    axs[1].set_xlabel("pair separation bin (wires / ticks)"); style(axs[1]); axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "aspects_near_pairs.png"), dpi=130); plt.close(fig)
    # validity summary: 0.1-0.2 MeV efficiency per plane, near and far, default vs tuned vs model
    fams = [("default", "classical, MicroBooNE default", "#8a8a85"), ("tuned_any", "classical, tuned (tuning split)", "#eda100"),
            ("N64_p", "model, 64 labelled events", "#e87ba4"), ("M4n_p", "model M4n, all labels", "#2a78d6")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, side in zip(axs, ("near", "far")):
        groups = ["U", "V", "Y", "pooled (per-plane thr)"]
        for j, (k, lab, col) in enumerate(fams):
            bb = B["test"].get(k, {}).get("boot", {})
            keys = [f"{side}_{p}_0.1-0.2" for p in "UVY"] + [f"{side}_pp_0.1-0.2"]
            y = np.array([bb.get(x, [np.nan] * 3) for x in keys])
            xs = np.arange(4) + (j - 1.5) * 0.2
            ax.bar(xs, y[:, 0], width=0.18, color=col, label=lab, edgecolor="white", linewidth=1)
            ax.errorbar(xs, y[:, 0], yerr=[y[:, 0] - y[:, 1], y[:, 2] - y[:, 0]], fmt="none", ecolor="#333", lw=1, capsize=2)
        ax.set_xticks(range(4)); ax.set_xticklabels(groups)
        ax.set_title(f"0.1-0.2 MeV deposits, 1% FPR from {side}-activity noise windows"); style(ax)
    axs[0].set_ylabel("efficiency (128 test events, 16-84% bootstrap)"); axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "validity_summary.png"), dpi=130); plt.close(fig)
    print("figs written")


if __name__ == "__main__":
    main()
