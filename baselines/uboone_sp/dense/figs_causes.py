#!/usr/bin/env python3
"""Figures for causes_ade.md from results/causes_ade.json.

    scripts/helix_run.sh python baselines/uboone_sp/dense/figs_causes.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"cls": "#eb6834", "M6d": "#2a78d6", "M6dcn": "#1baf7a", "KL_gate": "#8a8984", "KL_k1": "#4a3aa7", "KL_corp": "#e87ba4"}
LAB = {"cls": "classical tuned (its noise draw)", "M6d": "M6d presence (stored corpus)", "M6dcn": "M6d presence (classical's noise draw)",
       "KL_gate": "known-location MF, full gated waveform", "KL_k1": "known-location MF, corpus-like input (same draw)",
       "KL_corp": "known-location MF, stored corpus"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0"})


def main():
    R = json.load(open(os.path.join(HERE, "results", "causes_ade.json")))
    T, Z = R["eff_vs_snr"], R["eff_vs_evidence"]
    sb = ["0-3", "3-5", "5-7", "7-10", "10-15", ">15"]
    zb = ["<2", "2-3", "3-4", "4-5", "5-6", "6-8", "8-12", ">12"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for k, pl in enumerate("UVY"):
        ax = axs[0, k]
        for m in ("cls", "M6d", "M6dcn", "KL_gate", "KL_k1", "KL_corp"):
            y = [T.get(f"{m}|{pl}|clean|{b}", [np.nan])[0] for b in sb]
            ax.plot(range(len(sb)), y, marker="o", ms=4, color=COL[m], ls="--" if m.startswith("KL") else "-", label=LAB[m])
        ax.set_xticks(range(len(sb))); ax.set_xticklabels(sb); ax.set_ylim(0, 1.02)
        ax.set_title(f"{pl}: clean isolated deposits, 1% FPR near-activity threshold")
        ax.set_xlabel("deposit optimal SNR (full waveform, analytic)"); ax.set_ylabel("efficiency")
        ax = axs[1, k]
        for m, zn in (("cls", "z_k1"), ("M6dcn", "z_k1"), ("cls", "z_gate"), ("M6dcn", "z_gate")):
            y = [Z.get(f"{zn}|{m}|{pl}|clean|{b}", [np.nan])[0] for b in zb]
            ax.plot(range(len(zb)), y, marker="o" if zn == "z_k1" else "s", ms=4, color=COL[m], ls="-" if zn == "z_k1" else ":",
                    label=f"{LAB[m]} vs {'corpus-input' if zn == 'z_k1' else 'full-waveform'} evidence")
        ax.set_xticks(range(len(zb))); ax.set_xticklabels(zb); ax.set_ylim(0, 1.02)
        ax.set_title(f"{pl}: same noise draw for both methods")
        ax.set_xlabel("matched-filter evidence on the input (gate-noise sigma)"); ax.set_ylabel("efficiency")
    axs[0, 0].legend(fontsize=6.5); axs[1, 0].legend(fontsize=6.5)
    fig.savefig(os.path.join(HERE, "figs", "causes_A_D.png"), dpi=130); plt.close(fig)
    E = R["E"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for k, pl in enumerate("UVY"):
        ax = axs[k]
        for m, c in (("M4n", "#eda100"), ("M6d", "#2a78d6")):
            f = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
            ax.plot(f, [E[f"{m}|{pl}|cellfpr{x:g}"]["iso_window_eff"] for x in f], marker="o", color=c, label=f"{m}: iso-window efficiency")
            ax.plot(f, [E[f"{m}|{pl}|cellfpr{x:g}"]["window_fpr"] for x in f], marker="s", ls=":", color=c, label=f"{m}: noise-window FPR")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-3, 1.05)
        ax.set_xlabel("cell-level FPR (empty cells of near-activity noise windows)"); ax.set_title(f"{pl}: 0.1-0.2 MeV deposits")
    axs[0].legend(fontsize=7)
    fig.savefig(os.path.join(HERE, "figs", "causes_E.png"), dpi=130); plt.close(fig)
    print("figs done")


if __name__ == "__main__":
    main()
