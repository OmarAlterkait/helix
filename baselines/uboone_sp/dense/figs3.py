#!/usr/bin/env python3
"""Figures for dense3.md (v3 models: M6d raw / gated, M6dc gated; M4n reference).

    scripts/helix_run.sh python baselines/uboone_sp/dense/figs3.py --tag v3 \
        --cls /global/cfs/.../dense2/cls_fix,/pscratch/.../dense/cls_fit --model /global/cfs/.../dense3/model
Reuses figs.py's plotting helpers with the v2 method set.
"""
import argparse
import json
import os

import numpy as np

import figs as F
from figs import DL, QL, RQ, heat, plt, LogNorm, SymLogNorm

HERE = F.HERE
F.COL.update({"M4n_g": "#2a78d6", "M6d_q": "#1baf7a", "M6d_g": "#1baf7a", "M6d_p": "#1baf7a",
              "M6dc_g": "#4a3aa7", "M6dc_p": "#4a3aa7"})
F.LAB.update({"M4n_q": "M4n raw", "M4n_g": "M4n gated", "M6d_q": "M6d raw", "M6d_g": "M6d gated", "M6dc_g": "M6dc gated",
              "M6d_p": "M6d presence", "M6dc_p": "M6dc presence"})
MS = ["cls_default", "cls_tuned_q", "cls_oracle", "M4n_q", "M4n_g", "M6d_q", "M6d_g", "M6dc_g"]
DASH = {"M4n_q": ":", "M6d_q": ":", "cls_oracle": "--"}


def lines(ax, x, ys, m, lw=None):
    ax.plot(x, ys, marker="o", ms=3.5, color=F.COL[m], ls=DASH.get(m, "-"),
            lw=lw or (2 if m in ("M6d_g", "cls_tuned_q") else 1.3), label=F.LAB[m])


def fig_margins(R, out, model="M6d_g", pres="M6d_p"):
    Fi, P = R["fidelity"], R["presence"]
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for k, pl in enumerate(["U", "V", "Y", "all"]):
        M = np.full((len(DL), len(QL)), np.nan); A = [["" for _ in QL] for _ in DL]
        for i, d in enumerate(DL):
            for j, q in enumerate(QL):
                a, b = Fi["cls_tuned_q"].get(f"{pl}|{d}|{q}"), Fi[model].get(f"{pl}|{d}|{q}")
                if a and b:
                    ec, em = a["median_rel_abs_err"][0], b["median_rel_abs_err"][0]
                    M[i, j] = ec - em; A[i][j] = f"{ec:.2f}\n{em:.2f}"
        im = heat(axs[0, k], M, DL, QL, f"{pl}: median |pred-true|/true\nclassical tuned (top) / {F.LAB[model]} (bottom)", 1.0, ann=A)
        M = np.full((len(DL), len(QL)), np.nan); A = [["" for _ in QL] for _ in DL]
        for i, d in enumerate(DL):
            for j, q in enumerate(QL):
                a, b = P.get(f"cls_tuned_any|{pl}|{d}|{q}"), P.get(f"{pres}|{pl}|{d}|{q}")
                if a and b:
                    M[i, j] = b["eff"][0] - a["eff"][0]; A[i][j] = f"{a['eff'][0]:.2f}\n{b['eff'][0]:.2f}"
        im2 = heat(axs[1, k], M, DL, QL, f"{pl}: eff at 1% FPR (same-density empties)\nclassical best stat (top) / {F.LAB[pres]} (bottom)", 0.6, ann=A)
        if k == 0:
            axs[0, k].set_ylabel("true charge within +-8 cells (e-)"); axs[1, k].set_ylabel("true charge within +-8 cells (e-)")
        axs[1, k].set_xlabel("cell true charge (e-)")
    fig.colorbar(im, ax=axs[0, :], shrink=0.8, label="classical error - model error (blue: model better)")
    fig.colorbar(im2, ax=axs[1, :], shrink=0.8, label="model eff - classical eff (blue: model better)")
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_gating(R, out):
    """Where gating zeroes true charge: share of true charge in zeroed cells, density x cell charge."""
    Z = R["zeroed"]
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
    for ax, m in zip(axs, ("cls_tuned_q", "M4n_g", "M6d_g", "M6dc_g")):
        M = np.full((len(DL), len(QL)), np.nan)
        for i, d in enumerate(DL):
            for j, q in enumerate(QL):
                e = Z.get(f"{m}|all|{d}|{q}")
                if e:
                    M[i, j] = e["charge_frac"]
        im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#0b0b0b" if M[i, j] < 0.55 else "#ffffff")
        ax.set_xticks(range(len(QL))); ax.set_xticklabels(QL, rotation=45, ha="right")
        ax.set_yticks(range(len(DL))); ax.set_yticklabels(DL); ax.grid(False)
        tot = Z[f"{m}|all|total_charge_zeroed"] / Z[f"{m}|all|total_charge"]
        ax.set_title(f"{F.LAB[m]}: share of true charge set to 0\n(all planes; {100 * tot:.2f}% of all charge)")
        ax.set_xlabel("cell true charge (e-)")
    axs[0].set_ylabel("true charge within +-8 cells (e-)")
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_false(R, out):
    fc = R["false_charge"]
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.5), constrained_layout=True, sharex=True)
    x = np.arange(len(DL))
    for r, (key, lab) in enumerate((("gt005Q0", "> 0.05 Q0 (100 e-)"), ("gt1Q0", "> 1 Q0 (2,043 e-)"))):
        for k, pl in enumerate(["U", "V", "Y", "all"]):
            ax = axs[r, k]
            for m in MS:
                y = np.array([fc[f"{m}|{pl}|{d}"][key][0] if f"{m}|{pl}|{d}" in fc else np.nan for d in DL])
                lines(ax, x, np.maximum(y, 1e-7), m)
            ax.set_yscale("log"); ax.set_title(f"{pl}: empty cells {lab}")
            ax.set_xticks(x); ax.set_xticklabels(DL, rotation=30)
            if r == 1:
                ax.set_xlabel("true charge within +-8 cells (e-)")
            if k == 0:
                ax.set_ylabel("fraction of empty cells (1e-7 = none)")
    axs[0, 0].legend(fontsize=7, loc="lower right")
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_regions(R, out):
    fig, axs = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True, sharex=True)
    x = np.arange(len(RQ))
    for r, key in enumerate(("regions16x8", "regions4x4")):
        G = R[key]
        for k, pl in enumerate(["U", "V", "Y", "all"]):
            ax = axs[r, k]
            for m in MS:
                e = [G[m].get(f"{pl}|q|{q}") for q in RQ]
                lines(ax, x, np.array([v["sum_ratio"][0] if v else np.nan for v in e]), m)
            ax.axhline(1, color="#52514e", lw=0.8)
            ax.set_yscale("log"); ax.set_ylim(0.2, 12); ax.set_xticks(x); ax.set_xticklabels(RQ, rotation=35)
            ax.set_title(f"{pl}: {'16 x 8' if r == 0 else '4 x 4'} cell regions" + (" (32 w x 128 t)" if r == 0 else " (8 w x 64 t)"))
            if k == 0:
                ax.set_ylabel("sum predicted / sum true (log)")
            if r == 1:
                ax.set_xlabel("true region charge (e-)")
    axs[0, 0].legend(fontsize=7, loc="upper right")
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_particles_sep(R, out):
    P, S = R["particles"], R["separation"]
    EBL = ["0.1-0.3", "0.3-1", "1-3", "3-10", "10-30", "30-100", ">100"]
    OBL = ["<0.1", "0.1-0.3", "0.3-0.6", ">0.6"]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    for ax, (nm, labs, xl) in zip(axs[:2], (("E", EBL, "particle deposited energy (MeV)"), ("overlap", OBL, "overlap fraction"))):
        x = np.arange(len(labs))
        for i, m in enumerate(MS):
            e = [P[m].get(f"all|{nm}|{l}") for l in labs]
            y = np.array([v["median_ratio"][0] if v else np.nan for v in e]); hw = np.array([v["hw68"][0] if v else np.nan for v in e])
            ax.errorbar(x + 0.04 * i - 0.15, y, hw, fmt="o", ms=3.5, color=F.COL[m], label=F.LAB[m], capsize=2,
                        mfc="white" if m in ("M4n_q", "M6d_q") else F.COL[m])
        ax.axhline(1, color="#52514e", lw=0.8); ax.set_ylim(0, 1.8)
        ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlabel(xl)
        ax.set_ylabel("pred / true on majority cells\nmedian +- 68% half-width")
    axs[0].legend(fontsize=6.5)
    for ax, axn in zip(axs[2:], ("wire", "time")):
        x = np.arange(1, len(DL))
        for m in MS:
            lines(ax, x, [S[m].get(f"all|{axn}|sall|{d}", {}).get("resolved", [np.nan])[0] for d in DL[1:]], m)
        ax.set_xticks(x); ax.set_xticklabels(DL[1:]); ax.set_ylim(0, 1.02)
        ax.set_xlabel("true charge within +-8 cells of the valley (e-)")
        ax.set_title(f"truth double peaks 2-4 cells apart along {axn}")
        ax.set_ylabel("resolved fraction")
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_displays(cls_dirs, model_dir, picks, out):
    n = len(picks)
    fig, axs = plt.subplots(n, 6, figsize=(23, 3.9 * n), constrained_layout=True)
    for r, p in enumerate(picks):
        ev, g, i, j = p["ev"], p["g"], p["cell_w"], p["cell_t"]
        Fz = np.load(next(os.path.join(d, f"ev{ev:03d}_g{g}.npz") for d in cls_dirs if os.path.exists(os.path.join(d, f"ev{ev:03d}_g{g}.npz"))))
        Mz = np.load(os.path.join(model_dir, f"ev{ev:03d}.npz"))                                  # M6d / M6dc
        M4 = np.load(os.path.join("/global/cfs/cdirs/m5238/users/oalter/helix_runs/dense2/model", f"ev{ev:03d}.npz"))
        T = Fz["truth"]; to = int(Mz["toff"])
        crop = lambda X: X[:T.shape[0], to:to + T.shape[1]]
        g5 = np.where(crop(Mz[f"p_M6d_g{g}"]) > 0.5, crop(Mz[f"q_M6d_g{g}"]), 0)
        m4 = np.where(crop(Mz[f"p_M6dc_g{g}"]) > 0.5, crop(Mz[f"q_M6dc_g{g}"]), 0)          # M6dc gated
        sl = (slice(max(i - 20, 0), i + 20), slice(max(j - 20, 0), j + 20))
        t, c, a4, a5 = T[sl], Fz["tuned_q"][sl], m4[sl], g5[sl]
        vmax = max(t.max(), c.max(), a5.max())
        norm = LogNorm(vmin=50, vmax=vmax); rn = SymLogNorm(linthresh=200, vmin=-vmax / 4, vmax=vmax / 4)
        ext = (sl[1].start * 16, (sl[1].start + t.shape[1]) * 16, (sl[0].start + t.shape[0]) * 2, sl[0].start * 2)
        for k, (X, ttl) in enumerate(((t, "truth"), (c, "classical tuned"), (a4, "M6dc gated"), (a5, "M6d gated"))):
            im = axs[r, k].imshow(np.where(X > 0, X, np.nan), norm=norm, cmap="viridis", aspect="auto", extent=ext, interpolation="nearest")
            axs[r, k].set_facecolor("#f3f2ee")
            axs[r, k].set_title(f"ev{ev} {'UVY'[g % 3]}: {ttl}\nsum {X.sum() / 1e6:.2f}M e-")
        fig.colorbar(im, ax=axs[r, 3], label="e- per cell")
        for k, (X, ttl) in enumerate(((a4 - t, "M6dc gated - truth"), (a5 - t, "M6d gated - truth"))):
            im2 = axs[r, 4 + k].imshow(X, norm=rn, cmap="RdBu_r", aspect="auto", extent=ext, interpolation="nearest")
            axs[r, 4 + k].set_title(f"{ttl}\nmedian |res| on charged cells {np.median(np.abs(X[t > 0])):.0f} e-")
        fig.colorbar(im2, ax=axs[r, 5], label="residual e- (symlog)")
        for a_ in axs[r]:
            a_.grid(False); a_.set_xlabel("tick"); a_.set_ylabel("wire")
    fig.savefig(out, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--cls", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    R = json.load(open(os.path.join(HERE, "results", f"dense_{a.tag}.json")))
    fd = os.path.join(HERE, "figs")
    fig_margins(R, os.path.join(fd, f"margin_{a.tag}.png"))
    fig_margins(R, os.path.join(fd, f"margin_M6dcgated_{a.tag}.png"), model="M6dc_g", pres="M6dc_p")
    fig_gating(R, os.path.join(fd, f"gating_zeroed_{a.tag}.png"))
    fig_false(R, os.path.join(fd, f"false_charge_{a.tag}.png"))
    fig_regions(R, os.path.join(fd, f"regions_{a.tag}.png"))
    fig_particles_sep(R, os.path.join(fd, f"particles_separation_{a.tag}.png"))
    picks = json.load(open(os.path.join(HERE, "results", "displays_fit.json")))     # same dense windows as dense.md
    fig_displays(a.cls.split(","), a.model, picks, os.path.join(fd, f"displays_{a.tag}.png"))
    print("figs done")


if __name__ == "__main__":
    main()
