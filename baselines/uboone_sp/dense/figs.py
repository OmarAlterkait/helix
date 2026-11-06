#!/usr/bin/env python3
"""Figures for dense.md from results/dense_<tag>.json (+ event displays from the maps).

    scripts/helix_run.sh python baselines/uboone_sp/dense/figs.py --tag fit \
        --cls /pscratch/sd/o/oalter/helix_work/uboone_sp/dense/cls_fit
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
Q0 = 2043.258
DL = ["iso", "<=10k", "10k-100k", "100k-1M", "1M-4M", ">4M"]
QL = ["<500", "0.5-1k", "1-2k", "2-4k", "4-8k", "8-16k", "16-64k", ">64k"]
RQ = ["<1k", "1-4k", "4-16k", "16-64k", "64-256k", "256k-1M", "1-4M", ">4M"]
COL = {"M4n_q": "#2a78d6", "M4n_p": "#2a78d6", "cls_tuned_q": "#eb6834", "cls_tuned_any": "#eb6834",
       "N64_q": "#1baf7a", "N64_p": "#1baf7a", "M2n_q": "#4a3aa7", "M2n_p": "#4a3aa7",
       "cls_default": "#eda100", "cls_oracle": "#8a8984"}
LAB = {"M4n_q": "model M4n (charge)", "M4n_p": "model M4n (presence)", "cls_tuned_q": "classical tuned (ROI charge)",
       "cls_tuned_any": "classical tuned (best statistic)", "N64_q": "model 64-label", "N64_p": "model 64-label (presence)",
       "M2n_q": "model M2n scratch", "M2n_p": "model M2n (presence)", "cls_default": "classical MicroBooNE default",
       "cls_oracle": "classical oracle ROI (reference)"}
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f3f2ee", "#2a78d6"])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def heat(ax, M, rows, cols, title, vlim, fmt="{:+.2f}", ann=None):
    im = ax.imshow(M, cmap=DIV, vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.grid(False); ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]) if ann is None else ann[i][j], ha="center", va="center", fontsize=6.5,
                        color="#0b0b0b")
    return im


def fig_margins(R, out):
    F, P = R["fidelity"], R["presence"]
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for k, pl in enumerate(["U", "V", "Y", "all"]):
        M = np.full((len(DL), len(QL)), np.nan); A = [["" for _ in QL] for _ in DL]
        for i, d in enumerate(DL):
            for j, q in enumerate(QL):
                a, b = F["cls_tuned_q"].get(f"{pl}|{d}|{q}"), F["M4n_q"].get(f"{pl}|{d}|{q}")
                if a and b:
                    ec, em = a["median_rel_abs_err"][0], b["median_rel_abs_err"][0]
                    M[i, j] = ec - em
                    A[i][j] = f"{ec:.2f}\n{em:.2f}"
        im = heat(axs[0, k], M, DL, QL, f"{pl}: median |pred-true|/true\nclassical tuned (top) / M4n (bottom)", 1.0, ann=A)
        M = np.full((len(DL), len(QL)), np.nan); A = [["" for _ in QL] for _ in DL]
        for i, d in enumerate(DL):
            for j, q in enumerate(QL):
                a, b = P.get(f"cls_tuned_any|{pl}|{d}|{q}"), P.get(f"M4n_p|{pl}|{d}|{q}")
                if a and b:
                    M[i, j] = b["eff"][0] - a["eff"][0]
                    A[i][j] = f"{a['eff'][0]:.2f}\n{b['eff'][0]:.2f}"
        im2 = heat(axs[1, k], M, DL, QL, f"{pl}: eff at 1% FPR (same-density empties)\nclassical best stat (top) / M4n presence (bottom)", 0.6, ann=A)
        axs[0, k].set_ylabel("true charge within +-8 cells (e-)") if k == 0 else None
        axs[1, k].set_ylabel("true charge within +-8 cells (e-)") if k == 0 else None
        axs[1, k].set_xlabel("cell true charge (e-)")
    fig.colorbar(im, ax=axs[0, :], shrink=0.8, label="classical error - model error (blue: model better)")
    fig.colorbar(im2, ax=axs[1, :], shrink=0.8, label="model eff - classical eff (blue: model better)")
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_false(R, out):
    fc = R["false_charge"]
    ms = ["cls_default", "cls_tuned_q", "cls_oracle", "M4n_q", "N64_q", "M2n_q"]
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.5), constrained_layout=True, sharex=True)
    x = np.arange(len(DL))
    for r, (key, lab) in enumerate((("gt005Q0", "> 0.05 Q0 (100 e-)"), ("gt1Q0", "> 1 Q0 (2,043 e-)"))):
        for k, pl in enumerate(["U", "V", "Y", "all"]):
            ax = axs[r, k]
            for m in ms:
                y = np.array([fc[f"{m}|{pl}|{d}"][key] if f"{m}|{pl}|{d}" in fc else [np.nan] * 3 for d in DL])
                ax.plot(x, np.maximum(y[:, 0], 1e-7), marker="o", ms=4, lw=2 if m in ("M4n_q", "cls_tuned_q") else 1.2,
                        ls="--" if m == "cls_oracle" else "-", color=COL[m], label=LAB[m])
                ax.fill_between(x, np.maximum(y[:, 1], 1e-7), np.maximum(y[:, 2], 1e-7), color=COL[m], alpha=0.15, lw=0)
            ax.set_yscale("log"); ax.set_title(f"{pl}: empty cells {lab}")
            ax.set_xticks(x); ax.set_xticklabels(DL, rotation=30)
            if r == 1:
                ax.set_xlabel("true charge within +-8 cells (e-)")
            if k == 0:
                ax.set_ylabel("fraction of empty cells")
    axs[0, 0].legend(fontsize=7, loc="upper left")
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_regions(R, out):
    ms = ["cls_default", "cls_tuned_q", "cls_oracle", "M4n_q", "N64_q", "M2n_q"]
    fig, axs = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True, sharex=True)
    x = np.arange(len(RQ))
    for r, key in enumerate(("regions16x8", "regions4x4")):
        G = R[key]
        for k, pl in enumerate(["U", "V", "Y", "all"]):
            ax = axs[r, k]
            for m in ms:
                e = [G[m].get(f"{pl}|q|{q}") for q in RQ]
                y = np.array([v["median_ratio"][0] if v else np.nan for v in e])
                hw = np.array([v["hw68"][0] if v else np.nan for v in e])
                s = np.array([v["sum_ratio"][0] if v else np.nan for v in e])
                ax.plot(x, y, marker="o", ms=4, color=COL[m], ls="--" if m == "cls_oracle" else "-",
                        lw=2 if m in ("M4n_q", "cls_tuned_q") else 1.2, label=LAB[m])
                ax.fill_between(x, y - hw, y + hw, color=COL[m], alpha=0.10, lw=0)
                ax.plot(x, s, ":", color=COL[m], lw=1)
            ax.axhline(1, color="#52514e", lw=0.8)
            ax.set_ylim(0, 2.5); ax.set_xticks(x); ax.set_xticklabels(RQ, rotation=35)
            ax.set_title(f"{pl}: {'16 x 8' if r == 0 else '4 x 4'} cell regions" + (" (32 wires x 128 ticks)" if r == 0 else " (8 wires x 64 ticks)"))
            if k == 0:
                ax.set_ylabel("predicted / true region charge\nmedian, 68% band; dotted = sum ratio")
            if r == 1:
                ax.set_xlabel("true region charge (e-)")
    axs[0, 0].legend(fontsize=7, loc="upper right")
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_particles_sep(R, out):
    P, S = R["particles"], R["separation"]
    ms = ["cls_default", "cls_tuned_q", "cls_oracle", "M4n_q", "N64_q", "M2n_q"]
    EBL = ["0.1-0.3", "0.3-1", "1-3", "3-10", "10-30", "30-100", ">100"]
    OBL = ["<0.1", "0.1-0.3", "0.3-0.6", ">0.6"]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    for ax, (nm, labs, xl) in zip(axs[:2], (("E", EBL, "particle deposited energy (MeV)"),
                                             ("overlap", OBL, "fraction of the particle's charge in cells shared >= 10% with others"))):
        x = np.arange(len(labs))
        for m in ms:
            e = [P[m].get(f"all|{nm}|{l}") for l in labs]
            y = np.array([v["median_ratio"][0] if v else np.nan for v in e]); hw = np.array([v["hw68"][0] if v else np.nan for v in e])
            ax.errorbar(x + 0.05 * ms.index(m) - 0.12, y, hw, fmt="o", ms=4, color=COL[m], label=LAB[m], capsize=2)
        ax.axhline(1, color="#52514e", lw=0.8); ax.set_ylim(0, 1.8)
        ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlabel(xl)
        ax.set_ylabel("pred / true on the particle's majority cells\nmedian +- 68% half-width")
    axs[0].legend(fontsize=7)
    for ax, axn in zip(axs[2:], ("wire", "time")):
        x = np.arange(1, len(DL))
        for m in ms:
            y = [S[m].get(f"all|{axn}|sall|{d}", {}).get("resolved", [np.nan])[0] for d in DL[1:]]
            yf = [S[m].get(f"all|{axn}|sall|{d}", {}).get("resolved_given_found", [np.nan])[0] for d in DL[1:]]
            ax.plot(x, y, marker="o", ms=4, color=COL[m], ls="--" if m == "cls_oracle" else "-", label=LAB[m])
            ax.plot(x, yf, ":", color=COL[m], lw=1)
        ax.set_xticks(x); ax.set_xticklabels(DL[1:]); ax.set_ylim(0, 1.02)
        ax.set_xlabel("true charge within +-8 cells of the valley (e-)")
        ax.set_title(f"truth double peaks 2-4 cells apart along {axn}")
        ax.set_ylabel("resolved (both > 0.25 Q0, valley < 0.5 x weaker)\ndotted: given both found")
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_displays(cls_dir, out, n=3):
    from scipy.ndimage import uniform_filter
    V2M = "/pscratch/sd/o/oalter/helix_work/uboone_sp/v2/model"
    cands = []
    for ev in range(260, 388, 3):
        for g in range(6):
            T = np.load(os.path.join(cls_dir, f"ev{ev:03d}_g{g}.npz"))["truth"]
            occ = uniform_filter((T > 0).astype(float), size=(40, 40), mode="constant")
            i, j = np.unravel_index(np.argmax(occ), occ.shape)
            cands.append((occ[i, j], ev, g, i, j))
    cands.sort(reverse=True)
    picked, used = [], set()
    for c in cands:
        if c[1] in used or any(c[2] % 3 == p[2] % 3 for p in picked):
            continue
        picked.append(c); used.add(c[1])
        if len(picked) == n:
            break
    fig, axs = plt.subplots(n, 5, figsize=(19, 3.9 * n), constrained_layout=True)
    for r, (_, ev, g, i, j) in enumerate(picked):
        F = np.load(os.path.join(cls_dir, f"ev{ev:03d}_g{g}.npz"))
        Mz = np.load(os.path.join(V2M, f"ev{ev:03d}.npz"))
        T = F["truth"]; to = int(Mz["toff"])
        M = Mz[f"q_M4n_g{g}"][:T.shape[0], to:to + T.shape[1]]
        sl = (slice(max(i - 20, 0), i + 20), slice(max(j - 20, 0), j + 20))
        t, c, m = T[sl], F["tuned_q"][sl], M[sl]
        vmax = max(t.max(), c.max(), m.max())
        norm = LogNorm(vmin=50, vmax=vmax)
        rn = SymLogNorm(linthresh=200, vmin=-vmax / 4, vmax=vmax / 4)
        ext = (sl[1].start * 16, (sl[1].start + t.shape[1]) * 16, (sl[0].start + t.shape[0]) * 2, sl[0].start * 2)
        for k, (X, ttl) in enumerate(((t, "truth"), (c, "classical tuned ROI charge"), (m, "model M4n charge"))):
            im = axs[r, k].imshow(np.where(X > 0, X, np.nan), norm=norm, cmap="viridis", aspect="auto", extent=ext,
                                  interpolation="nearest")
            axs[r, k].set_facecolor("#f3f2ee")
            axs[r, k].set_title(f"ev{ev} {'UVY'[g % 3]} (gid {g}): {ttl}\nsum {X.sum() / 1e6:.2f}M e-")
        fig.colorbar(im, ax=axs[r, 2], label="e- per cell (2 wires x 16 ticks)")
        for k, (X, ttl) in enumerate(((c - t, "classical - truth"), (m - t, "M4n - truth"))):
            im2 = axs[r, 3 + k].imshow(X, norm=rn, cmap="RdBu_r", aspect="auto", extent=ext, interpolation="nearest")
            axs[r, 3 + k].set_title(f"{ttl}\nmedian |res| on charged cells {np.median(np.abs(X[t > 0])):.0f} e-")
        fig.colorbar(im2, ax=axs[r, 4], label="residual e- (symlog)")
        for a in axs[r]:
            a.grid(False); a.set_xlabel("tick"); a.set_ylabel("wire")
    fig.savefig(out, dpi=120); plt.close(fig)
    return [dict(ev=int(p[1]), g=int(p[2]), cell_w=int(p[3]), cell_t=int(p[4]), occupancy=float(p[0])) for p in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fit")
    ap.add_argument("--cls", required=True)
    a = ap.parse_args()
    R = json.load(open(os.path.join(HERE, "results", f"dense_{a.tag}.json")))
    fd = os.path.join(HERE, "figs"); os.makedirs(fd, exist_ok=True)
    fig_margins(R, os.path.join(fd, f"margin_{a.tag}.png"))
    fig_false(R, os.path.join(fd, f"false_charge_{a.tag}.png"))
    fig_regions(R, os.path.join(fd, f"regions_{a.tag}.png"))
    fig_particles_sep(R, os.path.join(fd, f"particles_separation_{a.tag}.png"))
    d = fig_displays(a.cls, os.path.join(fd, f"displays_{a.tag}.png"))
    json.dump(d, open(os.path.join(HERE, "results", f"displays_{a.tag}.json"), "w"), indent=1)
    print("figs done", d)


if __name__ == "__main__":
    main()
