"""The figure for docs/PERFORMANCE.md.

Curves are the SIZE SWEEP: one corpus event tiled to a target token count, which
is how sizes the corpus does not contain get measured. The corpus's own spread is
shown as a band on the cost panels and as a distribution in the last panel.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

P = "/sdf/data/neutrino/omara/exp/helix/profiling/out"
d = json.load(open(f"{P}/p23_before_after.json"))
D = json.load(open(f"{P}/p25_distribution.json"))
sweep = sorted([r for r in d["synth"] if "speedup" in r], key=lambda r: r["n_cells"])
oom = min([r["n_cells"] for r in d["synth"] if r.get("before", {}).get("oom")],
          default=None)

CUR, OPT, ACC = "#eb6834", "#2a78d6", "#1baf7a"
INK, INK2, INK3, BAND = "#0b0b0b", "#52514e", "#8a8984", "#dcdbd5"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "axes.edgecolor": INK3, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "grid.color": "#ececE6", "grid.linewidth": 0.7,
})
K = lambda v, _: f"{v:,.0f}"
sw = np.array([r["n_cells"] for r in sweep]) / 1e3
c5, c95 = D["cells"]["p5"] / 1e3, D["cells"]["p95"] / 1e3
XL = (3.4, 150)


def style(a, xlab=None, ylab=None, title=None):
    a.grid(True, alpha=1.0, zorder=0); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    if xlab: a.set_xlabel(xlab)
    if ylab: a.set_ylabel(ylab)
    if title: a.set_title(title, color=INK, loc="left", pad=9)


def band(a):
    a.axvspan(c5, c95, color=BAND, alpha=0.55, zorder=0, lw=0)


fig, ax = plt.subplots(1, 4, figsize=(17.4, 4.3))

# ---- A: time ----------------------------------------------------------------
a = ax[0]; band(a)
a.plot(sw, [r["before"]["ms"] for r in sweep], "-", color=CUR, lw=2.2, zorder=3)
a.plot(sw, [r["after"]["ms"] for r in sweep], "-", color=OPT, lw=2.2, zorder=3)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(*XL)
a.set_xticks([4, 8, 16, 32, 64, 128]); a.set_yticks([40, 60, 100, 200, 320])
a.xaxis.set_major_formatter(FuncFormatter(K)); a.yaxis.set_major_formatter(FuncFormatter(K))
a.xaxis.set_minor_formatter(lambda *_: ""); a.yaxis.set_minor_formatter(lambda *_: "")
style(a, "cells (tokens) per event, thousands", "step time (ms)", "Step time")
a.text(4.1, 96, "current", color=CUR, fontsize=10, fontweight="semibold")
a.text(4.1, 50, "optimised", color=OPT, fontsize=10, fontweight="semibold")
a.text(np.sqrt(c5 * c95), 340, "corpus\np5–p95", color=INK2, fontsize=8,
       ha="center", va="bottom", linespacing=1.4)

# ---- B: memory --------------------------------------------------------------
a = ax[1]; band(a)
if oom:
    a.axvspan(oom / 1e3, XL[1], color=CUR, alpha=0.09, zorder=0, lw=0)
a.plot(sw, [r["before"]["peak_MiB"] for r in sweep], "-", color=CUR, lw=2.2, zorder=3)
a.plot(sw, [r["after"]["peak_MiB"] for r in sweep], "-", color=OPT, lw=2.2, zorder=3)
a.axhline(40960, color=INK3, ls=(0, (5, 3)), lw=1.1, zorder=2)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(*XL); a.set_ylim(2300, 62000)
a.set_xticks([4, 8, 16, 32, 64, 128]); a.set_yticks([3000, 6000, 12000, 25000, 50000])
a.xaxis.set_major_formatter(FuncFormatter(K)); a.yaxis.set_major_formatter(FuncFormatter(K))
a.xaxis.set_minor_formatter(lambda *_: ""); a.yaxis.set_minor_formatter(lambda *_: "")
style(a, "cells (tokens) per event, thousands", "peak allocated (MiB)", "Peak memory")
a.text(3.7, 36000, "A100 40 GB", color=INK2, fontsize=8, va="top")
a.text(oom / 1e3 * 0.93, 3050, "current\nOOMs", color=CUR, fontsize=8.5,
       va="bottom", ha="right", linespacing=1.4)
a.text(4.1, 5300, "current", color=CUR, fontsize=10, fontweight="semibold")
a.text(4.1, 2480, "optimised", color=OPT, fontsize=10, fontweight="semibold")

# ---- C: ratio ---------------------------------------------------------------
a = ax[2]; band(a)
a.plot(sw, [r["speedup"] for r in sweep], "-", color=ACC, lw=2.2, zorder=3)
a.plot(sw, [1 / r["mem_ratio"] for r in sweep], "--", color=ACC, lw=2.2, zorder=3)
a.axhline(1, color=INK3, lw=0.9, zorder=2)
a.set_xscale("log"); a.set_xlim(*XL); a.set_ylim(0.95, 2.45)
a.set_xticks([4, 8, 16, 32, 64, 128])
a.xaxis.set_major_formatter(FuncFormatter(K)); a.xaxis.set_minor_formatter(lambda *_: "")
style(a, "cells (tokens) per event, thousands", "current / optimised",
      "Speed-up and memory saved")
a.text(4.1, 1.90, "speed-up", color=ACC, fontsize=10, fontweight="semibold")
a.text(4.1, 1.05, "memory saved", color=ACC, fontsize=10, fontweight="semibold")
a.text(0.955, 0.05, "at the corpus median\n1.84x faster,  1.81x less memory",
       transform=a.transAxes, color=INK2, fontsize=8.2, ha="right", va="bottom",
       linespacing=1.6)

# ---- D: corpus distribution -------------------------------------------------
a = ax[3]
kk = np.array(D["coeff_list"]) / 1e3
a.hist(kk, bins=34, color=INK3, alpha=0.55, edgecolor="white", linewidth=0.6, zorder=2)
top = a.get_ylim()[1]
a.set_ylim(0, top * 1.42)
for q, lab, ha_ in ((D["coeff"]["p5"]/1e3, "p5", "right"),
                    (D["coeff"]["p50"]/1e3, "median", "center"),
                    (D["coeff"]["p95"]/1e3, "p95", "left")):
    a.axvline(q, color=INK2, lw=1.3 if lab == "median" else 1.0,
              ls="-" if lab == "median" else (0, (3, 3)), zorder=3,
              ymax=1 / 1.42 * 1.02)
    a.text(q, top * 1.05, f"{lab}\n{q:,.0f}k", color=INK2, fontsize=8,
           va="bottom", ha=ha_, linespacing=1.4)
style(a, "coefficients per event, thousands", "events",
      "Corpus spread, 1,200 events")

fig.suptitle("helix coefficient-FM training step: current vs optimised, one A100-40GB",
             x=0.005, ha="left", fontsize=14.5, color=INK, y=0.985, va="top")
fig.text(0.005, 0.905,
         "curves are a size sweep (one corpus event tiled to size).  "
         "optimised = sparse-active head, permuted residual stream, fused RoPE, compiled blocks;  "
         "loss unchanged to 1.3e-5 relative.",
         fontsize=9, color=INK2, ha="left", va="top")
fig.text(0.005, 0.028,
         "Across 1,200 corpus events the coefficient count spans 14.4x (52k - 755k) while the "
         "token count spans only 1.68x (26.8k - 45.1k): the tokenizer absorbs almost all of the "
         "variation, so the cost sits in a narrow band.",
         fontsize=8.6, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.055, 1, 0.865))
out = f"{P}/fig_step_cost.png"
fig.savefig(out, dpi=160)
print("wrote", out)
