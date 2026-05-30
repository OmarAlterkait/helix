"""Professional RMS-vs-noise frontier. Linear RMS (ADC) y, log compression x,
noise multiples (1x/2x/3x) as reference lines. Reads big/abs_rms.json."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

OUT = "/sdf/group/neutrino/omara/helix/temp/figures"
D = json.load(open(f"{OUT}/big/abs_rms.json"))
rows, NOISE = D["rows"], D["noise"]


def frontier(pts):
    pts = sorted(pts, key=lambda p: -p["comp"]); best = 1e18; fr = []
    for r in pts:
        if r["rms"] < best:
            fr.append(r); best = r["rms"]
    return sorted(fr, key=lambda r: r["comp"])


fr = frontier(rows)
fc = np.array([r["comp"] for r in fr]); fr_rms = np.array([r["rms"] for r in fr])
logc = np.log10(fc)

plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9, 5.8))

mult = [(1, "#c0392b", "--"), (2, "#e67e22", "-."), (3, "#f1c40f", ":")]
xmin, xmax = fc.min() * 0.9, fc.max() * 1.35
crossings = {}
for m, col, ls in mult:
    y0 = m * NOISE
    ax.axhline(y0, color=col, ls=ls, lw=1.6, zorder=2)
    # label sitting just above its line, left side
    ax.text(xmin * 1.04, y0 + 0.10, f"{m}× noise = {y0:.1f} ADC", color=col, fontsize=10.5,
            va="bottom", ha="left", fontweight="medium")
    # true intersection of the frontier with this line (interpolate in log-compression)
    if fr_rms.min() <= y0 <= fr_rms.max():
        xc = 10 ** np.interp(y0, fr_rms, logc)
        crossings[m] = xc
        ax.plot([xc], [y0], "o", color=col, ms=10, mfc="white", mew=2.2, zorder=6)

# faint config cloud + frontier
ax.scatter([r["comp"] for r in rows], [r["rms"] for r in rows], s=9, c="0.85", zorder=1)
ax.plot(fc, fr_rms, "-o", color="#1f4e79", ms=5, lw=2.4, zorder=4,
        label="VisuShrink-hard (coif3/sym6) + quantization")

# crossing numbers in a bottom-right box
lines = [f"{m}× noise  →  {crossings[m]:.0f}×" for m, _, _ in mult if m in crossings]
ax.text(0.975, 0.04, "compression at noise multiples\n" + "\n".join(lines),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.6", lw=0.8))

ax.set_xscale("log")
ax.set_xlabel("compression ratio", fontsize=13)
ax.set_ylabel("reconstruction RMS over signal  [ADC]", fontsize=13)
ax.set_ylim(0, 3.4 * NOISE)
ax.set_xlim(xmin, xmax)
ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(ScalarFormatter())
ax.tick_params(which="both", direction="out")
ax.grid(axis="y", alpha=0.25)
ax.set_title("Optical waveform compression — distortion vs. noise floor", fontsize=14, pad=10)
ax.legend(loc="upper left", frameon=True, fontsize=11)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/pareto_rms_noise.png", dpi=150)
print("wrote pareto_rms_noise.png; crossings:", {m: round(v, 1) for m, v in crossings.items()})
