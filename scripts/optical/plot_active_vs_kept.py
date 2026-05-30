"""Three distributions (one per operating point): per-chunk active signal pixels
vs kept coefficients, across all events. Reads big/active_vs_kept.json."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = "/sdf/group/neutrino/omara/helix/temp/figures"
D = json.load(open(f"{OUT}/big/active_vs_kept.json"))
OPS, data = D["ops"], D["data"]

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)
# global ranges for shared, comparable axes
allx = np.concatenate([np.array(data[l]["active"]) for l, *_ in OPS])
ally = np.concatenate([np.array(data[l]["kept"]) for l, *_ in OPS])
xmax = np.percentile(allx, 99.8); ymax = np.percentile(ally, 99.8)

for ax, (lab, m, comp, wl, kap) in zip(axes, OPS):
    a = np.array(data[lab]["active"], float); k = np.array(data[lab]["kept"], float)
    hb = ax.hexbin(a, k, gridsize=45, bins="log", cmap="viridis", mincnt=1,
                   extent=(0, xmax, 0, ymax))
    # median trend: kept vs active in active-bins
    bins = np.linspace(0, xmax, 12); idx = np.digitize(a, bins)
    bx = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    by = [np.median(k[idx == i + 1]) if (idx == i + 1).any() else np.nan for i in range(len(bins) - 1)]
    ax.plot(bx, by, "-", color="crimson", lw=2, label="median kept | active")
    ratio = np.median(k[a > 100] / a[a > 100])
    ax.set_title(f"{m}× noise   ({comp}× compression)\n{wl}, κ={kap}   |   median kept/active ≈ {ratio:.2f}",
                 fontsize=11)
    ax.set_xlabel("active signal pixels per chunk  (|x| > 3σ)")
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.grid(alpha=0.2)
    if ax is axes[0]:
        ax.set_ylabel("kept coefficients per chunk")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
cb = fig.colorbar(hb, ax=axes, fraction=0.025, pad=0.01)
cb.set_label("chunk count")
fig.suptitle("Active signal pixels vs. kept coefficients — per chunk across 100 events", fontsize=13)
fig.savefig(f"{OUT}/active_vs_kept.png", dpi=140, bbox_inches="tight")
print("wrote active_vs_kept.png")
for lab, m, comp, wl, kap in OPS:
    a = np.array(data[lab]["active"], float); k = np.array(data[lab]["kept"], float)
    print(f"{m}x noise: active med {np.median(a):.0f} (max {a.max():.0f}), kept med {np.median(k):.0f}, "
          f"kept/active med {np.median(k[a>100]/a[a>100]):.2f}")
