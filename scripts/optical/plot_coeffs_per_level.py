"""Per-DWT-level coefficient distributions (default optical method, κ=1.2).
Reads big/coeffs_per_level.json (restyle without recomputing). Writes two PNGs:
  coeffs_per_level_dist.png   — kept coefficients per chunk, distribution per band
  coeffs_per_level_budget.png — survival fraction + share of the kept budget per band
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/sdf/group/neutrino/omara/helix/temp/figures"
D = json.load(open(f"{OUT}/big/coeffs_per_level.json"))
rows, KAP = D["rows"], D["kappa"]
names = rows[0]["names"]; nb = len(names)
PKMIN = D["signal_peak_min"]

# ---- gather per-band, signal chunks only (|x|max > PKMIN) ----
per_band = [[] for _ in range(nb)]          # kept counts per chunk
agg_kept = np.zeros(nb); agg_cap = np.zeros(nb); nsig = 0
noise_kept = [];
for r in rows:
    kept = np.array(r["kept"]); peak = np.array(r["peak"]); bl = np.array(r["bandlen"])
    m = peak > PKMIN
    for b in range(nb):
        per_band[b].extend(kept[m, b].tolist())
    agg_kept += kept[m].sum(0); agg_cap += bl * int(m.sum()); nsig += int(m.sum())
    noise_kept.extend(kept[~m].sum(1).tolist())

# order finest -> coarsest -> approx for a left-to-right cascade reading
order = list(range(nb - 1, 0, -1)) + [0]    # D1, D2, ..., D10, A10
lab = [names[i] for i in order]
dist = [np.array(per_band[i]) for i in order]
frac = (agg_kept / np.maximum(agg_cap, 1))[order]
share = (agg_kept / max(agg_kept.sum(), 1))[order]
cap = (agg_cap / max(nsig, 1))[order]
mean_kept = (agg_kept / max(nsig, 1))[order]

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9, "font.family": "DejaVu Sans"})
cmap = plt.cm.viridis(np.linspace(0.12, 0.9, nb))[::-1]   # fine=dark -> coarse=bright

# ============================ Figure 1: distribution per band ============================
fig, ax = plt.subplots(figsize=(9.2, 5.4))
xs = np.arange(nb)
bp = ax.boxplot(dist, positions=xs, widths=0.62, whis=(5, 95), showfliers=False,
                patch_artist=True, medianprops=dict(color="k", lw=1.3),
                whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
for patch, c in zip(bp["boxes"], cmap):
    patch.set_facecolor(c); patch.set_edgecolor("0.3"); patch.set_alpha(0.85)
ax.plot(xs, [d.mean() for d in dist], "D", ms=5, color="#c0392b", zorder=5, label="mean")
ax.set_yscale("symlog", linthresh=1.0)
ax.set_xticks(xs); ax.set_xticklabels(lab)
ax.set_ylim(-0.4, max(d.max() for d in dist) * 1.3)
ax.set_xlabel("decomposition band   (finest D1  →  coarsest A10)")
ax.set_ylabel("kept coefficients per chunk")
ax.set_title(f"Surviving wavelet coefficients per level  —  default method (coif3, L10, VisuShrink-hard κ={KAP})",
             fontsize=12, pad=10)
ax.text(0.015, 0.97, f"{nsig:,} signal chunks  (|x|$_{{max}}$ > {PKMIN:.0f} ADC)\n"
                     f"box = IQR, whiskers 5–95%",
        transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.7", lw=0.7))
ax.legend(loc="upper right", frameon=True, fontsize=10)
ax.grid(axis="y", alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/coeffs_per_level_dist.png", dpi=150)
print("wrote coeffs_per_level_dist.png")

# ============================ Figure 2: budget (fraction + share) ========================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
xs = np.arange(nb)

# (a) mean kept vs band capacity, with survival fraction annotated
a1.bar(xs, cap, color="0.85", width=0.74, label="band capacity")
a1.bar(xs, mean_kept, color=cmap, width=0.74, edgecolor="0.3", label="kept")
a1.set_yscale("log")
for x, f, mk in zip(xs, frac, mean_kept):
    a1.text(x, mk * 1.25, f"{f*100:.0f}%", ha="center", va="bottom", fontsize=8, color="#16324f")
a1.set_xticks(xs); a1.set_xticklabels(lab, fontsize=9.5)
a1.set_ylabel("coefficients per chunk (log)")
a1.set_xlabel("band   (finest → coarsest)")
a1.set_title("kept vs available per level  (% = survival fraction)", fontsize=11)
a1.legend(loc="upper right", frameon=True, fontsize=9)
a1.grid(axis="y", alpha=0.2)

# (b) share of the kept budget
a2.bar(xs, share * 100, color=cmap, width=0.74, edgecolor="0.3")
for x, s in zip(xs, share):
    if s > 0.02:
        a2.text(x, s * 100 + 0.4, f"{s*100:.0f}", ha="center", va="bottom", fontsize=8.5)
a2.set_xticks(xs); a2.set_xticklabels(lab, fontsize=9.5)
a2.set_ylabel("share of kept coefficients  [%]")
a2.set_xlabel("band   (finest → coarsest)")
a2.set_title("where the kept budget goes", fontsize=11)
a2.grid(axis="y", alpha=0.2)
for ax in (a1, a2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle(f"Multi-scale coefficient budget — default optical method (κ={KAP}); "
             f"mean {agg_kept.sum()/max(nsig,1):.0f} kept / signal chunk", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/coeffs_per_level_budget.png", dpi=150)
print("wrote coeffs_per_level_budget.png")
print(f"noise chunks: {len(noise_kept):,}, median kept {np.median(noise_kept):.0f}, mean {np.mean(noise_kept):.0f}")
