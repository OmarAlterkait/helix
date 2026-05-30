"""Clean Pareto frontier (corrected sigma). Frontier curve + faint cloud +
a few labeled operating points. Reads depth jsonl (fixed-bit event-total rate)."""
import json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/sdf/group/neutrino/omara/helix/temp/figures/big"


def load(stage, compkey):
    rows = [json.loads(l) for f in glob.glob(f"{OUT}/{stage}_gpu*.jsonl") for l in open(f)]
    rows = [r for r in rows if "error" not in r and compkey in r]
    agg = {}
    for r in rows:
        k = (r["wavelet"], r["level"], r["method"], r.get("func", ""), r.get("kappa", ""), r["bits"])
        agg.setdefault(k, []).append(r)
    A = []
    for k, v in agg.items():
        A.append(dict(c=np.mean([x[compkey] for x in v]), peak=np.mean([x["peak_err"] for x in v]),
                      rms=np.mean([x["rms_vs_ref"] for x in v]), kappa=k[4], bits=k[5], wl=k[0]))
    return A


def frontier(A, dk):
    pts = sorted(A, key=lambda r: -r["c"]); best = 1e9; fr = []
    for r in pts:
        if r[dk] < best:
            fr.append(r); best = r[dk]
    return sorted(fr, key=lambda r: r["c"])


fix = load("depth", "comp")
nev = len({json.loads(l)["event"] for f in glob.glob(f"{OUT}/depth_gpu*.jsonl") for l in open(f)})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, dk, lab in [(axes[0], "peak", "prompt-peak error  [%]"),
                    (axes[1], "rms", "signal-shape error  [%]")]:
    ax.scatter([r["c"] for r in fix], [100 * r[dk] for r in fix], s=5, c="0.82", zorder=1)  # faint cloud
    fr = frontier(fix, dk)
    ax.plot([r["c"] for r in fr], [100 * r[dk] for r in fr], "-o", color="tab:blue", ms=4, lw=2,
            label="VisuShrink + quantization", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("compression  (×)"); ax.set_ylabel(lab)
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=9, loc="upper left")

# annotate 3 operating points on the peak panel
frp = frontier(fix, "peak")
def near(target):
    return min(frp, key=lambda r: abs(r["c"] - target))
for tgt, name in [(28, "high-fidelity"), (45, "balanced"), (60, "aggressive")]:
    r = near(tgt)
    axes[0].annotate(f"{name}\n{r['c']:.0f}×, {100*r['peak']:.2f}%",
                     (r["c"], 100 * r["peak"]), textcoords="offset points", xytext=(6, -22),
                     fontsize=8, color="navy",
                     arrowprops=dict(arrowstyle="->", color="navy", lw=0.7))
fig.suptitle(f"Optical compression — Pareto frontier (corrected σ, {nev} events, VisuShrink-hard coif/sym)",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/../pareto_clean.png", dpi=130)
print(f"wrote pareto_clean.png  ({nev} events, {len(fix)} configs)")
print("\nfrontier (peak):")
for r in frontier(fix, "peak"):
    if 0.0002 < r["peak"] < 0.01:
        print(f"  {r['c']:5.0f}x  peak {100*r['peak']:.3f}%  rms {100*r['rms']:.2f}%  ({r['wl']} k{r['kappa']} B{r['bits']})")
