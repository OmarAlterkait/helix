import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
r = [json.loads(l) for l in open(os.path.join(HERE, "bnoise2.jsonl"))][-1]
Gsq, tr, Bn = r["Gsq"], r["trSigma"], r["Bnoise"]
lo, hi = r["ci68"]
Bs = np.array([int(k) for k in r["curve"]]); order = np.argsort(Bs); Bs = Bs[order]
direct = np.array([r["curve"][str(b)]["direct"] for b in Bs])
se = np.array([r["curve"][str(b)]["se"] for b in Bs])
closed = np.array([r["curve"][str(b)]["closed"] for b in Bs])

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(f"Gradient noise scale — 80k @600k, M=384 pool, random-0.75\n"
             f"B_noise = {Bn:.0f} events  (bootstrap 68% CI [{lo:.0f}, {hi:.0f}])",
             fontsize=12, fontweight="bold")

# Panel 1: E[|G_B|^2] vs B (log-log) with 1/B law + |G|^2 floor + B_noise band
ax[0].errorbar(Bs, direct, yerr=se, fmt="o", c="#1f77b4", ms=5, label="measured (form batches)", zorder=3)
bb = np.logspace(0, np.log10(300), 100)
ax[0].plot(bb, Gsq + tr / bb, c="#d62728", lw=1.5, label=r"fit $|G|^2 + \mathrm{tr}\Sigma/B$")
ax[0].axhline(Gsq, ls="--", c="gray", lw=1, label=fr"$|G|^2$ floor = {Gsq:.1f}")
ax[0].axvspan(lo, hi, color="orange", alpha=.18, label=f"B_noise CI [{lo:.0f},{hi:.0f}]")
ax[0].axvline(Bn, c="orange", lw=1.5)
ax[0].set(xscale="log", yscale="log", xlabel="batch size B (events)", ylabel=r"$E\,[\,|G_B|^2\,]$",
          title="A. Batch gradient magnitude vs batch size")
ax[0].legend(fontsize=9); ax[0].grid(alpha=.25, which="both")

# Panel 2: efficiency — steps-to-target factor S(B)/S_min = 1 + B_noise/B, and per-doubling speedup
sf = 1 + Bn / Bs
ax[1].plot(Bs, sf, "o-", c="#2ca02c", label=r"steps-to-target $\propto 1+B_{noise}/B$")
for B in [2, 4, 8, 16, 32]:
    s = 1 + Bn / B
    ax[1].annotate(f"B={B}", (B, s), textcoords="offset points", xytext=(4, 6), fontsize=8)
ax[1].axvline(Bn, c="orange", lw=1.5, label=f"B_noise={Bn:.0f} (returns halve here)")
ax[1].axvspan(lo, hi, color="orange", alpha=.18)
ax[1].axvspan(2, 8, color="#2ca02c", alpha=.08, label="our regime (B=2..8): near-linear")
ax[1].set(xscale="log", yscale="log", xlabel="batch size B (events)",
          ylabel="relative steps-to-target", title="B. Efficiency: fewer steps as batch grows")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.25, which="both")

out = os.path.join(HERE, "bnoise_curve.png")
plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig(out, dpi=115)
print("wrote", out)
