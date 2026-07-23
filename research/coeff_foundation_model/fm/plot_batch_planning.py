"""Batch-size planning: LR law, steps/samples in 2h, and wall-clock to N epochs, at each batch.
Step-time model is an ESTIMATE (measured ~0.155s at B=2, 1 node; comm overhead at 32-64 GPU modeled)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

Bnoise = 31.0
lr2 = 1.6e-3
emax = lr2 * (2 + Bnoise) / 2                     # anchor lr(2)
lr = lambda B: emax * B / (B + Bnoise)

EV = 79700                                        # 80k train events / epoch
def tstep(B):                                     # seconds/step: 0.155 base + modeled multinode comm
    return 0.155 * (1 + 0.045 * np.log2(np.maximum(B, 1)))

Bs = np.array([2, 4, 8, 16, 32, 64, 128])
print(f"{'B':>4} {'lr':>9} {'t_step':>7} {'steps/2h':>9} {'samp/2h':>10} {'epochs/2h':>9} "
      f"{'h->30ep':>8} {'h->60ep':>8}")
for B in Bs:
    ts = tstep(B); s2h = 7200 / ts; samp2h = s2h * B; ep2h = samp2h / EV
    h30 = (30 * EV / B) * ts / 3600; h60 = (60 * EV / B) * ts / 3600
    print(f"{B:>4} {lr(B):>9.2e} {ts:>7.3f} {s2h:>9.0f} {samp2h/1e6:>9.2f}M {ep2h:>9.1f} {h30:>8.1f} {h60:>8.1f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(f"Batch-size planning (B_noise≈{Bnoise:.0f}, 80k dataset, step-time modeled)",
             fontsize=12, fontweight="bold")

# A: LR law
bb = np.logspace(np.log2(2), np.log2(256), 60, base=2)
ax[0].plot(bb, lr(bb) * 1e3, c="#1f77b4", lw=2, label=r"$lr=\varepsilon_{max}\,B/(B+B_{noise})$")
ax[0].plot(bb, lr2 * bb / 2 * 1e3, "--", c="#2ca02c", lw=1, label="linear ∝B")
ax[0].plot(bb, lr2 * np.sqrt(bb / 2) * 1e3, "--", c="#d62728", lw=1, label="√B")
ax[0].axhline(emax * 1e3, ls=":", c="gray", label=fr"$\varepsilon_{{max}}$={emax*1e3:.1f}e-3")
ax[0].axvline(Bnoise, c="orange", lw=1.2, alpha=.7)
for B in [2, 4, 8, 16, 32, 64]:
    ax[0].plot(B, lr(B) * 1e3, "o", c="#1f77b4"); ax[0].annotate(f"{lr(B)*1e3:.1f}", (B, lr(B) * 1e3),
               textcoords="offset points", xytext=(3, 5), fontsize=8)
ax[0].set(xscale="log", xlabel="batch B (events = GPUs)", ylabel="learning rate (×1e-3)",
          title="A. LR scaling law"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.25, which="both")

# B: wall-clock to epoch targets vs batch
for E, c in [(15, "#aec7e8"), (30, "#1f77b4"), (60, "#08306b")]:
    hrs = (E * EV / Bs) * tstep(Bs) / 3600
    ax[1].plot(Bs, hrs, "o-", c=c, label=f"{E} epochs")
ax[1].axhline(2, ls="--", c="red", lw=1, label="2 hours")
for B in Bs:
    ax[1].axvline(B, c="gray", lw=.3, alpha=.3)
ax[1].set(xscale="log", yscale="log", xlabel="batch B (events = GPUs)", ylabel="wall-clock hours",
          title="B. Time to reach N epochs of 80k"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.25, which="both")
ax[1].set_xticks(Bs); ax[1].set_xticklabels(Bs)

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_planning.png")
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(out, dpi=115); print("wrote", out)
