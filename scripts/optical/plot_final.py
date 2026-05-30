"""Honest event-total frontier: fixed-bit vs entropy-coded (both from final stage)."""
import json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/sdf/group/neutrino/omara/helix/temp/figures/big"
rows = [json.loads(l) for f in glob.glob(f"{OUT}/final_gpu*.jsonl") for l in open(f)]
rows = [r for r in rows if "error" not in r]
agg = {}
for r in rows:
    k = (r["wavelet"], r["level"], r["method"], r.get("func", ""), r.get("kappa", ""), r["bits"])
    agg.setdefault(k, []).append(r)
A = [dict(cfg=k, cf=np.mean([x["comp_fixed"] for x in v]), ce=np.mean([x["comp_entropy"] for x in v]),
          peak=np.mean([x["peak_err"] for x in v]), rms=np.mean([x["rms_vs_ref"] for x in v]))
     for k, v in agg.items()]


def front(items, compk, dk):
    pts = sorted(items, key=lambda r: -r[compk]); best = 1e9; fr = []
    for r in pts:
        if r[dk] < best:
            fr.append(r); best = r[dk]
    return sorted(fr, key=lambda r: r[compk])


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, dk, lab, gs in [(axes[0], "peak", "prompt-peak error (vs input)", (0.001, 0.005)),
                        (axes[1], "rms", "signal-shape distortion (rms vs denoised)", (0.01, 0.05))]:
    for compk, col, name in [("cf", "tab:blue", "fixed-bit"), ("ce", "tab:red", "entropy-coded")]:
        fr = front(A, compk, dk)
        ax.plot([r[compk] for r in fr], [r[dk] for r in fr], "o-", color=col, ms=4, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("compression  (total raw 16-bit / total stored bits)"); ax.set_ylabel(lab)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)
    for g in gs:
        ax.axhline(g, color="grey", ls=":", lw=0.8)
axes[0].set_title("Peak fidelity vs compression"); axes[1].set_title("Shape fidelity vs compression")
fig.suptitle("Optical compression — HONEST event-total frontier (100 events). "
             "VisuShrink denoise + quantization; entropy coding = red gain.", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/../final_frontier_master.png", dpi=130)
print("wrote final_frontier_master.png")

print("\n=== honest event-total: max compression at fidelity targets ===")
for tgt, dk in [(0.001, "peak"), (0.002, "peak"), (0.005, "peak"), (0.01, "rms"), (0.02, "rms"), (0.05, "rms")]:
    cf = max([r["cf"] for r in A if r[dk] <= tgt], default=0)
    ce = max([r["ce"] for r in A if r[dk] <= tgt], default=0)
    print(f"  {dk} <= {tgt*100:.2f}% :  fixed-bit {cf:5.1f}x   entropy-coded {ce:5.1f}x")
# best config at peak<0.2%
cand = [r for r in A if r["peak"] < 0.002]
if cand:
    b = max(cand, key=lambda r: r["ce"])
    print(f"\nrecommended (peak<0.2%): {b['cfg']}  -> fixed {b['cf']:.1f}x, entropy {b['ce']:.1f}x, "
          f"peak {b['peak']*100:.3f}%, rms {b['rms']*100:.2f}%")
