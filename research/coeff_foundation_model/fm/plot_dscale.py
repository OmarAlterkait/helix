"""Multi-panel training/eval curves for the data-scaling runs (dscale600_20k vs 80k)."""
import os, re, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {"20k train": ("dscale600_20k", "#1f77b4"), "80k train": ("dscale600_80k", "#d62728")}
STEP_RE = re.compile(r"^\s*step\s+(\d+):\s+loss\s+([-\d.]+)\s+\(bce\s+([-\d.]+)\s+val\s+([-\d.]+)\).*?gn=([-\d.]+)")


def load_steps(tag):
    """Concatenate step-loss lines across all (preemption-resumed) log files, dedup by step."""
    d = {}
    for f in glob.glob(os.path.join(HERE, "slurm_logs", f"{tag}_*.out")):
        for ln in open(f, errors="ignore"):
            m = STEP_RE.match(ln)
            if m:
                s = int(m.group(1))
                d[s] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
    xs = sorted(d)
    a = np.array([d[s] for s in xs])
    return np.array(xs), a  # cols: loss, bce, val, gn


def load_evals(tag):
    rows = []
    for ln in open(os.path.join(HERE, "fm_curve.jsonl"), errors="ignore"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("tag") == tag:
            rows.append(r)
    rows.sort(key=lambda r: r["step"])
    # dedup by step (last wins)
    seen = {r["step"]: r for r in rows}
    rows = [seen[s] for s in sorted(seen)]
    g = lambda k: np.array([r.get(k) if r.get(k) is not None else np.nan for r in rows], float)
    return np.array([r["step"] for r in rows]), g("var_expl"), g("mse"), g("val_nll"), g("plane_var_expl")


def smooth(x, y, k=15):
    """Return (x,y) both trimmed to the valid-convolution length so they align."""
    if len(y) < k:
        return x, y
    ys = np.convolve(y, np.ones(k) / k, mode="valid")
    xs = x[k - 1:]
    return xs, ys


fig, ax = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Data-scaling runs: grouped-serial, plane_frac 0.1, lr 1.6e-3, d512  (20k vs 80k train events)",
             fontsize=13, fontweight="bold")

for label, (tag, c) in RUNS.items():
    xs, A = load_steps(tag)
    ex, ve, mse, vnll, pve = load_evals(tag)
    if len(xs):
        x0, y0 = smooth(xs, A[:, 0]);       ax[0, 0].plot(x0 / 1e3, y0, c=c, label=label, lw=1.2)        # total loss
        xv, yv = smooth(xs, A[:, 2]);       ax[0, 1].plot(xv / 1e3, yv, c=c, label=label, lw=1.2)        # val (recon NLL)
        xb, yb = smooth(xs, A[:, 1]);       ax[0, 1].plot(xb / 1e3, yb, c=c, ls="--", lw=1.0, alpha=.7)  # bce (occupancy)
        xg, yg = smooth(xs, A[:, 3], 25);   ax[1, 2].plot(xg / 1e3, yg, c=c, label=label, lw=1.0, alpha=.8)  # grad norm
    if len(ex):
        ax[0, 2].plot(ex / 1e3, ve * 100, c=c, marker="o", ms=3, label=label)       # val var_expl
        ax[1, 0].plot(ex / 1e3, vnll, c=c, marker="o", ms=3, label=label)           # val NLL
        ax[1, 1].plot(ex / 1e3, pve * 100, c=c, marker="o", ms=3, label=label)      # PLANE var_expl

ax[0, 0].set(title="A. Training loss (total)", xlabel="step (k)", ylabel="loss"); ax[0, 0].legend()
ax[0, 1].set(title="B. Loss components: val=recon-NLL (solid), bce=occupancy (dashed)", xlabel="step (k)", ylabel="loss")
ax[0, 2].set(title="C. Validation masked var_expl", xlabel="step (k)", ylabel="var_expl (%)"); ax[0, 2].legend()
ax[1, 0].set(title="D. Validation NLL (held-out)", xlabel="step (k)", ylabel="val NLL"); ax[1, 0].legend()
ax[1, 1].set(title="E. PLANE var_expl (triangulation reconstruction)", xlabel="step (k)", ylabel="plane var_expl (%)"); ax[1, 1].legend()
ax[1, 2].set(title="F. Grad-norm (stability)", xlabel="step (k)", ylabel="||g||"); ax[1, 2].legend()
for a in ax.flat:
    a.grid(alpha=.25)

out = os.path.join(HERE, "dscale_curves.png")
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(out, dpi=110)
print("wrote", out)
# also print the latest numbers for the write-up
for label, (tag, _) in RUNS.items():
    ex, ve, mse, vnll, pve = load_evals(tag)
    if len(ex):
        print(f"{label}: last step {int(ex[-1])}  var_expl={ve[-1]*100:.1f}%  val_nll={vnll[-1]:.3f}  plane_ve={pve[-1]*100:.1f}%")
