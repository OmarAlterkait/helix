"""Everything-over-training for the data-scaling runs: B=4 (20k/80k, lr3e-3, 1.2M) with B=2
references (lr1.6e-3, 600k). Shows 3D probe, var_expl, PLANE var_expl, val NLL, LR schedule,
train loss -- to judge convergence."""
import os, re, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C = {"b4_20k": "#1f77b4", "b4_80k": "#d62728", "b2_20k": "#9ecae1", "b2_80k": "#ff9896"}
LBL = {"b4_20k": "20k B=4 lr3e-3", "b4_80k": "80k B=4 lr3e-3",
       "b2_20k": "20k B=2 lr1.6e-3", "b2_80k": "80k B=2 lr1.6e-3"}


def probe_traj(path, prefix):
    out = {"20k": {}, "80k": {}}
    if not os.path.exists(os.path.join(HERE, path)): return out
    for l in open(os.path.join(HERE, path)):
        d = json.loads(l); t = d["tag"]
        if not t.startswith(prefix): continue
        run = "20k" if "20k" in t else "80k"
        s = int("".join(c for c in t.split("_")[-1] if c.isdigit()))
        out[run][s] = d["trained"]["fisher_r"]
    return out


def evals(tag):
    rows = {}
    for l in open(os.path.join(HERE, "fm_curve.jsonl"), errors="ignore"):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("tag") == tag: rows[r["step"]] = r
    ss = sorted(rows)
    g = lambda k: np.array([rows[s].get(k) if rows[s].get(k) is not None else np.nan for s in ss], float)
    return np.array(ss), g("var_expl"), g("plane_var_expl"), g("val_nll")


def steps_loss(tag):
    d = {}
    pat = re.compile(r"^\s*step\s+(\d+):\s+loss\s+([-\d.]+)")
    for f in glob.glob(os.path.join(HERE, "slurm_logs", f"{tag}_*.out")):
        for ln in open(f, errors="ignore"):
            m = pat.match(ln)
            if m: d[int(m.group(1))] = float(m.group(2))
    xs = sorted(d); y = np.array([d[s] for s in xs], float)
    if len(y) >= 25: y = np.convolve(y, np.ones(25)/25, mode="valid"); xs = xs[24:]
    return np.array(xs), y


def lr_sched(steps, peak, warm, total, floor_frac=0.1):
    lr = np.where(steps <= warm, peak * steps / warm,
                  peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + np.cos(np.pi * (steps - warm) / (total - warm)))))
    return lr


b4 = probe_traj("probe_b4.jsonl", "b4_"); b2 = probe_traj("probe_ds600.jsonl", "ds")
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
fig.suptitle("Data-scaling over training — B=4 runs (solid) vs B=2 references (dashed). Not yet converged.",
             fontsize=13, fontweight="bold")

# A: 3D probe
a = ax[0, 0]
for key, tr in [("b4_20k", b4["20k"]), ("b4_80k", b4["80k"])]:
    s = sorted(tr); a.plot(np.array(s)/1e3, [tr[x] for x in s], "o-", c=C[key], label=LBL[key])
for key, tr in [("b2_20k", b2["20k"]), ("b2_80k", b2["80k"])]:
    s = sorted(tr); a.plot(np.array(s)/1e3, [tr[x] for x in s], "s--", c=C[key], ms=4, label=LBL[key])
a.set(title="A. 3D probe (fisher_r on u) — THE metric", xlabel="step (k)", ylabel="fisher_r")
a.legend(fontsize=8); a.grid(alpha=.25)

# B/C/D: eval metrics
panels = [("var_expl", "B. masked var_expl (random)", 1, 0, 100), ("plane", "C. PLANE var_expl (triangulation)", 1, 1, 100),
          ("nll", "D. val NLL (lower=better)", 1, 2, 1)]
for tag, key in [("dscale600b4_20k", "b4_20k"), ("dscale600b4_80k", "b4_80k"),
                 ("dscale600_20k", "b2_20k"), ("dscale600_80k", "b2_80k")]:
    ss, ve, pv, nll = evals(tag)
    if not len(ss): continue
    st = "-" if key.startswith("b4") else "--"
    ax[1, 0].plot(ss/1e3, ve*100, st, c=C[key], lw=1.2, label=LBL[key])
    ax[1, 1].plot(ss/1e3, pv*100, st, c=C[key], lw=1.2, label=LBL[key])
    ax[1, 2].plot(ss/1e3, nll, st, c=C[key], lw=1.2, label=LBL[key])
ax[1, 0].set(title="B. masked var_expl (random)", xlabel="step (k)", ylabel="%")
ax[1, 1].set(title="C. PLANE var_expl (triangulation recon)", xlabel="step (k)", ylabel="%")
ax[1, 2].set(title="D. val NLL (lower = better)", xlabel="step (k)", ylabel="NLL")
for i in range(3): ax[1, i].legend(fontsize=7); ax[1, i].grid(alpha=.25)

# E: LR schedules + the 20k dip zone
a = ax[0, 1]
s4 = np.linspace(1, 1.2e6, 400); s2 = np.linspace(1, 6e5, 300)
a.plot(s4/1e3, lr_sched(s4, 3.0e-3, 3000, 1.2e6)*1e3, c="k", label="B=4: 3.0e-3 cosine/1.2M")
a.plot(s2/1e3, lr_sched(s2, 1.6e-3, 2000, 6e5)*1e3, "--", c="gray", label="B=2: 1.6e-3 cosine/600k")
a.axvspan(200, 450, color="#1f77b4", alpha=.12, label="20k B=4 3D dip zone")
a.set(title="E. LR schedules (dip aligns with peak LR)", xlabel="step (k)", ylabel="lr (x1e-3)")
a.legend(fontsize=8); a.grid(alpha=.25)

# F: train loss (B=4)
a = ax[0, 2]
for tag, key in [("dscale600b4_20k", "b4_20k"), ("dscale600b4_80k", "b4_80k")]:
    xs, y = steps_loss(tag)
    if len(xs): a.plot(np.array(xs)/1e3, y, c=C[key], lw=1, label=LBL[key])
a.set(title="F. train loss (smoothed)", xlabel="step (k)", ylabel="loss")
a.legend(fontsize=8); a.grid(alpha=.25)

out = os.path.join(HERE, "convergence_curves.png")
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out, dpi=115)
print("wrote", out)
