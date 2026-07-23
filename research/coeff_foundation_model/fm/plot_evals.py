"""Plot across-checkpoint evaluation: RankMe + eff_dims (representation) vs
var_explained + MSE + NLL (reconstruction on unseen), MSE-objective vs NLL-objective."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open("eval_ckpts.jsonl") if l.strip()]
def series(obj, key):
    r = sorted([x for x in rows if x["obj"] == obj], key=lambda x: x["step"])
    return np.array([x["step"] for x in r]), np.array([x[key] if x[key] is not None else np.nan for x in r])

C = {"mse": "#1f77b4", "nll": "#d62728"}
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

def panel(a, key, title, ylab, pct=False):
    for obj in ("mse", "nll"):
        s, v = series(obj, key)
        if np.all(np.isnan(v)): continue
        a.plot(s / 1000, v * (100 if pct else 1), "o-", color=C[obj], label=f"MAE-{obj.upper()}", lw=2, ms=5)
    a.set_title(title, fontweight="bold"); a.set_xlabel("step (k)"); a.set_ylabel(ylab)
    a.grid(alpha=0.3); a.legend()

panel(ax[0,0], "rankme", "RankMe (effective rank) — REPRESENTATION", "RankMe / 512")
ax[0,0].annotate("collapses as it trains", (0.5, 0.9), xycoords="axes fraction", color="gray", ha="center")
panel(ax[0,1], "eff_dims", "Effective dims (>1% max std)", "eff dims / 512")
panel(ax[0,2], "var_expl", "Val var-explained — RECONSTRUCTION (unseen)", "var-explained %", pct=True)
panel(ax[1,0], "mse", "Val MSE (unseen)  [lower=better]", "masked MSE")
panel(ax[1,1], "val_nll", "Val NLL (unseen, NLL run)  [lower=better]", "nats/coeff")

# the money panel: representation (RankMe) vs reconstruction (var_expl) — the anti-correlation
a = ax[1,2]
for obj in ("mse", "nll"):
    _, rk = series(obj, "rankme"); _, ve = series(obj, "var_expl")
    a.plot(ve * 100, rk, "o-", color=C[obj], label=f"MAE-{obj.upper()}", lw=2, ms=6)
    for i in (0, len(rk)-1):
        a.annotate(f"{[25,50,75,100,125,150][i if i==0 else 5]}k", (ve[i]*100, rk[i]), fontsize=7, color=C[obj])
a.set_title("Representation vs Reconstruction\n(better recon -> LOWER rank = collapse)", fontweight="bold")
a.set_xlabel("val var-explained %"); a.set_ylabel("RankMe"); a.grid(alpha=0.3); a.legend()

fig.suptitle("MAE standardized runs: representation collapses while reconstruction improves (MSE vs NLL)", fontweight="bold", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("eval_ckpts.png", dpi=120)
print("wrote eval_ckpts.png")
