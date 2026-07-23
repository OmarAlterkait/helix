import json,numpy as np,matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
rows=[json.loads(l) for l in open("probe_3d_scaling.jsonl") if l.strip()]
agg=defaultdict(list)
for r in rows: agg[r['tag']].append(r['eval_per_event'])
runs=[("nll_long","NLL baseline","#d62728"),("nll_data","NLL +4x data","#2ca02c"),
      ("nll_deep","NLL +depth(24)","#9467bd"),("mse_long","MSE","#1f77b4")]
fig,ax=plt.subplots(figsize=(8.5,6))
for tag,lab,c in runs:
    xs=[150,300]; ys=[np.mean(agg[f"{tag}@{s}000"]) for s in (150,300)]
    es=[np.std(agg[f"{tag}@{s}000"]) for s in (150,300)]
    ax.errorbar(xs,ys,yerr=es,marker="o",color=c,lw=2.5,ms=9,capsize=4,label=lab)
ax.axhline(-0.346,color="gray",ls="--",lw=1.5,label="random-init floor (−0.35)")
ax.axhline(0.24,color="k",ls=":",lw=1,alpha=0.5,label="ref mae_nll_s0@150k (+0.24)")
ax.axhline(0,color="k",lw=0.5)
ax.annotate("MSE below random\n= no 3D",(300,-0.48),xytext=(210,-0.30),
            arrowprops=dict(arrowstyle="->"),color="#1f77b4",fontsize=10,fontweight="bold")
ax.annotate("3D IMPROVES with\ntraining (all NLL)",(300,0.25),xytext=(200,0.32),
            arrowprops=dict(arrowstyle="->"),color="#d62728",fontsize=10,fontweight="bold")
ax.set_xlabel("training step (k)");ax.set_ylabel("per-event within-plane 3D R² (context-removed)")
ax.set_xticks([150,300]);ax.set_title("CORRECTED 3D probe (early-stopped, per-event, random floor):\nNLL≫MSE, more iters help, data HURTS 3D",fontweight="bold")
ax.grid(alpha=0.3);ax.legend(loc="center right",fontsize=9)
fig.tight_layout();fig.savefig("probe_3d_corrected.png",dpi=120);print("wrote probe_3d_corrected.png")
