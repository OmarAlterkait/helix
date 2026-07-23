import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=[json.loads(l.split(' ',1)[1]) for l in open("results_3d_sweep.jsonl") if l.startswith("AWFINAL")]
def get(tag): 
    r=[x for x in rows if x["tag"]==tag]; return r[0]["within_plane"] if r else np.nan
L=[4,8,12]
series={"MSE @50k":"mse_50k","MSE @150k":"mse_150k","NLL @50k":"nll_50k","NLL @150k":"nll_150k"}
col={"MSE @50k":"#9ecae1","MSE @150k":"#1f77b4","NLL @50k":"#fcae91","NLL @150k":"#d62728"}
fig,ax=plt.subplots(figsize=(8,5.5))
for name,pre in series.items():
    y=[get(f"{pre}_L{l}") for l in L]
    ax.plot(L,y,"o-",color=col[name],lw=2.5 if "150k" in name else 1.5,ms=8 if "150k" in name else 5,
            label=name,ls="-" if "150k" in name else "--")
ax.axhline(0,color="k",lw=1,ls=":",alpha=0.6)
ax.annotate("NLL final layer:\nstrong 3D (+0.31)",(12,0.31),xytext=(8.5,0.2),
            arrowprops=dict(arrowstyle="->"),fontsize=10,color="#d62728",fontweight="bold")
ax.annotate("MSE: no 3D at any\nlayer (collapsed)",(8,-0.03),xytext=(5,-0.15),fontsize=10,color="#1f77b4")
ax.set_xlabel("encoder layer"); ax.set_ylabel("within-plane 3D R² (>0 = real 3D content)")
ax.set_xticks(L); ax.set_title("3D along-wire probe: MSE loses 3D, NLL develops it at the final layer",fontweight="bold")
ax.grid(alpha=0.3); ax.legend(title="objective @ step")
fig.tight_layout(); fig.savefig("probe_3d.png",dpi=120); print("wrote probe_3d.png")
