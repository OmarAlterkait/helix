import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
scal=[json.loads(l) for l in open("eval_ckpts.jsonl") if l.strip()]
orig=[json.loads(l) for l in open("eval_ckpts_150k.jsonl") if l.strip()]
def ser(rows,key,mk):
    r=sorted([x for x in rows if x['obj']==key],key=lambda x:x['step'])
    return [x['step']/1000 for x in r],[x[mk] for x in r]
fig,ax=plt.subplots(1,2,figsize=(14,5.5))
runs=[("nll_data","4x data","#2ca02c","-",3),("nll_deep","24 blocks","#9467bd","-",3),
      ("nll_long","NLL baseline (300k sched)","#d62728","-",2),("mse_long","MSE (300k sched)","#1f77b4","-",2)]
for name,lab,c,ls,lw in runs:
    for mk,a in [("rankme",ax[0]),("var_expl",ax[1])]:
        x,y=ser(scal,name,mk)
        if mk=="var_expl": y=[v*100 for v in y]
        a.plot(x,y,ls,color=c,lw=lw,ms=9,marker="o",label=lab)
# original 150k refs (dashed, thin)
for name,c in [("nll","#d62728"),("mse","#1f77b4")]:
    for mk,a in [("rankme",ax[0]),("var_expl",ax[1])]:
        x,y=ser(orig,name,mk)
        if mk=="var_expl": y=[v*100 for v in y]
        a.plot(x,y,"--",color=c,lw=1,alpha=0.5,marker="s",ms=4,label=f"orig {name} (150k)")
ax[0].set_title("RankMe (representation) — more DATA slows the collapse",fontweight="bold")
ax[0].set_ylabel("RankMe / 512"); ax[0].axhline(0,color="k",lw=0.5)
ax[1].set_title("Val var-explained (reconstruction, unseen)",fontweight="bold"); ax[1].set_ylabel("var-expl %")
for a in ax: a.set_xlabel("step (k)"); a.grid(alpha=0.3); a.legend(fontsize=8)
ax[0].annotate("4x data holds rank ~194\nvs baseline ~104",(100,194),xytext=(40,160),
               arrowprops=dict(arrowstyle="->"),color="#2ca02c",fontweight="bold",fontsize=9)
fig.suptitle("Scaling ablation (partial, to 100k of 300k): data preserves representation rank at no recon cost",fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig("scaling_rankme.png",dpi=120); print("wrote scaling_rankme.png")
