import torch, json
D="/global/cfs/cdirs/m5238/users/oalter/helix_runs/denoise/"
out={}
for d in ["M6d_ft_full_dec","M6dc_ft_full_dec_cond","lb_dec_hits","lb_dec_ct","lb_dec_cond_hits","lb_dec_cond_ct"]:
    ck=torch.load(D+d+"/best.pt",map_location="cpu",weights_only=False)
    m=ck["meta"]; a=m.get("args"); a=vars(a) if not isinstance(a,dict) else a
    out[d]={k:v for k,v in a.items() if not callable(v)}
    out[d]["_q0"]=m.get("q0")
json.dump(out,open("/global/cfs/cdirs/m5238/users/oalter/helix_runs/firstp/ckpt_args.json","w"),indent=1,default=str)
keys=sorted(set().union(*[set(v) for v in out.values()]))
for k in keys:
    vals=[str(out[d].get(k))[:40] for d in out]
    if len(set(vals))>1 or k in("truth_root","n_events","cov_per_sig","win_per_event","near_any_per_sig","win_near_per_event","runs","sig_cap"): print(k, vals)
