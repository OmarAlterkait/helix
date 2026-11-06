import json, numpy as np, collections
W = json.load(open("/pscratch/sd/o/oalter/helix_work/denoise/results/dn_near_debug_ab_ft1024_win_trained_windows.json"))
print(len(W), collections.Counter(w["kind"] for w in W), W[0], [w for w in W if w["kind"] == "bgn"][:1])
print("iso with snr", sum("snr" in w for w in W if w["kind"] == "iso"), "evs", min(w["ev"] for w in W), max(w["ev"] for w in W))
z = np.load("/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near/ev300.npz", allow_pickle=True); print(z.files, json.loads(str(z["meta"]))[:2], len(z["wkey"]))
A = np.load("/pscratch/sd/o/oalter/helix_work/denoise/noise_vs_hits.npz"); print(A.files, len(A["ev"]))
