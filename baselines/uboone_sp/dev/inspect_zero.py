import json, sys, numpy as np
sys.path.insert(0, "baselines/uboone_sp")
d = sys.argv[1]; r = sys.argv[2]
W = json.load(open(f"{d}/windows_{r}.json"))
T = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
meta = []; wq = []
for ev in sorted(set(w["ev"] for w in W)):
    z = np.load(f"{T}/ev{ev:03d}.npz", allow_pickle=True)
    mm = json.loads(str(z["meta"])); meta += mm
    for n in range(len(mm)):
        wq.append(z["wq"][z["wid"] == n].reshape(16, 8))
rows = []
for w, m, q in zip(W, meta, wq):
    if w["kind"] == "iso" and w["score"] == 0:
        rows.append((w["ev"], w["g"], w["w0"], w["t0"], float(np.sum(m["E"])), float(q.sum()), m["cen"]))
rows.sort(key=lambda x: -x[4])
for x in rows[:25]: print(x)
print("t0 distribution of zero windows:", np.percentile([x[3] for x in rows], [0, 10, 50, 90, 100]))
print("w0 distribution:", np.percentile([x[2] for x in rows], [0, 10, 50, 90, 100]))
