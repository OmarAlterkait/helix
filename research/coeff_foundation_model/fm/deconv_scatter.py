"""Per-band predicted-vs-true CHARGE coefficient density (asinh space) for the deconv model.
Visualizes where the R² comes from: tight diagonal = good recovery; D2 (finest) is looser."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import data as D
from data import DEV, N_BAND
from model import FMModel

D.init_pipeline_cpu()
CK = "ckpt_deconv_mae6k.pt"; NEV = 40; BN = ["A4", "D4", "D3", "D2"]
cdir = "../artifacts/fm_cache_tpc"; qdir = "../artifacts/fm_charge_tpc"
c = torch.load(CK, map_location=DEV)
m = FMModel(128, 4, 6, n_wirefeat=1, d=c["d"], blocks=c["blocks"], dec_blocks=c["dec_blocks"]).to(DEV); m.load_state_dict(c["model"]); m.eval()
print(f"loaded {CK} step {c['step']}")

# test events (the 6k run used first 20% as test -> events 0..NEV are test)
B = [D.get_cached_charge(f"{cdir}/ev_{i:05d}.npz", f"{qdir}/ev_charge_{i:05d}.npz") for i in range(NEV)]
sq = np.zeros(N_BAND); n = np.zeros(N_BAND)
for b in B:
    cc = b["target_charge"].cpu().numpy(); bd = b["band_id"][b["cell"]].cpu().numpy()
    np.add.at(sq, bd, cc**2); np.add.at(n, bd, 1)
S = torch.tensor(np.sqrt(sq/np.maximum(n,1))+1e-6, dtype=torch.float32, device=DEV)

P = {b: [] for b in range(4)}; T = {b: [] for b in range(4)}
with torch.no_grad():
    for bb in B:
        bb = {k:(v.to(DEV) if torch.is_tensor(v) else v) for k,v in bb.items()}
        msk = torch.zeros(int(bb["n_cells"]), dtype=torch.bool, device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, mu, _ = m(bb, msk)
        pred = mu[bb["cell"], bb["slot"]].float()
        tgt = torch.asinh(bb["target_charge"]/S[bb["band_id"][bb["cell"]]])
        bd = bb["band_id"][bb["cell"]].cpu().numpy()
        pr = pred.cpu().numpy(); tt = tgt.cpu().numpy()
        for b in range(4):
            mm = bd==b; P[b].append(pr[mm]); T[b].append(tt[mm])

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for b in range(4):
    p = np.concatenate(P[b]); t = np.concatenate(T[b])
    r2 = 1 - ((p-t)**2).mean()/(t.var()+1e-9)
    lim = np.percentile(np.abs(t), 99.5)
    ax[b].hist2d(t, p, bins=120, range=[[-lim,lim],[-lim,lim]], cmap="viridis", cmin=1)
    ax[b].plot([-lim,lim],[-lim,lim],"r--",lw=1)
    ax[b].set_title(f"{BN[b]}: R²={r2*100:.1f}%"); ax[b].set_xlabel("true charge (asinh)"); ax[b].set_ylabel("predicted")
    ax[b].set_aspect("equal")
fig.suptitle(f"Deconv: predicted vs true charge coeff per band (6000-ev model, step {c['step']})")
fig.tight_layout(); fig.savefig("deconv_scatter.png", dpi=105)
print("saved deconv_scatter.png")
