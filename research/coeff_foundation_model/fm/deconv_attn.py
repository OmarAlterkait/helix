"""How much attention is same-plane vs cross-plane (and cross-volume)?
Inspect the trained deconv model: sample query tokens, recompute their attention over ALL
keys (post-RoPE) per encoder layer, bin attention mass by key plane. Compare to the uniform
baseline (= token-count fraction per plane). Same-plane >> baseline => attention is local;
cross-plane >> its baseline => model genuinely uses other planes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import data as D
from data import DEV
from model import FMModel, rope_angles, apply_rope

D.init_pipeline_cpu()
CK = "ckpt_deconv_mae6k.pt"; EV = 0; NQ = 600
c = torch.load(CK, map_location=DEV)
m = FMModel(128, 4, 6, n_wirefeat=1, d=c["d"], blocks=c["blocks"], dec_blocks=c["dec_blocks"]).to(DEV)
m.load_state_dict(c["model"]); m.eval()
B = D.get_cached_charge(f"../artifacts/fm_cache_tpc/ev_{EV:05d}.npz", f"../artifacts/fm_charge_tpc/ev_charge_{EV:05d}.npz")
B = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in B.items()}
print(f"loaded {CK} step {c['step']} | event {EV}, {int(B['n_cells'])} tokens")


@torch.no_grad()
def attn_by_plane(layers=(0, 3, 6, 9)):
    d = m.d; plane = B["plane_id"]; N = plane.shape[0]
    at = rope_angles(B["t_phys"], d // 4); aw = rope_angles(B["wire_pos"], d // 4)
    x = m.embed(torch.cat([B["inp"], B["occ"]], -1))
    if m.film is not None:
        g, b = m.film(B["band_id"], plane, B["wirefeat"]); x = g * x + b
    x = x + m.band_emb(B["band_id"]) + m.plane_emb(plane)
    g = torch.Generator(device=DEV).manual_seed(0)
    qi = torch.randperm(N, generator=g, device=DEV)[:NQ]
    qp = plane[qi]
    onehot = F.one_hot(plane, 6).float()                 # (N,6)
    pcount = onehot.sum(0) / N                            # uniform baseline per plane
    vol = (torch.arange(6, device=DEV) // 3)              # plane->volume
    out = {}
    for L, blk in enumerate(m.enc):
        if L in layers:
            h, hd = blk.h, blk.hd
            q, k, _ = blk.qkv(blk.n1(x)).chunk(3, -1)
            q = apply_rope(q.view(N, h, hd), at, aw); k = apply_rope(k.view(N, h, hd), at, aw)
            scores = torch.einsum("qhd,khd->hqk", q[qi], k) / (hd ** 0.5)
            a = scores.softmax(-1).mean(0)                # (NQ, N) avg over heads
            a2p = a @ onehot                              # (NQ, 6) mass to each plane
            same = a2p[torch.arange(NQ, device=DEV), qp]  # same-plane mass
            qvol = qp // 3
            volmass = torch.stack([a2p[:, vol == v].sum(1) for v in (0, 1)], 1)  # (NQ,2) mass per volume
            samevol = volmass[torch.arange(NQ, device=DEV), qvol]
            out[L] = dict(same_plane=float(same.mean()), same_vol=float(samevol.mean()),
                          baseline_same_plane=float(pcount[qp].mean()))
        x = blk(x, at, aw)
    return out, pcount.cpu().numpy()


res, pcount = attn_by_plane()
print("\nplane token fractions (uniform baseline):", pcount.round(3).tolist())
print("\nlayer  same-plane%  (uniform%)   same-volume%   cross-plane%   cross-volume%")
rows = []
for L, r in res.items():
    sp, sv, base = r["same_plane"], r["same_vol"], r["baseline_same_plane"]
    print(f"  {L:>2}    {sp*100:6.1f}   ({base*100:5.1f})     {sv*100:6.1f}        "
          f"{(1-sp)*100:6.1f}        {(1-sv)*100:6.1f}")
    rows.append((L, sp, sv, base))

fig, ax = plt.subplots(figsize=(7, 4.5))
Ls = [r[0] for r in rows]
ax.plot(Ls, [r[1]*100 for r in rows], "o-", label="same-plane attn %")
ax.plot(Ls, [r[2]*100 for r in rows], "s-", label="same-volume attn %")
ax.plot(Ls, [r[3]*100 for r in rows], "k--", label="uniform baseline (same-plane)")
ax.set_xlabel("encoder layer"); ax.set_ylabel("% of attention mass"); ax.set_ylim(0, 100)
ax.set_title(f"Attention locality by plane (deconv model, event {EV})"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("deconv_attn.png", dpi=110); print("saved deconv_attn.png")
