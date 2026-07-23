"""Coverage curve: on the trained full-attn model, what fraction of attention mass is captured by
(same-plane) UNION (drift-time window <= W, ANY volume) — the connectivity a drift-time uniform-group
of size g provides (g <-> W via token density). Tells us how BROAD groups must be to keep the mass.
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import data as D; D.init_pipeline_cpu()
import model as M
from model import FMModel
dev = "cuda"
G = {}; STATS = []; _real = F.scaled_dot_product_attention
WS = [75, 150, 300, 600, 1200, 2400]     # drift-time half-windows (ticks)


def hook(q, k, v, attn_mask=None, scale=None, **kw):
    T = q.shape[2]
    if G.get("plane") is not None and T == len(G["plane"]):
        hd = q.shape[-1]; sc = scale if scale is not None else 1.0 / (hd ** 0.5)
        nq = min(2000, T); qi = torch.randperm(T, device=q.device)[:nq]
        w = ((q[0, :, qi].float() @ k[0].transpose(-1, -2).float()) * sc).softmax(-1).mean(0)   # (nq,T)
        pl, tt = G["plane"], G["t"]; pq, tq = pl[qi], tt[qi]
        sp = (pq[:, None] == pl[None, :]); dt = (tq[:, None] - tt[None, :]).abs()
        row = [float((w * sp).sum(-1).mean())]                                   # same-plane alone
        for W in WS: row.append(float((w * (sp | (dt <= W))).sum(-1).mean()))     # same-plane U |dt|<=W (any vol)
        STATS.append(row)
    return _real(q, k, v, attn_mask=attn_mask, scale=scale, **kw)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="ckpt_nll_base_fix.pt"); ap.add_argument("--events", type=int, default=4)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=dev)
    heads = ck["d"] // ck.get("head_dim", 64) if ck.get("head_dim") else ck.get("heads", 8)
    m = FMModel(ck.get("n_slot", 128), 4, 6, n_wirefeat=1, d=ck.get("d", 512), blocks=ck.get("blocks", 12),
                dec_blocks=ck.get("dec_blocks", 4), heads=heads, film=tuple(ck.get("film", "band,plane,wire").split(",")),
                nll=ck.get("nll", True), cond=ck.get("cond", "film"), dec_mode=ck.get("dec_mode", "cross"),
                mup=ck.get("mup", True), d_base=ck.get("d_base", 128), wire_rope=ck.get("wire_rope", True)).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    F.scaled_dot_product_attention = hook; M.F.scaled_dot_product_attention = hook
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/fm_cache_tpc/ev_*.npz")))
    dens = []
    for f in files[3:3 + a.events]:
        B = D.get_cached(f, device=dev); n = len(B["plane_id"])
        dens.append(n / float((B["t_phys"].max() - B["t_phys"].min()).item()))
        G["plane"] = B["plane_id"]; G["t"] = B["t_phys"]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): m.encode(B)
    F.scaled_dot_product_attention = _real
    s = np.array(STATS).mean(0); tpt = np.mean(dens)
    print(f"loaded {a.ckpt}; tokens/tick={tpt:.1f}  (drift-group g <-> W: g ~= 2*W*tpt)")
    print("\n== Coverage: mass in (same-plane U |dt|<=W, any volume) ==")
    print(f"  {'same-plane only':22s} {s[0]*100:5.1f}%")
    for W, cov in zip(WS, s[1:]):
        g = int(2 * W * tpt)
        print(f"  {'+ |dt|<=' + str(W) + ' ticks':22s} {cov*100:5.1f}%   (uniform group g~{g})")
    print(f"  {'dense (full attn)':22s} 100.0%")


if __name__ == "__main__":
    main()
