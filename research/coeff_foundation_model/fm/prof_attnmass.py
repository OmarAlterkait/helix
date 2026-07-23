"""Attention-mass audit: on a TRAINED full-attention model, what fraction of softmax mass falls inside
each candidate sparse-connection? Validates the proposed connections (within-plane / cross-plane drift
slab / cross-volume) BY MEASUREMENT before building anything. Monkeypatches SDPA to bin attention mass
by geometry for a random query subset, per encoder layer, averaged over events. No training.
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import data as D; D.init_pipeline_cpu()
import model as M
from model import FMModel
dev = "cuda"

G = {}   # per-token geometry (plane, vol, t) set before each event's forward
STATS = []
_real = F.scaled_dot_product_attention


def hook(q, k, v, attn_mask=None, scale=None, **kw):
    # q,k: (1,h,T,hd). Only audit the encoder self-attention (T == n_tokens).
    T = q.shape[2]
    if G.get("plane") is not None and T == len(G["plane"]):
        hd = q.shape[-1]; sc = scale if scale is not None else 1.0 / (hd ** 0.5)
        nq = min(2000, T); qi = torch.randperm(T, device=q.device)[:nq]
        s = (q[0, :, qi].float() @ k[0].transpose(-1, -2).float()) * sc          # (h,nq,T)
        w = s.softmax(-1).mean(0)                                                 # (nq,T) head-mean mass
        pl, vo, tt = G["plane"], G["vol"], G["t"]
        pq, vq, tq = pl[qi], vo[qi], tt[qi]
        sp = (pq[:, None] == pl[None, :]); sv = (vq[:, None] == vo[None, :])
        dt = (tq[:, None] - tt[None, :]).abs(); cross = (~sp) & sv                # same-vol, other plane
        row = lambda m: float((w * m).sum(-1).mean())
        STATS.append([row(sp), row(cross & (dt <= 16)), row(cross & (dt <= 64)),
                      row(cross & (dt <= 128)), row(cross), row(~sv)])
    return _real(q, k, v, attn_mask=attn_mask, scale=scale, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_nll_base_fix.pt"); ap.add_argument("--events", type=int, default=4)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=dev)
    heads = ck["d"] // ck.get("head_dim", 64) if ck.get("head_dim") else ck.get("heads", 8)
    m = FMModel(ck.get("n_slot", 128), 4, 6, n_wirefeat=1, d=ck.get("d", 512), blocks=ck.get("blocks", 12),
                dec_blocks=ck.get("dec_blocks", 4), heads=heads, film=tuple(ck.get("film", "band,plane,wire").split(",")),
                nll=ck.get("nll", True), cond=ck.get("cond", "film"), dec_mode=ck.get("dec_mode", "cross"),
                mup=ck.get("mup", True), d_base=ck.get("d_base", 128), wire_rope=ck.get("wire_rope", True)).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    print(f"loaded {a.ckpt} step {ck.get('step')} d={ck.get('d')} wire_rope={ck.get('wire_rope')}", flush=True)
    F.scaled_dot_product_attention = hook; M.F.scaled_dot_product_attention = hook
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/fm_cache_tpc/ev_*.npz")))
    for f in files[3:3 + a.events]:
        B = D.get_cached(f, device=dev)
        G["plane"] = B["plane_id"]; G["vol"] = B["plane_id"] // 3; G["t"] = B["t_phys"]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m.encode(B)
    F.scaled_dot_product_attention = _real
    s = np.array(STATS)                                                          # (layers*events, 6)
    lab = ["within-plane", "cross ±16t", "cross ±64t", "cross ±128t", "cross-plane(all dt, same vol)", "cross-VOLUME"]
    mean = s.mean(0)
    print("\n== Attention-mass audit (fraction of softmax mass, trained full-attn model) ==")
    for l, v in zip(lab, mean): print(f"  {l:32s} {v*100:5.1f}%")
    print(f"  {'--> within-plane + cross ±128t':32s} {(mean[0]+mean[3])*100:5.1f}%  (the proposed mask)")
    print(f"  {'residual (elsewhere)':32s} {(1-mean[0]-mean[4]-mean[5])*100:5.1f}%")


if __name__ == "__main__":
    main()
