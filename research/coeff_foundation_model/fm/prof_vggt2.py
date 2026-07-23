"""VGGT-faithful profile: reshape/BATCHED grouped attention (like real VGGT frame/global), NOT masks.
  BASE          = full attention (current model)
  VGGT-vanilla  = frame(per-plane batched) + GLOBAL DENSE  -> should reproduce ~1.7x (the ceiling of plain VGGT)
  VGGT-gated    = frame(per-plane batched) + global as per-DRIFT-TIME-BIN batched (the epipolar gate) -> the real win
Grouped attention = gather tokens into (n_groups, max_group, h, hd) padded, batched SDPA with key-pad mask,
scatter back — exactly how VGGT's rearrange('b s p c') frame/global attention works, generalized to variable groups.
"""
import sys, os, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import apply_rope, rope_angles
import data as D
D.init_pipeline_cpu()
dev = "cuda"


def precompute_group(gid, G, device):
    """STATIC per-event indices for grouped attention (computed once, reused every layer)."""
    T = len(gid); counts = torch.bincount(gid, minlength=G); maxc = int(counts.max())
    order = torch.argsort(gid, stable=True); gid_s = gid[order]
    first = torch.zeros(G, dtype=torch.long, device=device); first[1:] = torch.cumsum(counts, 0)[:-1]
    pos = torch.arange(T, device=device) - first[gid_s]
    valid = torch.zeros(G, maxc, dtype=torch.bool, device=device); valid[gid_s, pos] = True
    return dict(order=order, gid_s=gid_s, pos=pos, maxc=maxc, G=G, valid=valid[:, None, None, :])


def grouped_sdpa(q, k, v, pre, scale=None):
    k, v = k.to(q.dtype), v.to(q.dtype)
    T, h, hd = q.shape; G, maxc = pre["G"], pre["maxc"]; o_, g_, p_ = pre["order"], pre["gid_s"], pre["pos"]
    bq = q.new_zeros(G, maxc, h, hd); bk = q.new_zeros(G, maxc, h, hd); bv = q.new_zeros(G, maxc, h, hd)
    bq[g_, p_] = q[o_]; bk[g_, p_] = k[o_]; bv[g_, p_] = v[o_]
    o = F.scaled_dot_product_attention(bq.permute(0, 2, 1, 3), bk.permute(0, 2, 1, 3), bv.permute(0, 2, 1, 3),
                                       attn_mask=pre["valid"], scale=scale).permute(0, 2, 1, 3)
    out = o.new_zeros(T, h, hd); out[o_] = o[g_, p_]
    return out


class GBlock(nn.Module):
    def __init__(self, d, heads, ffn_mult=4):
        super().__init__(); self.h, self.hd = heads, d // heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x, ang_t, ang_w, pre):
        T, d = x.shape; h = self.n1(x)
        q, k, val = self.qkv(h).chunk(3, -1)
        q = apply_rope(q.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        k = apply_rope(k.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        val = val.view(T, self.h, self.hd)
        if pre is None:                                            # dense full/global attention
            o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                               val.transpose(0, 1)[None])[0].transpose(0, 1)
        else:
            o = grouped_sdpa(q, k, val, pre)
        x = x + self.proj(o.reshape(T, d))
        return x + self.mlp(self.n2(x))


def timed(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--bin", type=float, default=256.0)
    ap.add_argument("--caps", default="8000,16000,24000,32000")
    a = ap.parse_args()
    B = D.get_cached(sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../artifacts/fm_cache_tpc/ev_*.npz")))[3], device=dev)
    pa, ta, wa = B["plane_id"].to(dev), B["t_phys"].to(dev), B["wire_pos"].to(dev)
    Nf = len(pa); print(f"event N={Nf} d={a.d} heads={a.heads} blocks={a.blocks} timebin={a.bin}", flush=True)
    lam_t, lam_w = (8.0, 4336.0), (32.0, 2048.0)
    mk = lambda: nn.ModuleList([GBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)
    base, van, gat = mk(), mk(), mk()

    print(f"\n{'N':>6} | {'BASE full':>11} | {'VGGT-vanilla':>13} | {'VGGT-gated':>12} | van×  gat×   bins avgbin", flush=True)
    for cap in [int(c) for c in a.caps.split(",")]:
        n = min(cap, Nf); pl, tp, wp = pa[:n], ta[:n], wa[:n]
        at = rope_angles(tp, a.d // a.heads, *lam_t); aw = rope_angles(wp, a.d // a.heads, *lam_w)
        tb = ((tp - tp.min()) / a.bin).long(); tb = torch.unique(tb, return_inverse=True)[1]   # contiguous time-bins
        nbin = int(tb.max()) + 1; avgbin = n / nbin
        pre_plane = precompute_group(pl, 6, dev)                        # STATIC groupings, computed once
        pre_tbin = precompute_group(tb, nbin, dev)

        def run_base():
            x = torch.randn(n, a.d, device=dev, requires_grad=True); h = x
            for blk in base: h = blk(h, at, aw, None)                  # every layer dense full attention
            h.sum().backward()
        def run_van():
            x = torch.randn(n, a.d, device=dev, requires_grad=True); h = x
            for i, blk in enumerate(van):
                h = blk(h, at, aw, pre_plane) if i % 2 == 0 else blk(h, at, None, None)   # frame | dense global
            h.sum().backward()
        def run_gat():
            x = torch.randn(n, a.d, device=dev, requires_grad=True); h = x
            for i, blk in enumerate(gat):
                h = blk(h, at, aw, pre_plane) if i % 2 == 0 else blk(h, at, None, pre_tbin)  # frame | time-bin global
            h.sum().backward()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bt, _ = timed(run_base); vt, _ = timed(run_van); gt, _ = timed(run_gat)
        print(f"{n:>6} | {bt:>9.1f}ms | {vt:>11.1f}ms | {gt:>10.1f}ms | {bt/vt:>4.2f}x {bt/gt:>4.2f}x  "
              f"{nbin:>5d} {avgbin:>6.0f}", flush=True)


if __name__ == "__main__":
    main()
