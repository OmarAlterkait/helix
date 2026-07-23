"""Speed vs group-size g (the other axis of the coverage-vs-speed frontier), at full-event N.
Configs: BASE (dense) | VGGT-vanilla (within-plane groups + DENSE global) | UNIFORM-g (within-plane + drift-time groups of size g).
Pairs with prof_cov.py's coverage curve to trace the frontier: bigger g -> more coverage, less speedup; g=dense -> vanilla VGGT.
"""
import sys, os, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import apply_rope, rope_angles
import data as D; D.init_pipeline_cpu()
dev = "cuda"


def uniform_attn(q, k, v, order, g):
    T, h, hd = q.shape; npad = ((T + g - 1) // g) * g; nb = npad // g
    def grp(x):
        b = x.new_zeros(npad, h, hd); b[:T] = x[order]; return b.view(nb, g, h, hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(grp(q), grp(k.to(q.dtype)), grp(v.to(q.dtype)))
    o = o.permute(0, 2, 1, 3).reshape(npad, h, hd)[:T]
    out = o.new_empty(T, h, hd); out[order] = o; return out


class UBlock(nn.Module):
    def __init__(self, d, heads, ffn_mult=4):
        super().__init__(); self.h, self.hd = heads, d // heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x, ang_t, ang_w, order, g):
        T, d = x.shape; h = self.n1(x)
        q, k, val = self.qkv(h).chunk(3, -1)
        q = apply_rope(q.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        k = apply_rope(k.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        val = val.view(T, self.h, self.hd)
        if order is None:
            o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], val.transpose(0, 1)[None])[0].transpose(0, 1)
        else:
            o = uniform_attn(q, k, val, order, g)
        x = x + self.proj(o.reshape(T, d)); return x + self.mlp(self.n2(x))


def timed(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / iters * 1e3, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--gs", default="512,1024,2048,4096,8192")
    a = ap.parse_args()
    B = D.get_cached(sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/fm_cache_tpc/ev_*.npz")))[3], device=dev)
    pl, tp, wp = B["plane_id"].to(dev), B["t_phys"].to(dev), B["wire_pos"].to(dev)
    n = len(pl); print(f"N={n} d={a.d} blocks={a.blocks}", flush=True)
    lam_t, lam_w = (8.0, 4336.0), (32.0, 2048.0)
    at = rope_angles(tp, a.d // a.heads, *lam_t); aw = rope_angles(wp, a.d // a.heads, *lam_w)
    op = torch.argsort(pl * 1e7 + tp); ot = torch.argsort(tp)
    net = nn.ModuleList([UBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)

    def run(mode, g=0):
        x = torch.randn(n, a.d, device=dev, requires_grad=True); h = x
        for i, blk in enumerate(net):
            if mode == "base":     h = blk(h, at, aw, None, 0)
            elif i % 2 == 0:       h = blk(h, at, aw, op, max(g, 512))            # within-plane groups
            elif mode == "vggt":   h = blk(h, at, None, None, 0)                  # dense global
            else:                  h = blk(h, at, None, ot, g)                    # drift-time groups
        h.sum().backward()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        bt, _ = timed(lambda: run("base"))
        vt, _ = timed(lambda: run("vggt"))
        print(f"\n  {'config':16s} {'ms':>8} {'speedup':>8}")
        print(f"  {'BASE dense':16s} {bt:>7.1f} {1.0:>7.2f}x")
        print(f"  {'VGGT dense-global':16s} {vt:>7.1f} {bt/vt:>7.2f}x   (100% coverage)")
        for g in [int(x) for x in a.gs.split(",")]:
            gt, _ = timed(lambda: run("uni", g))
            print(f"  {'uniform g=' + str(g):16s} {gt:>7.1f} {bt/gt:>7.2f}x", flush=True)


if __name__ == "__main__":
    main()
