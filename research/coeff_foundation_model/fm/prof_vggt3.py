"""Decisive test: FlexAttention with BOTH tiers block-local.
Within-plane layers run in PLANE-sorted order (plane mask = block-diagonal -> sparse).
Cross-plane layers run in TIME-sorted order (|dt|<=DT mask = banded -> sparse), permuting per layer.
If this hits the ~5x theoretical, the gate is realizable; if ~2x, sparse-kernel overhead is the ceiling at this N.
"""
import sys, os, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from model import apply_rope, rope_angles
import data as D
D.init_pipeline_cpu()
dev = "cuda"; _flex = torch.compile(flex_attention)


class FBlock(nn.Module):
    def __init__(self, d, heads, ffn_mult=4):
        super().__init__(); self.h, self.hd = heads, d // heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x, ang_t, ang_w, bm, perm, inv):
        # optionally permute into the order where `bm` is block-local, then permute back
        T, d = x.shape
        h = self.n1(x)
        q, k, val = self.qkv(h).chunk(3, -1)
        q = apply_rope(q.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        k = apply_rope(k.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        val = val.view(T, self.h, self.hd).to(q.dtype)
        if perm is not None: q, k, val = q[perm], k[perm], val[perm]
        o = _flex(q.transpose(0, 1)[None], k.transpose(0, 1)[None], val.transpose(0, 1)[None], block_mask=bm)[0].transpose(0, 1)
        if inv is not None: o = o[inv]
        x = x + self.proj(o.reshape(T, d))
        return x + self.mlp(self.n2(x))


def bmask(mod, Tpad): return create_block_mask(mod, 1, 1, Tpad, Tpad, device=dev)


def timed(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--dt", type=float, default=128.0)
    ap.add_argument("--caps", default="8000,16000,24000,32000")
    a = ap.parse_args()
    B = D.get_cached(sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../artifacts/fm_cache_tpc/ev_*.npz")))[3], device=dev)
    pa, ta, wa = B["plane_id"].to(dev), B["t_phys"].to(dev), B["wire_pos"].to(dev)
    Nf = len(pa); print(f"event N={Nf} d={a.d} dt={a.dt}", flush=True)
    lam_t, lam_w = (8.0, 4336.0), (32.0, 2048.0)
    base = nn.ModuleList([FBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)
    vg = nn.ModuleList([FBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)
    print(f"\n{'N':>6} | {'BASE ms':>9} | {'VGGT-gated ms':>13} | speedup | plane_sp cross_sp", flush=True)
    for cap in [int(c) for c in a.caps.split(",")]:
        n = min(cap, Nf); pl, tp, wp = pa[:n], ta[:n], wa[:n]
        npad = ((n + 127) // 128) * 128
        def pad(v, f):
            o = torch.full((npad,) + v.shape[1:], f, dtype=v.dtype, device=dev); o[:n] = v; return o
        # canonical = plane-sorted; permute to t-sorted for cross layers
        p2 = torch.argsort(pl * 1e7 + tp)                              # -> plane order
        plp = pl[p2]; tpp = tp[p2]; wpp = wp[p2]
        t2 = torch.argsort(tpp)                                        # plane-order -> t-order
        inv_t2 = torch.argsort(t2)
        at = rope_angles(tpp, a.d // a.heads, *lam_t); aw = rope_angles(wpp, a.d // a.heads, *lam_w)
        att = rope_angles(tpp[t2], a.d // a.heads, *lam_t)             # time-only RoPE, in t-order
        plpp = pad(plp, 99); tpt = pad(tpp[t2], 1e9)                   # padded to npad for block grids
        p2f = torch.cat([t2, torch.arange(n, npad, device=dev)]);  ivf = torch.cat([inv_t2, torch.arange(n, npad, device=dev)])
        atp = pad(at, 0.0); awp = pad(aw, 0.0); attp = pad(att, 0.0)
        bm_pl = bmask(lambda b, h, q, k: plpp[q] == plpp[k], npad)
        bm_cr = bmask(lambda b, h, q, k: (tpt[q] - tpt[k]).abs() <= a.dt, npad)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            def run_base():
                x = torch.randn(npad, a.d, device=dev, requires_grad=True); h = x
                bm = bmask(lambda b, hh, q, k: (q < n) & (k < n), npad)   # dense over real tokens
                for blk in base: h = blk(h, atp, awp, bm, None, None)
                h.sum().backward()
            def run_vg():
                x = torch.randn(npad, a.d, device=dev, requires_grad=True); h = x
                for i, blk in enumerate(vg):
                    if i % 2 == 0: h = blk(h, atp, awp, bm_pl, None, None)      # within-plane (plane order)
                    else:          h = blk(h, attp, None, bm_cr, p2f, ivf)      # cross-plane (permute to t order)
                h.sum().backward()
            bt = timed(run_base); vt = timed(run_vg)
        print(f"{n:>6} | {bt:>7.1f}ms | {vt:>11.1f}ms | {bt/vt:>5.2f}x | {bm_pl.sparsity():>6.1f}% {bm_cr.sparsity():>6.1f}%", flush=True)


if __name__ == "__main__":
    main()
