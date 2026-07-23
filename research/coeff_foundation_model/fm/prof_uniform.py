"""Uniform-COUNT group attention (FlatFormer / PTv3 lesson) vs full-attention base.
Sort tokens by a key, view(-1, g), plain BATCHED flash-SDPA (no masks, no ragged padding). Alternate:
  even layers -> sort by (plane, t)  (within-plane-ish groups, axial RoPE)
  odd  layers -> sort by (t)         (same-drift-time-across-planes groups, time-only RoPE)
This is the implementation the serialization lens says should finally CONVERT sparsity to wall-clock
(unlike ragged plane/time-bin grouping). Timing only (random token values).
"""
import sys, os, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import apply_rope, rope_angles
import data as D; D.init_pipeline_cpu()
dev = "cuda"


def uniform_attn(q, k, v, order, inv, g):
    T, h, hd = q.shape; npad = ((T + g - 1) // g) * g; nb = npad // g
    def grp(x):
        b = x.new_zeros(npad, h, hd); b[:T] = x[order]
        return b.view(nb, g, h, hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(grp(q), grp(k.to(q.dtype)), grp(v.to(q.dtype)))
    o = o.permute(0, 2, 1, 3).reshape(npad, h, hd)[:T]
    out = o.new_empty(T, h, hd); out[order] = o
    return out


class UBlock(nn.Module):
    def __init__(self, d, heads, ffn_mult=4):
        super().__init__(); self.h, self.hd = heads, d // heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x, ang_t, ang_w, order, inv, g):
        T, d = x.shape; h = self.n1(x)
        q, k, val = self.qkv(h).chunk(3, -1)
        q = apply_rope(q.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        k = apply_rope(k.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        val = val.view(T, self.h, self.hd)
        if order is None:
            o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], val.transpose(0, 1)[None])[0].transpose(0, 1)
        else:
            o = uniform_attn(q, k, val, order, inv, g)
        x = x + self.proj(o.reshape(T, d))
        return x + self.mlp(self.n2(x))


def timed(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / iters * 1e3, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--g", type=int, default=1024)
    ap.add_argument("--caps", default="8000,16000,24000,32000")
    a = ap.parse_args()
    B = D.get_cached(sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/fm_cache_tpc/ev_*.npz")))[3], device=dev)
    pa, ta, wa = B["plane_id"].to(dev), B["t_phys"].to(dev), B["wire_pos"].to(dev)
    Nf = len(pa); print(f"event N={Nf} d={a.d} g={a.g} blocks={a.blocks}", flush=True)
    lam_t, lam_w = (8.0, 4336.0), (32.0, 2048.0)
    base = nn.ModuleList([UBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)
    uni = nn.ModuleList([UBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)
    print(f"\n{'N':>6} | {'BASE ms':>9} {'GB':>6} | {'UNIFORM ms':>11} {'GB':>6} | speedup mem×", flush=True)
    for cap in [int(c) for c in a.caps.split(",")]:
        n = min(cap, Nf); pl, tp, wp = pa[:n], ta[:n], wa[:n]
        at = rope_angles(tp, a.d // a.heads, *lam_t); aw = rope_angles(wp, a.d // a.heads, *lam_w)
        op = torch.argsort(pl * 1e7 + tp); ip = torch.argsort(op)                # within-plane order
        ot = torch.argsort(tp); it = torch.argsort(ot)                           # drift-time order
        def run_base():
            x = torch.randn(n, a.d, device=dev, requires_grad=True); h = x
            for blk in base: h = blk(h, at, aw, None, None, a.g)
            h.sum().backward()
        def run_uni():
            x = torch.randn(n, a.d, device=dev, requires_grad=True); h = x
            for i, blk in enumerate(uni):
                if i % 2 == 0: h = blk(h, at, aw, op, ip, a.g)                    # within-plane groups, axial
                else:          h = blk(h, at, None, ot, it, a.g)                  # drift-time groups, time-only
            h.sum().backward()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bt, bm = timed(run_base); ut, um = timed(run_uni)
        print(f"{n:>6} | {bt:>7.1f}ms {bm:>5.2f} | {ut:>9.1f}ms {um:>5.2f} | {bt/ut:>5.2f}x {bm/um:>4.2f}x", flush=True)


if __name__ == "__main__":
    main()
