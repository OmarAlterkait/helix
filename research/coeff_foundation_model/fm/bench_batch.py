"""Batching benchmark at the REAL ~32k tokens/event. Three regimes:
  base    : 1 event, fwd+bwd (current training regime) -> ms, peak mem, tokens/s
  accum K : K events, backward+FREE per event (grad accumulation) -> memory = 1 event, time = K*base
  varlen K: pack K events into one graph, flash varlen (block-diagonal) -> one fwd+bwd; OOMs when
            K events' activations exceed VRAM. Tells us if packing is even feasible / faster.
Sets expandable_segments to reduce fragmentation.
"""
import sys, os, time, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, glob
import data as D
from data import DEV
from model import Block, rope_angles, apply_rope
from flash_attn import flash_attn_varlen_func


def varlen_fwd(blocks, x, ang_t, ang_w, cu, max_s):
    for blk in blocks:
        T, d = x.shape; h, hd = blk.h, blk.hd
        q, k, v = blk.qkv(blk.n1(x)).chunk(3, -1)
        q = apply_rope(q.view(T, h, hd), ang_t, ang_w); k = apply_rope(k.view(T, h, hd), ang_t, ang_w)
        o = flash_attn_varlen_func(q.bfloat16(), k.bfloat16(), v.view(T, h, hd).bfloat16(),
                                   cu, cu, max_s, max_s, softmax_scale=hd ** -0.5)
        x = x + blk.proj(o.reshape(T, d).float())
        x = x + blk.mlp(blk.n2(x))
    return x


def timed(fn, iters):
    for _ in range(2): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--Ks", default="1,2,3,4")
    args = ap.parse_args()
    D.init_pipeline_cpu()
    d, L = args.d, args.blocks
    blocks = nn.ModuleList(Block(d, args.heads) for _ in range(L)).to(DEV)
    opt = torch.optim.SGD(blocks.parameters(), lr=0.0)
    fs = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[:8]
    F = []
    for f in fs:
        B = D.get_cached(f); n = int(B["n_cells"])
        F.append((torch.randn(n, d, device=DEV), rope_angles(B["t_phys"].to(DEV), d // 4),
                  rope_angles(B["wire_pos"].to(DEV), d // 4), n))
    print(f"d={d} L={L} heads={args.heads} | params(blocks)={sum(p.numel() for p in blocks.parameters())/1e6:.1f}M")

    # base: single event
    x0, at0, aw0, n0 = F[0]
    def base():
        opt.zero_grad(set_to_none=True)
        x = x0.detach().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):     # match real training precision
            for blk in blocks: x = blk(x, at0, aw0)
        x.float().pow(2).mean().backward()
    dt, pk = timed(base, args.iters)
    print(f"\nBASE 1 event ({n0} tok): {dt*1000:.0f} ms  {n0/dt:.0f} tok/s  peak {pk:.1f} GB")

    print(f"\n{'K':>2} {'mode':>7} {'tokens':>8} {'ms/step':>8} {'tok/s':>9} {'peakGB':>7} {'note':>6}")
    for K in [int(k) for k in args.Ks.split(",")]:
        sub = F[:K]; tot = sum(s[3] for s in sub)
        # grad-accum: per-event backward+free
        def accum():
            opt.zero_grad(set_to_none=True)
            for x_, at_, aw_, _ in sub:
                x = x_.detach().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for blk in blocks: x = blk(x, at_, aw_)
                x.float().pow(2).mean().backward()
        dt, pk = timed(accum, max(2, args.iters // K))
        print(f"{K:>2} {'accum':>7} {tot:>8} {dt*1000:>8.0f} {tot/dt:>9.0f} {pk:>7.1f}")
        # varlen pack
        try:
            xp_base = torch.cat([s[0] for s in sub]); ang_t = torch.cat([s[1] for s in sub]); ang_w = torch.cat([s[2] for s in sub])
            cu = torch.tensor(np.cumsum([0] + [s[3] for s in sub]), dtype=torch.int32, device=DEV); ms = max(s[3] for s in sub)
            def vl():
                opt.zero_grad(set_to_none=True)
                xp = xp_base.detach().requires_grad_(True)
                varlen_fwd(blocks, xp, ang_t, ang_w, cu, ms).float().pow(2).mean().backward()
            dt, pk = timed(vl, max(2, args.iters // K))
            print(f"{K:>2} {'varlen':>7} {tot:>8} {dt*1000:>8.0f} {tot/dt:>9.0f} {pk:>7.1f}")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"{K:>2} {'varlen':>7} {tot:>8} {'OOM':>8}")


if __name__ == "__main__":
    main()
