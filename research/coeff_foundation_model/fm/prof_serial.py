"""THOROUGH profile of the PTv3-style 3-order cyclic serialized equal-count-group encoder vs full attention.
Orders: O_pt=(plane,t) axial RoPE | O_pw=(plane,wire,t) axial RoPE | O_t=(t) time-only RoPE.
12 layers = 3 x [O_pt, O_t, O_pw, O_t(shift g/2)]; plane layers use g_plane, drift layers g_drift.
Equal-count groups via sort -> view(-1,g) -> batched flash-SDPA -> unsort. Last group DUPLICATE-padded
(repeat last token) so all keys are valid -> no attn_mask -> flash stays fast (the correct, fast fix for
the pad-key leak). Reports: group-span sanity, per-order-type layer cost, N-sweep, g-sweep, memory.
"""
import sys, os, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import apply_rope, rope_angles
import data as D; D.init_pipeline_cpu()
dev = "cuda"


def uniform_attn(q, k, v, order, g):
    """Equal-count group attention in `order`, group size g, duplicate-padded last group."""
    T, h, hd = q.shape; npad = ((T + g - 1) // g) * g; nb = npad // g
    def grp(x):
        b = x.new_empty(npad, h, hd); b[:T] = x[order]
        if npad > T: b[T:] = x[order[-1]]                          # duplicate-pad (valid keys, no mask)
        return b.view(nb, g, h, hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(grp(q), grp(k.to(q.dtype)), grp(v.to(q.dtype)))
    o = o.permute(0, 2, 1, 3).reshape(npad, h, hd)[:T]
    out = o.new_empty(T, h, hd); out[order] = o; return out


class SBlock(nn.Module):
    def __init__(self, d, heads, ffn_mult=4):
        super().__init__(); self.h, self.hd = heads, d // heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x, ang_t, ang_w, order, g):
        T, d = x.shape; hh = self.n1(x)
        q, k, val = self.qkv(hh).chunk(3, -1)
        q = apply_rope(q.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        k = apply_rope(k.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        val = val.view(T, self.h, self.hd)
        o = (F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], val.transpose(0, 1)[None])[0].transpose(0, 1)
             if order is None else uniform_attn(q, k, val, order, g))
        x = x + self.proj(o.reshape(T, d)); return x + self.mlp(self.n2(x))


def timed(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / iters * 1e3, torch.cuda.max_memory_allocated() / 1e9


def orders_for(pl, tp, wp, n):
    o_pt = torch.argsort(pl.double() * 1e7 + tp.double())
    o_pw = torch.argsort(pl.double() * 1e12 + wp.double() * 1e5 + tp.double())
    o_t = torch.argsort(tp.double())
    return o_pt, o_pw, o_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=8); ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--gp", type=int, default=1024); ap.add_argument("--gd", type=int, default=2048)
    a = ap.parse_args()
    B = D.get_cached(sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/fm_cache_tpc/ev_*.npz")))[3], device=dev)
    pa, ta, wa = B["plane_id"].to(dev), B["t_phys"].to(dev), B["wire_pos"].to(dev)
    Nf = len(pa); lam_t, lam_w = (8.0, 4336.0), (32.0, 2048.0)
    print(f"event N={Nf} d={a.d} blocks={a.blocks} g_plane={a.gp} g_drift={a.gd}", flush=True)
    dens = Nf / float((ta.max() - ta.min()).item()); print(f"token density = {dens:.1f}/tick", flush=True)
    net = nn.ModuleList([SBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)

    def build(n, gp, gd):
        pl, tp, wp = pa[:n], ta[:n], wa[:n]
        at = rope_angles(tp, a.d // a.heads, *lam_t); aw = rope_angles(wp, a.d // a.heads, *lam_w)
        o_pt, o_pw, o_t = orders_for(pl, tp, wp, n)
        o_ts = torch.roll(o_t, gd // 2)                                            # shifted drift order
        # 12-layer cycle: [O_pt, O_t, O_pw, O_t-shift]; RoPE axial on plane orders, time-only on O_t
        sched = []
        for _ in range(a.blocks // 4):
            sched += [(o_pt, gp, at, aw), (o_t, gd, at, None), (o_pw, gp, at, aw), (o_ts, gd, at, None)]
        return at, aw, sched

    def run(n, sched, mode):
        x = torch.randn(n, a.d, device=dev, requires_grad=True); h = x
        for i, blk in enumerate(net):
            if mode == "base": h = blk(h, sched[0][2], sched[0][3], None, 0)
            else:              o, g, angt, angw = sched[i]; h = blk(h, angt, angw, o, g)
        h.sum().backward()

    # (1) N-sweep at default g
    print("\n== N-sweep (g_plane={}, g_drift={}): full-attn vs 3-order serial ==".format(a.gp, a.gd), flush=True)
    print(f"  {'N':>6} | {'BASE ms':>9} {'GB':>5} | {'SERIAL ms':>10} {'GB':>5} | speedup mem×", flush=True)
    for n in [8000, 16000, 24000, min(32000, Nf), Nf]:
        n = min(n, Nf); at, aw, sched = build(n, a.gp, a.gd)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bt, bm = timed(lambda: run(n, sched, "base")); st, sm = timed(lambda: run(n, sched, "serial"))
        print(f"  {n:>6} | {bt:>7.1f} {bm:>5.2f} | {st:>8.1f} {sm:>5.2f} | {bt/st:>5.2f}x {bm/sm:>4.2f}x", flush=True)

    # (2) g-sweep at full N (uniform g for simplicity of the sweep)
    print("\n== g-sweep at N={} (uniform g both tiers) ==".format(Nf), flush=True)
    at, aw, _ = build(Nf, a.gp, a.gd)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        bt, _ = timed(lambda: run(Nf, build(Nf, a.gp, a.gd)[2], "base"))
        print(f"  {'BASE dense':14s} {bt:>7.1f}ms  1.00x", flush=True)
        for g in [512, 1024, 2048, 4096, 8192]:
            _, _, sched = build(Nf, g, g)
            gt, _ = timed(lambda: run(Nf, sched, "serial"))
            print(f"  {'serial g=' + str(g):14s} {gt:>7.1f}ms  {bt/gt:>4.2f}x", flush=True)

    # (3) per-order-type single-layer cost at full N
    print("\n== per-order single-layer cost at N={} ==".format(Nf), flush=True)
    at, aw, sched = build(Nf, a.gp, a.gd)
    o_pt, o_pw, o_t = orders_for(pa, ta, wa, Nf)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for nm, o, g, angw in [("full-attn", None, 0, aw), ("O_pt g%d" % a.gp, o_pt, a.gp, aw),
                               ("O_pw g%d" % a.gp, o_pw, a.gp, aw), ("O_t g%d" % a.gd, o_t, a.gd, None)]:
            def one():
                x = torch.randn(Nf, a.d, device=dev, requires_grad=True)
                net[0](x, at, angw, o, g).sum().backward()
            t, _ = timed(one, iters=12)
            print(f"  {nm:14s} {t:>7.1f}ms/layer", flush=True)


if __name__ == "__main__":
    main()
