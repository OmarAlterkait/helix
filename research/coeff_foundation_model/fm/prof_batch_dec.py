"""Memory (fwd vs fwd+bwd), batching under the serial scheme, and the DECODER — the real questions.
- Part 1: serial encoder peak memory, forward-only vs forward+backward, at full N.
- Part 2: batching = pack B events (event id in the top sort bits -> event-pure groups). ms/event + peak mem.
- Part 3: decoder (CrossMAE: masked queries cross-attend visible keys). full O(N_mask*N_vis) vs grouped O(N_mask*g),
  under plane-masking (mask 1/6 planes -> N_mask~N/6 queries, N_vis~5N/6 keys). The decoder runs ALL N and was
  ~65% of GPU time, so this is where grouping matters most.
"""
import sys, os, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from prof_serial import SBlock, uniform_attn
from model import apply_rope, rope_angles
import data as D; D.init_pipeline_cpu()
dev = "cuda"


def timed(fn, iters=6, warmup=3, backward=True):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / iters * 1e3, torch.cuda.max_memory_allocated() / 1e9


def grouped_cross(q, kv, order_q, order_kv, g):
    """masked-query q cross-attends visible kv, both drift-sorted, per equal-count group. q,kv already (T,h,hd)."""
    Tq, h, hd = q.shape; Tk = kv.shape[0]
    # bucket queries and keys into the same number of drift groups by rank fraction
    nb = (max(Tq, Tk) + g - 1) // g
    def grp(x, order, T):
        gg = (T + nb - 1) // nb
        npad = nb * gg; b = x.new_empty(npad, h, hd); b[:T] = x[order]
        if npad > T: b[T:] = x[order[-1]]
        return b.view(nb, gg, h, hd).permute(0, 2, 1, 3), gg
    qb, gq = grp(q, order_q, Tq); kb, gk = grp(kv.to(q.dtype), order_kv, Tk)
    o = F.scaled_dot_product_attention(qb, kb, kb)                 # (nb,h,gq,hd)
    o = o.permute(0, 2, 1, 3).reshape(nb * gq, h, hd)[:Tq]
    out = o.new_empty(Tq, h, hd); out[order_q] = o; return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--gp", type=int, default=1024); ap.add_argument("--gd", type=int, default=2048)
    a = ap.parse_args()
    B = D.get_cached(sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/fm_cache_tpc/ev_*.npz")))[3], device=dev)
    pa, ta, wa = B["plane_id"].to(dev), B["t_phys"].to(dev), B["wire_pos"].to(dev)
    Nf = len(pa); lam_t, lam_w = (8.0, 4336.0), (32.0, 2048.0); H, hd = a.heads, a.d // a.heads
    net = nn.ModuleList([SBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)

    def sched_for(pl, tp, wp, ev, n, gp, gd):
        at = rope_angles(tp, hd, *lam_t); aw = rope_angles(wp, hd, *lam_w)
        base = ev.double() * 1e15 if ev is not None else 0.0        # event id in TOP bits -> event-pure groups
        o_pt = torch.argsort(base + pl.double() * 1e9 + tp.double())
        o_pw = torch.argsort(base + pl.double() * 1e13 + wp.double() * 1e6 + tp.double())
        o_t = torch.argsort((ev.double() * 1e9 if ev is not None else 0.0) + tp.double()); o_ts = torch.roll(o_t, gd // 2)
        cell = [(o_pt, gp, at, aw), (o_t, gd, at, None), (o_pw, gp, at, aw), (o_ts, gd, at, None)]
        return at, aw, cell * (a.blocks // 4)

    def run(n, sched, backward=True):
        x = torch.randn(n, a.d, device=dev, requires_grad=backward); h = x
        for i, blk in enumerate(net): o, g, angt, angw = sched[i]; h = blk(h, angt, angw, o, g)
        if backward: h.sum().backward()

    # Part 1 — fwd-only vs fwd+bwd memory (serial encoder, full N)
    print(f"N={Nf} d={a.d} blocks={a.blocks} gp={a.gp} gd={a.gd}", flush=True)
    _, _, sched = sched_for(pa, ta, wa, None, Nf, a.gp, a.gd)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad(): _, mf = timed(lambda: run(Nf, sched, backward=False))
        _, mb = timed(lambda: run(Nf, sched, backward=True))
    print(f"\n== Part 1: serial encoder memory (full N) ==\n  forward-only  {mf:.2f} GB\n  fwd+backward  {mb:.2f} GB  (activation retention = {mb-mf:.2f} GB)", flush=True)

    # Part 2 — batching: pack B events (event-pure groups)
    print("\n== Part 2: batching (pack B events, event-pure groups) ==", flush=True)
    print(f"  {'B':>2} {'tokens':>7} | {'ms/step':>8} {'ms/event':>9} {'peak GB':>8}", flush=True)
    for Bn in [1, 2, 3, 4]:
        pl = pa.repeat(Bn); tp = ta.repeat(Bn); wp = wa.repeat(Bn)
        ev = torch.arange(Bn, device=dev).repeat_interleave(Nf)
        n = Bn * Nf; _, _, sc = sched_for(pl, tp, wp, ev, n, a.gp, a.gd)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t, m = timed(lambda: run(n, sc, backward=True), iters=5)
            print(f"  {Bn:>2} {n:>7} | {t:>7.1f} {t/Bn:>8.1f} {m:>7.2f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  {Bn:>2} {n:>7} | OOM", flush=True); torch.cuda.empty_cache()

    # Part 3 — decoder: full cross-attn vs grouped, under plane-masking (mask 1 plane)
    print("\n== Part 3: DECODER cross-attention (plane-mask: N_mask~N/6 queries, N_vis~5N/6 keys) ==", flush=True)
    vis = pa != 0; msk = ~vis                                       # hide plane 0 as the masked set
    kv_t = ta[vis]; q_t = ta[msk]; Nv, Nm = int(vis.sum()), int(msk.sum())
    q = torch.randn(Nm, H, hd, device=dev); kv = torch.randn(Nv, H, hd, device=dev)
    oq = torch.argsort(q_t); okv = torch.argsort(kv_t)
    def dec_full():
        x = torch.randn(Nm, H, hd, device=dev, requires_grad=True)
        o = F.scaled_dot_product_attention(x.transpose(0, 1)[None], kv.transpose(0, 1)[None], kv.transpose(0, 1)[None])[0]
        o.sum().backward()
    def dec_grp():
        x = torch.randn(Nm, H, hd, device=dev, requires_grad=True)
        grouped_cross(x, kv, oq, okv, a.gd).sum().backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dt, dm = timed(dec_full, iters=8); gt, gm = timed(dec_grp, iters=8)
    print(f"  N_mask={Nm} N_vis={Nv}", flush=True)
    print(f"  full  cross-attn  {dt:>7.2f} ms  {dm:.2f} GB", flush=True)
    print(f"  grouped cross-attn{gt:>7.2f} ms  {gm:.2f} GB  -> {dt/gt:.2f}x", flush=True)


if __name__ == "__main__":
    main()
