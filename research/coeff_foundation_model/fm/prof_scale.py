"""Scaling: full attention (O(N^2)) vs 3-order serial schedule (O(N*g), ~linear) as N grows —
the token-growth regime (finer patches / D1 band / multi-event / larger detectors). Synthetic tokens
at the measured density (~8.7/tick) so group structure + cost are realistic; base OOM/slow is caught.
Also profiles a VGGT-style DENSE-global variant (serial within-plane + a few dense global layers) to
show what re-introducing a dense tier costs at scale.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from prof_serial import SBlock, uniform_attn, timed
from model import rope_angles
dev = "cuda"


def synth(N):
    pl = torch.randint(0, 6, (N,), device=dev)
    t = torch.rand(N, device=dev) * (N / 8.7)                      # keep ~8.7 tokens/tick
    w = torch.rand(N, device=dev) * 2000
    return pl, t, w


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--gp", type=int, default=1024); ap.add_argument("--gd", type=int, default=2048)
    ap.add_argument("--ns", default="16000,32000,64000,128000,256000")
    a = ap.parse_args()
    lam_t, lam_w = (8.0, 4336.0), (32.0, 2048.0)
    net = nn.ModuleList([SBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)
    print(f"d={a.d} blocks={a.blocks} gp={a.gp} gd={a.gd}", flush=True)
    print(f"\n{'N':>7} | {'FULL ms':>10} {'GB':>5} | {'SERIAL ms':>10} {'GB':>5} | {'VGGT-dg ms':>11} | full×  vggt×", flush=True)
    for N in [int(x) for x in a.ns.split(",")]:
        pl, tp, wp = synth(N)
        at = rope_angles(tp, a.d // a.heads, *lam_t); aw = rope_angles(wp, a.d // a.heads, *lam_w)
        o_pt = torch.argsort(pl.double() * 1e9 + tp.double())
        o_pw = torch.argsort(pl.double() * 1e13 + wp.double() * 1e6 + tp.double())
        o_t = torch.argsort(tp.double()); o_ts = torch.roll(o_t, a.gd // 2)
        cell = [(o_pt, a.gp, at, aw), (o_t, a.gd, at, None), (o_pw, a.gp, at, aw), (o_ts, a.gd, at, None)]
        sched = cell * (a.blocks // 4)

        def run_full():
            x = torch.randn(N, a.d, device=dev, requires_grad=True); h = x
            for blk in net: h = blk(h, at, aw, None, 0)
            h.sum().backward()
        def run_serial():
            x = torch.randn(N, a.d, device=dev, requires_grad=True); h = x
            for i, blk in enumerate(net): o, g, angt, angw = sched[i]; h = blk(h, angt, angw, o, g)
            h.sum().backward()
        def run_vggt_dg():                                          # within-plane serial + DENSE global on the 6 "O_t" slots
            x = torch.randn(N, a.d, device=dev, requires_grad=True); h = x
            for i, blk in enumerate(net):
                o, g, angt, angw = sched[i]
                if i % 2 == 1: h = blk(h, at, None, None, 0)        # dense global (VGGT-vanilla style)
                else:          h = blk(h, angt, angw, o, g)
            h.sum().backward()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            try: ft, fm = timed(run_full, iters=5)
            except torch.cuda.OutOfMemoryError: ft, fm = float('nan'), float('nan'); torch.cuda.empty_cache()
            st, sm = timed(run_serial, iters=5)
            try: vt, _ = timed(run_vggt_dg, iters=5)
            except torch.cuda.OutOfMemoryError: vt = float('nan'); torch.cuda.empty_cache()
        fs = ft / st if ft == ft else float('nan'); vs = vt / st if vt == vt else float('nan')
        print(f"{N:>7} | {ft:>8.1f} {fm:>5.2f} | {st:>8.1f} {sm:>5.2f} | {vt:>9.1f} | {fs:>5.2f}x {ft/vt if vt==vt and ft==ft else float('nan'):>5.2f}x", flush=True)


if __name__ == "__main__":
    main()
