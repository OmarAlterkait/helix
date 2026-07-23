"""VGGT-style alternating attention vs full-attention base — time & memory profile.

VGGT pattern (multi-view 3D): alternate WITHIN-PLANE dense layers (each token attends its own
plane's tokens; axial t+wire RoPE) with CROSS-PLANE epipolar layers (attend tokens within a
drift-time slab across planes; time-only RoPE). Implemented block-sparse via FlexAttention over
tokens sorted by (plane, t_phys). Base = current FMModel full self-attention (dense SDPA).
Profiles encoder fwd+bwd time and peak memory at full visibility (the regime where sparse wins:
plane-masking / probes / inference), across token counts.
"""
import sys, os, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from model import Block, apply_rope, rope_angles
import data as D
D.init_pipeline_cpu()
dev = "cuda"
_flex = torch.compile(flex_attention)
_cbm = create_block_mask                # eager: mask_mod tensor-indexing doesn't trace under compile


class VGGTBlock(nn.Module):
    """Same params as base Block; attention restricted to a FlexAttention block_mask, RoPE per-tier."""
    def __init__(self, d, heads, ffn_mult=4):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x, ang_t, ang_w, block_mask):
        T, d = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        q = apply_rope(q.view(T, self.h, self.hd), ang_t, ang_w)   # ang_w=None on cross-plane layers
        k = apply_rope(k.view(T, self.h, self.hd), ang_t, ang_w)
        v = v.view(T, self.h, self.hd)
        q, k = q.to(v.dtype), k.to(v.dtype)                        # apply_rope upcasts; flex needs matched dtype
        o = _flex(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                  block_mask=block_mask)[0].transpose(0, 1)
        x = x + self.proj(o.reshape(T, d))
        return x + self.mlp(self.n2(x))


def build_masks(plane, t_phys, dt, T):
    pl = plane.to(dev); tt = t_phys.to(dev)
    def plane_mod(b, h, q, k): return pl[q] == pl[k]                # within-plane block-diagonal
    def cross_mod(b, h, q, k): return (tt[q] - tt[k]).abs() <= dt   # cross-plane drift-time slab
    bm_p = _cbm(plane_mod, 1, 1, T, T, device=dev)
    bm_c = _cbm(cross_mod, 1, 1, T, T, device=dev)
    return bm_p, bm_c


def timed(fn, iters=10, warmup=3):
    for _ in range(warmup): fn(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--dt", type=float, default=128.0)
    ap.add_argument("--caps", default="8000,16000,24000,32000")   # token counts to test
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "../artifacts/fm_cache_tpc/ev_*.npz")))
    B = D.get_cached(files[3], device=dev)
    plane_all = B["plane_id"].to(dev); t_all = B["t_phys"].to(dev); w_all = B["wire_pos"].to(dev)
    Nfull = len(plane_all)
    print(f"event N={Nfull}  d={a.d} heads={a.heads} blocks={a.blocks} dt={a.dt}", flush=True)

    base = nn.ModuleList([Block(a.d, a.heads) for _ in range(a.blocks)]).to(dev)
    vggt = nn.ModuleList([VGGTBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)
    lam_t, lam_w = (8.0, 4336.0), (32.0, 2048.0)

    print(f"\n{'N':>7} | {'BASE full-attn':>22} | {'VGGT alternating':>22} | speedup  mem×", flush=True)
    print(f"{'':>7} | {'fwd+bwd ms':>11} {'peak GB':>10} | {'fwd+bwd ms':>11} {'peak GB':>10} |", flush=True)
    for cap in [int(c) for c in a.caps.split(",")]:
        n = min(cap, Nfull)
        # take a contiguous prefix, then sort by (plane, t) for VGGT block structure
        pl, tp, wp = plane_all[:n], t_all[:n], w_all[:n]
        order = torch.argsort(pl * 1e7 + tp)                        # sort by (plane, t_phys)
        pls, tps, wps = pl[order], tp[order], wp[order]
        at = rope_angles(tp, a.d // a.heads, *lam_t); aw = rope_angles(wp, a.d // a.heads, *lam_w)
        ats = rope_angles(tps, a.d // a.heads, *lam_t); aws = rope_angles(wps, a.d // a.heads, *lam_w)

        # BASE: dense full attention, natural order
        def run_base():
            x = torch.randn(n, a.d, device=dev, requires_grad=True)
            h = x
            for blk in base: h = blk(h, at, aw)
            h.sum().backward()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bt, bm = timed(run_base)
            base_ok = True
        except torch.cuda.OutOfMemoryError:
            bt, bm, base_ok = float('nan'), float('nan'), False; torch.cuda.empty_cache()

        # VGGT: block-sparse alternating, sorted order. Pad to 128-multiple (FlexAttention block grid);
        # padding gets sentinel plane 99 / t 1e9 so it attends only itself and never real tokens.
        npad = ((n + 127) // 128) * 128
        def pad(v, fill):
            o = torch.full((npad,) + v.shape[1:], fill, dtype=v.dtype, device=dev); o[:n] = v; return o
        pls_p, tps_p = pad(pls, 99), pad(tps, 1e9)
        atsp = torch.zeros(npad, ats.shape[1], device=dev); atsp[:n] = ats
        awsp = torch.zeros(npad, aws.shape[1], device=dev); awsp[:n] = aws
        bmp, bmc = build_masks(pls_p, tps_p, a.dt, npad)
        def run_vggt():
            x = torch.randn(npad, a.d, device=dev, requires_grad=True)
            h = x
            for i, blk in enumerate(vggt):
                if i % 2 == 0: h = blk(h, atsp, awsp, bmp)           # within-plane: axial RoPE
                else:          h = blk(h, atsp, None, bmc)           # cross-plane: time-only RoPE
            h.sum().backward()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vt, vm = timed(run_vggt)

        sp = bt / vt if base_ok else float('nan'); mr = bm / vm if base_ok else float('nan')
        print(f"{n:>7} | {bt:>11.1f} {bm:>10.2f} | {vt:>11.1f} {vm:>10.2f} | {sp:>6.2f}x {mr:>5.2f}x", flush=True)


if __name__ == "__main__":
    main()
