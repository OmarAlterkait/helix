"""Scaling head-to-head: FULL attention (O(N^2)) vs our SERIAL grouped scheme (O(N*g))
vs GENERIC RANDOM-SUBSET attention (BigBird-style random attention, O(N*k)) as N grows
16k -> 256k, on SYNTHETIC tokens at our measured density (~8.7/tick), fwd+bwd, bf16 autocast.

Method note (see GSA_VS_GROUPED.md for citations):
  There is no separately-named NEW "Google random sparse attention" from 2025-2026 that
  web search surfaces; the canonical Google method whose defining ingredient is RANDOM
  attention is BigBird (Zaheer et al., NeurIPS 2020, arXiv:2007.14062), where each query
  attends to `r` RANDOMLY chosen keys (Erdos-Renyi random graph -> expander), optionally
  plus a local window and a few global tokens, giving O(N) cost. Here we profile the pure
  RANDOM-SUBSET mechanism (each query attends k random keys, O(N*k)) -- clearly labeled as
  the GENERIC mechanism, NOT a specific unreleased paper. We set k ~= our group size g so
  per-token attention FLOPs match the grouped scheme (apples-to-apples).

Honesty: random gather touches k scattered keys per query -> the K/V gather materializes an
(N, k, hd) tensor with NO locality, exactly the memory-access cost that sorted equal-count
grouping avoids (grouped attention reads contiguous blocks). We report both time and peak GB.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from prof_serial import SBlock, uniform_attn, timed
from model import apply_rope, rope_angles
dev = "cuda"


def synth(N):
    pl = torch.randint(0, 6, (N,), device=dev)
    t = torch.rand(N, device=dev) * (N / 8.7)                      # keep ~8.7 tokens/tick
    w = torch.rand(N, device=dev) * 2000
    return pl, t, w


# ---- BigBird-style RANDOM-SUBSET attention (BLOCK-random gather, checkpointed) ----------
# BigBird's actual construction is BLOCK-random (each query BLOCK attends a few random KEY
# BLOCKS) precisely because token-level random gather is a GPU memory wall: the naive per-query
# (T, kbud, h, hd) gather materializes kbud SCATTERED keys per query and OOMs even at 16k under
# a training backward (we verified this: a 4096-query tile alone is ~12 GB). We realize it as a
# tiled block-random gather + batched flash-SDPA, wrapped in gradient CHECKPOINTING so backward
# recomputes each tile's gather (peak memory bounded by ONE tile, not the whole loop).
# Cost is O(N*kbud) FLOPs; the gathered K/V has NO cross-query reuse -- exactly the scattered-
# read cost sorted equal-count grouping avoids (a group's g queries share one contiguous block).
import torch.utils.checkpoint as _ckpt
BLK = 128       # key-block granularity (BigBird block-random). Each query attends whole K-blocks.
QPT = 8         # query-BLOCKS processed per checkpointed tile (batched over the block dim, so
                # each keeps its OWN m-block set -> no budget inflation; bounds the gather buffer).


def _rand_block_idx(N, nrand):
    """Per query-block: `nrand` random distinct K-blocks (+ its own). Returns a (nqb, m) int32
    tensor of chosen K-block ids. Tiny (nqb x nqb noise), no dense (N,N)."""
    nqb = (N + BLK - 1) // BLK
    ar = torch.arange(nqb, device=dev)
    m = min(nrand + 1, nqb)
    score = torch.rand(nqb, nqb, device=dev); score[ar, ar] = 2.0         # force own block in
    sel = torch.sort(score, dim=1, descending=True).indices[:, :m]        # (nqb, m) distinct
    return sel.to(torch.int64)


def _attn_tile(qblk, kf, vf, tok):
    """qblk: (nb, BLK, h, hd) queries for nb query-blocks. tok: (nb, m*BLK) int64 token ids of
    the K-blocks each query-block attends (its OWN random set -- no inflation). Batched SDPA:
    batch=nb, kv-len=m*BLK. Returns (nb, BLK, h, hd). Checkpoint target (recomputed in bwd)."""
    nb, B, h, hd = qblk.shape; mk = tok.shape[1]
    kk = kf[tok].permute(0, 2, 1, 3)                                      # (nb, h, m*BLK, hd)
    vv = vf[tok].permute(0, 2, 1, 3)
    qq = qblk.permute(0, 2, 1, 3)                                         # (nb, h, BLK, hd)
    o = F.scaled_dot_product_attention(qq, kk, vv)                       # (nb, h, BLK, hd)
    return o.permute(0, 2, 1, 3)                                         # (nb, BLK, h, hd)


def random_attn(q, k, v, sel):
    """Block-random subset attention. q,k,v: (T,h,hd). sel: (nqb, m) K-block ids per Q-block.
    Process QPT query-blocks per checkpointed tile; each query-block keeps its OWN m K-blocks
    (batched over the block dim -> no budget inflation). O(T*m*BLK)."""
    T, h, hd = q.shape
    kf = k.to(q.dtype); vf = v.to(q.dtype)
    nqb, m = sel.shape
    Tpad = nqb * BLK
    ar = torch.arange(BLK, device=q.device)
    # token ids per query-block: (nqb, m*BLK), clamped to valid range (last block may run past T)
    tokall = (sel[:, :, None] * BLK + ar[None, None, :]).reshape(nqb, m * BLK).clamp_max(T - 1)
    qpad = q if Tpad == T else torch.cat([q, q[-1:].expand(Tpad - T, h, hd)], 0)
    qb = qpad.view(nqb, BLK, h, hd)
    out = q.new_empty(Tpad, h, hd).view(nqb, BLK, h, hd)
    for s in range(0, nqb, QPT):
        e = min(s + QPT, nqb)
        o = _ckpt.checkpoint(_attn_tile, qb[s:e], kf, vf, tokall[s:e], use_reentrant=False)
        out[s:e] = o
    return out.view(Tpad, h, hd)[:T]


class RBlock(nn.Module):
    """Same dims as SBlock but attention = random-subset. Shares the SBlock recipe
    (pre-LN, qkv, axial RoPE, proj, MLP) so it is bit-for-bit apples-to-apples except
    for the attention pattern."""
    def __init__(self, d, heads, ffn_mult=4):
        super().__init__(); self.h, self.hd = heads, d // heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x, ang_t, ang_w, sel):
        T, d = x.shape; hh = self.n1(x)
        q, k, val = self.qkv(hh).chunk(3, -1)
        q = apply_rope(q.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        k = apply_rope(k.view(T, self.h, self.hd), ang_t, ang_w).to(x.dtype)
        val = val.view(T, self.h, self.hd)
        o = random_attn(q, k, val, sel)
        x = x + self.proj(o.reshape(T, d)); return x + self.mlp(self.n2(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--gp", type=int, default=1024); ap.add_argument("--gd", type=int, default=2048)
    # random budget k: set to the AVERAGE grouped budget so per-token FLOPs match.
    # serial cycle spends 2/4 layers at gp and 2/4 at gd -> mean group size (gp+gd)/2.
    ap.add_argument("--kbud", type=int, default=0)                 # 0 -> auto (gp+gd)//2
    ap.add_argument("--ns", default="16000,32000,64000,128000,256000")
    a = ap.parse_args()
    kbud = a.kbud or (a.gp + a.gd) // 2
    nrand = max(1, round(kbud / BLK) - 1)                            # +1 own block -> budget ~= kbud
    kbud_eff = (nrand + 1) * BLK
    lam_t, lam_w = (8.0, 4336.0), (32.0, 2048.0)
    net_s = nn.ModuleList([SBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)
    net_r = nn.ModuleList([RBlock(a.d, a.heads) for _ in range(a.blocks)]).to(dev)
    print(f"d={a.d} blocks={a.blocks} heads={a.heads} | grouped gp={a.gp} gd={a.gd} "
          f"(mean g={(a.gp+a.gd)//2}) | random target k={kbud} -> block-random "
          f"BLK={BLK} nrand={nrand} (eff k={kbud_eff}) | QPT={QPT}", flush=True)
    print(f"\n{'N':>7} | {'FULL ms':>9} {'GB':>5} | {'SERIAL ms':>10} {'GB':>5} | "
          f"{'RANDOM ms':>10} {'GB':>5} | {'full/ser':>8} {'full/rnd':>8} {'ser/rnd':>7}", flush=True)
    for N in [int(x) for x in a.ns.split(",")]:
        pl, tp, wp = synth(N)
        at = rope_angles(tp, a.d // a.heads, *lam_t); aw = rope_angles(wp, a.d // a.heads, *lam_w)
        o_pt = torch.argsort(pl.double() * 1e9 + tp.double())
        o_pw = torch.argsort(pl.double() * 1e13 + wp.double() * 1e6 + tp.double())
        o_t = torch.argsort(tp.double()); o_ts = torch.roll(o_t, a.gd // 2)
        cell = [(o_pt, a.gp, at, aw), (o_t, a.gd, at, None), (o_pw, a.gp, at, aw), (o_ts, a.gd, at, None)]
        sched = cell * (a.blocks // 4)
        # one fixed block-random selection, SHARED across layers -- per-layer-distinct sets don't
        # change attention FLOPs. Built once per N outside the timed loop.
        try:
            sel = _rand_block_idx(N, nrand)
        except torch.cuda.OutOfMemoryError:
            sel = None; torch.cuda.empty_cache()

        def run_full():
            x = torch.randn(N, a.d, device=dev, requires_grad=True); h = x
            for blk in net_s: h = blk(h, at, aw, None, 0)
            h.sum().backward()
        def run_serial():
            x = torch.randn(N, a.d, device=dev, requires_grad=True); h = x
            for i, blk in enumerate(net_s): o, g, angt, angw = sched[i]; h = blk(h, angt, angw, o, g)
            h.sum().backward()
        def run_random():
            x = torch.randn(N, a.d, device=dev, requires_grad=True); h = x
            for i, blk in enumerate(net_r): h = blk(h, at, aw, sel)
            h.sum().backward()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            try: ft, fm = timed(run_full, iters=5)
            except torch.cuda.OutOfMemoryError: ft, fm = float('nan'), float('nan'); torch.cuda.empty_cache()
            try: st, sm = timed(run_serial, iters=5)
            except torch.cuda.OutOfMemoryError: st, sm = float('nan'), float('nan'); torch.cuda.empty_cache()
            if sel is None:
                rt, rm = float('nan'), float('nan')
            else:
                try: rt, rm = timed(run_random, iters=5)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
                    rt, rm = float('nan'), float('nan'); torch.cuda.empty_cache()
                    print(f"  [random N={N} failed: {type(ex).__name__}: {str(ex)[:120]}]", flush=True)
        del sel; torch.cuda.empty_cache()

        def rat(a_, b_): return a_ / b_ if (a_ == a_ and b_ == b_) else float('nan')
        print(f"{N:>7} | {ft:>7.1f} {fm:>5.2f} | {st:>8.1f} {sm:>5.2f} | {rt:>8.1f} {rm:>5.2f} | "
              f"{rat(ft,st):>7.2f}x {rat(ft,rt):>7.2f}x {rat(st,rt):>6.2f}x", flush=True)


if __name__ == "__main__":
    main()
