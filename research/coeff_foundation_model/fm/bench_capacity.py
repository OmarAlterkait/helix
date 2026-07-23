"""Bigger model WITHOUT much more runtime. Attention is O(N^2 d) (fixed, the bottleneck);
FFN is O(N d^2) (~6% of time). So capacity added in the FFN (wider, or MoE) is ~free in
runtime. Measure params vs ms/step (fwd+bwd, bf16) for FFN-width and MoE variants.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, glob
import data as D
from data import DEV
from model import rope_angles, apply_rope


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__(); self.h, self.hd = heads, d // heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
    def forward(self, x, at, aw):
        T, d = x.shape; q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        q = apply_rope(q.view(T, self.h, self.hd), at, aw); k = apply_rope(k.view(T, self.h, self.hd), at, aw)
        o = Fn.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                            v.view(T, self.h, self.hd).transpose(0, 1)[None])[0].transpose(0, 1)
        return x + self.proj(o.reshape(T, d))


class DenseFFN(nn.Module):
    def __init__(self, d, mult): super().__init__(); self.n = nn.LayerNorm(d); self.m = nn.Sequential(nn.Linear(d, mult*d), nn.GELU(), nn.Linear(mult*d, d))
    def forward(self, x): return x + self.m(self.n(x))


class MoEFFN(nn.Module):
    def __init__(self, d, n_exp, topk, mult=4):
        super().__init__(); self.n = nn.LayerNorm(d); self.topk = topk; self.n_exp = n_exp
        self.router = nn.Linear(d, n_exp)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d, mult*d), nn.GELU(), nn.Linear(mult*d, d)) for _ in range(n_exp))
    def forward(self, x):
        h = self.n(x); w = self.router(h).softmax(-1); tw, ti = w.topk(self.topk, -1)   # (T,topk)
        out = torch.zeros_like(x)
        for e in range(self.n_exp):
            sel = (ti == e)
            if sel.any():
                tok = sel.any(-1); we = (tw * sel).sum(-1)[tok, None]
                out[tok] = out[tok] + we * self.experts[e](h[tok])
        return x + out


class Stack(nn.Module):
    def __init__(self, d, heads, L, ffn_fn):
        super().__init__(); self.blocks = nn.ModuleList(nn.ModuleList([Attn(d, heads), ffn_fn()]) for _ in range(L))
    def forward(self, x, at, aw):
        for a, f in self.blocks: x = f(a(x, at, aw))
        return x


def main():
    D.init_pipeline_cpu(); d, heads, L = 512, 4, 10
    B = D.get_cached(sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[0])
    N = int(B["n_cells"]); at = rope_angles(B["t_phys"].to(DEV), d//4); aw = rope_angles(B["wire_pos"].to(DEV), d//4)
    x0 = torch.randn(N, d, device=DEV)
    print(f"d={d} L={L} N={N} tokens (full event; decoder-like cost)")
    configs = [("dense ffn4 (base)", lambda: DenseFFN(d, 4)), ("dense ffn8", lambda: DenseFFN(d, 8)),
               ("dense ffn16", lambda: DenseFFN(d, 16)), ("moe 8e top2", lambda: MoEFFN(d, 8, 2)),
               ("moe 16e top2", lambda: MoEFFN(d, 16, 2)), ("moe 32e top2", lambda: MoEFFN(d, 32, 2))]
    print(f"{'config':>18} {'params(M)':>10} {'ms/step':>8} {'peakGB':>7}")
    base_ms = None
    for name, ff in configs:
        s = Stack(d, heads, L, ff).to(DEV); opt = torch.optim.SGD(s.parameters(), lr=0.0)
        npar = sum(p.numel() for p in s.parameters()) / 1e6
        def step():
            opt.zero_grad(set_to_none=True); x = x0.detach().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16): y = s(x, at, aw)
            y.float().pow(2).mean().backward()
        for _ in range(3): step()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        for _ in range(8): step()
        torch.cuda.synchronize(); ms = (time.time()-t)/8*1000; pk = torch.cuda.max_memory_allocated()/1e9
        if base_ms is None: base_ms = ms
        print(f"{name:>18} {npar:>10.1f} {ms:>8.0f} {pk:>7.1f}  ({ms/base_ms:.2f}x)")
        del s, opt; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
