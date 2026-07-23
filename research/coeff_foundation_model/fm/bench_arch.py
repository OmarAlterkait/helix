"""Benchmark architecture variants: effective tokens + speed + memory vs DEPTH.
Compares, on a real event:
  full      : self-attention over all N tokens (what deconv does; the expensive baseline)
  encdrop   : self-attention over the 25% visible tokens (current MAE encoder)
  vggt      : within-plane self-attention (6 plane-blocks) + every-Kth global layer
  perceiver : cross-attend N->M latents, deep stack on M, cross-attend back to N
  crossmae  : cross-attention decoder (masked queries x visible kv) -- one layer cost
Reports fwd+bwd ms/step and peak GB at depth in {10,24,48}.
"""
import sys, os, time, glob, math, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/sdf/group/neutrino/omara/helix/.pylibs'); sys.path.insert(0, '.')
import data as D
from data import DEV, N_SLOT, N_BAND
from model import rope_angles, apply_rope
D.init_pipeline_cpu()
d, heads = 512, 8; hd = d // heads

B = D.get_cached(sorted(glob.glob('../artifacts/fm_cache_tpc/ev_*.npz'))[0])
B = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in B.items()}
N = int(B['n_cells']); plane = B['plane_id']
g = torch.Generator(device=DEV).manual_seed(0); mask = torch.rand(N, generator=g, device=DEV) < 0.75
vis = ~mask; nv = int(vis.sum()); nm = int(mask.sum())
at = rope_angles(B['t_phys'], d // heads, 8, 4336); aw = rope_angles(B['wire_pos'], d // heads, 32, 2048)
# plane-sorted blocks for within-plane attention
order = torch.argsort(plane, stable=True); counts = torch.bincount(plane).tolist()
print(f"event N={N}, planes={len(counts)} sizes={counts}, visible={nv} masked={nm}\n")


def mha(q, k, v):
    T = q.shape[0]; Tk = k.shape[0]
    return F.scaled_dot_product_attention(q.view(T, heads, hd).transpose(0, 1)[None],
        k.view(Tk, heads, hd).transpose(0, 1)[None], v.view(Tk, heads, hd).transpose(0, 1)[None])[0].transpose(0, 1).reshape(T, d)


class Self(nn.Module):
    def __init__(s): super().__init__(); s.n = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3*d); s.p = nn.Linear(d, d); s.n2 = nn.LayerNorm(d); s.m = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x, blocks=None):       # blocks=list of (start,len) for within-plane; None=global
        q, k, v = s.qkv(s.n(x)).chunk(3, -1)
        if blocks is None:
            o = mha(q, k, v)
        else:
            o = torch.empty_like(q); st = 0
            for c in blocks:
                sl = slice(st, st+c); o[sl] = mha(q[sl], k[sl], v[sl]); st += c
        x = x + s.p(o); return x + s.m(s.n2(x))


class Cross(nn.Module):
    def __init__(s): super().__init__(); s.nq = nn.LayerNorm(d); s.nk = nn.LayerNorm(d); s.q = nn.Linear(d, d); s.kv = nn.Linear(d, 2*d); s.p = nn.Linear(d, d); s.n2 = nn.LayerNorm(d); s.m = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, q, kv):
        k, v = s.kv(s.nk(kv)).chunk(2, -1)
        q = q + s.p(mha(s.q(s.nq(q)), k, v)); return q + s.m(s.n2(q))


def timeit(fn, n=8, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t)/n*1000


def memrun(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()/1e9


xN = torch.randn(N, d, device=DEV); xV = torch.randn(nv, d, device=DEV); xM_ = torch.randn(nm, d, device=DEV)
results = []
for L in (10, 24, 48):
    # full self-attn on all N (deconv-style)
    full = nn.ModuleList(Self() for _ in range(L)).to(DEV)
    def f_full():
        x = xN.detach().requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for b in full: x = b(x)
        x.float().pow(2).mean().backward()
    # encoder-drop self-attn on visible
    enc = nn.ModuleList(Self() for _ in range(L)).to(DEV)
    def f_enc():
        x = xV.detach().requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for b in enc: x = b(x)
        x.float().pow(2).mean().backward()
    # vggt: within-plane, every 4th layer global
    vg = nn.ModuleList(Self() for _ in range(L)).to(DEV)
    xNs = xN[order]
    def f_vggt():
        x = xNs.detach().requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for i, b in enumerate(vg): x = b(x, blocks=None if i % 4 == 3 else counts)
        x.float().pow(2).mean().backward()
    # perceiver: cross-in N->M, L self-attn on M, cross-out M->N
    for M in (1024, 2048):
        cin = Cross().to(DEV); deep = nn.ModuleList(Self() for _ in range(L)).to(DEV); cout = Cross().to(DEV)
        lat0 = torch.randn(M, d, device=DEV)
        def f_perc(cin=cin, deep=deep, cout=cout, lat0=lat0):
            x = xN.detach().requires_grad_(True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lat = cin(lat0, x)
                for b in deep: lat = b(lat)
                out = cout(x, lat)
            out.float().pow(2).mean().backward()
        results.append((L, f"perceiver M={M}", M, timeit(f_perc), memrun(f_perc)))
    results.append((L, "full (deconv)", N, timeit(f_full), memrun(f_full)))
    results.append((L, "encdrop (MAE enc)", nv, timeit(f_enc), memrun(f_enc)))
    results.append((L, "vggt local+global", N, timeit(f_vggt), memrun(f_vggt)))

# crossmae decoder: single-layer-equivalent cost for 4 decoder layers
dec = nn.ModuleList(Cross() for _ in range(4)).to(DEV)
def f_cm():
    q = xM_.detach().requires_grad_(True); kv = xV.detach().requires_grad_(True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for b in dec: q = b(q, kv)
    q.float().pow(2).mean().backward()
print(f"{'depth':>6} {'variant':>20} {'tokens':>8} {'ms(fwd+bwd)':>12} {'peakGB':>8}")
for L, name, tok, ms, mem in sorted(results):
    print(f"{L:>6} {name:>20} {tok:>8} {ms:>12.0f} {mem:>8.1f}")
print(f"\n  crossmae 4-layer decoder ({nm}q x {nv}kv): {timeit(f_cm):.0f} ms, {memrun(f_cm):.1f} GB")
