import sys, os, time, glob, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/sdf/group/neutrino/omara/helix/.pylibs'); sys.path.insert(0, '.')
import data as D
from data import DEV, N_SLOT, N_BAND
from model import Block, rope_angles, apply_rope
D.init_pipeline_cpu()
d, heads, ENC, DEC = 512, 4, 10, 4
B = D.get_cached(sorted(glob.glob('../artifacts/fm_cache_tpc/ev_*.npz'))[0])
B = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in B.items()}
N = int(B['n_cells'])
g = torch.Generator(device=DEV).manual_seed(0)
mask = torch.rand(N, generator=g, device=DEV) < 0.75
nv = int((~mask).sum()); nm = int(mask.sum())
print(f"event N={N}, mask0.75 -> visible={nv} ({nv/N:.0%}) masked={nm}")
at = rope_angles(B['t_phys'], d // 4, 8, 4336); aw = rope_angles(B['wire_pos'], d // 4, 32, 2048)
atv, awv = at[~mask], aw[~mask]; atm, awm = at[mask], aw[mask]


def timeit(fn, n=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000


def mem_fwd_bwd(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


enc = nn.ModuleList(Block(d, heads) for _ in range(ENC)).to(DEV)
dec = nn.ModuleList(Block(d, heads) for _ in range(DEC)).to(DEV)


class CrossBlock(nn.Module):
    """Mask-token queries cross-attend to encoded VISIBLE tokens (O(nm*nv) not O(N^2))."""
    def __init__(self, d, heads, ffn=4):
        super().__init__(); self.h, self.hd = heads, d // heads
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, ffn * d), nn.GELU(), nn.Linear(ffn * d, d))

    def forward(self, q, kv, aqt, aqw, akt, akw):
        Tq, Tk = q.shape[0], kv.shape[0]
        qq = apply_rope(self.q(self.nq(q)).view(Tq, self.h, self.hd), aqt, aqw)
        k, v = self.kv(self.nk(kv)).chunk(2, -1)
        k = apply_rope(k.view(Tk, self.h, self.hd), akt, akw); v = v.view(Tk, self.h, self.hd)
        o = F.scaled_dot_product_attention(qq.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                           v.transpose(0, 1)[None])[0].transpose(0, 1)
        q = q + self.proj(o.reshape(Tq, d))
        return q + self.mlp(self.n2(q))


xdec = nn.ModuleList(CrossBlock(d, heads) for _ in range(DEC)).to(DEV)
xv0 = torch.randn(nv, d, device=DEV); xall0 = torch.randn(N, d, device=DEV); xm0 = torch.randn(nm, d, device=DEV)


def encoder():
    x = xv0
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for b in enc: x = b(x, atv, awv)
    return x


def dec_self():
    x = xall0
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for b in dec: x = b(x, at, aw)
    return x


def dec_cross():
    vis = xv0; q = xm0
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for b in xdec: q = b(q, vis, atm, awm, atv, awv)
    return q


te = timeit(encoder); ts = timeit(dec_self); tc = timeit(dec_cross)
print("\n--- FORWARD time ---")
print(f"  ENCODER  (10 blk on {nv} visible): {te:6.1f} ms")
print(f"  DECODER  self-attn (4 blk, full {N}): {ts:6.1f} ms   <- current")
print(f"  DECODER  cross-attn (4 blk, {nm}q x {nv}kv): {tc:6.1f} ms   ({ts/tc:.1f}x faster)")
print(f"  decoder share of (enc+dec): self {ts/(te+ts):.0%}, cross {tc/(te+tc):.0%}")
print(f"  total fwd: self-dec {te+ts:.0f} ms  ->  cross-dec {te+tc:.0f} ms ({(te+ts)/(te+tc):.2f}x)")


def fb_self():
    x = xall0.detach().requires_grad_(True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        y = x
        for b in dec: y = b(y, at, aw)
    y[mask].float().pow(2).mean().backward()


def fb_cross():
    q = xm0.detach().requires_grad_(True); vis = xv0.detach().requires_grad_(True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for b in xdec: q = b(q, vis, atm, awm, atv, awv)
    q.float().pow(2).mean().backward()


ms = mem_fwd_bwd(fb_self); mc = mem_fwd_bwd(fb_cross)
print("\n--- DECODER fwd+bwd peak memory ---")
print(f"  self-attn decoder:  {ms:.2f} GB   <- current")
print(f"  cross-attn decoder: {mc:.2f} GB   ({ms/mc:.1f}x less)")
