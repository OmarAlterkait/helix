"""P22 — does compiling a BLOCK need the missing setuptools, or only whole-model
compile? Decides whether the container has to be rebuilt before item (3).
"""
import os, sys, traceback
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, timeit, to_device
from helix.model.fm import rope_angles
R = {}
try:
    import setuptools; R["setuptools"] = setuptools.__version__
except Exception:
    R["setuptools"] = "MISSING"
print("setuptools:", R["setuptools"])
dev = "cuda"
model = build(device=dev)
HD = model.d // model.heads
x = torch.randn(8192, model.d, device=dev)
at = rope_angles(torch.rand(8192, device=dev) * 4000, HD, *model.lam_t)
aw = rope_angles(torch.rand(8192, device=dev) * 2000, HD, *model.lam_w)
c = torch.cat([torch.cos(at), torch.cos(aw)], -1)[:, None, :].contiguous()
s = torch.cat([torch.sin(at), torch.sin(aw)], -1)[:, None, :].contiguous()

def rope_fused(t, c, s):
    T, h, hd = t.shape
    v = t.view(T, h, hd // 2, 2)
    a, b = v[..., 0], v[..., 1]
    return torch.stack([a * c - b * s, b * c + a * s], -1).view(T, h, hd)

def _self_perm(blk, xp, c, s, nb, g):
    P, d = xp.shape
    q, k, v = blk.qkv(blk.n1(xp)).chunk(3, -1)
    q = rope_fused(q.view(P, blk.h, blk.hd), c, s)
    k = rope_fused(k.view(P, blk.h, blk.hd), c, s)
    shp = lambda t: t.view(nb, g, blk.h, blk.hd).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(shp(q), shp(k), shp(v.view(P, blk.h, blk.hd)))
    xp = xp + blk.proj(o.permute(0, 2, 1, 3).reshape(P, d))
    return xp + blk.mlp(blk.n2(xp))

for tag, fn, args in (("compile(rope_fused)", rope_fused, (torch.randn(8192, model.heads, HD, device=dev), c, s)),
                      ("compile(_self_perm) fwd", _self_perm, (model.enc[0], x, c, s, 8, 1024))):
    try:
        cf = torch.compile(fn, dynamic=True)
        with torch.autocast("cuda", torch.bfloat16):
            out = cf(*args)
        R[tag] = "OK (forward)"
        print(f"  {tag:28s} OK forward")
        if "self_perm" in tag:
            with torch.autocast("cuda", torch.bfloat16):
                cf(*args).float().pow(2).mean().backward()
            R[tag] = "OK (forward+backward)"
            print(f"  {tag:28s} OK forward+backward")
    except Exception as e:
        R[tag] = f"FAILED: {e!r}"
        print(f"  {tag:28s} FAILED {type(e).__name__}: {e}")
        print(traceback.format_exc()[-900:])
    torch._dynamo.reset()
emit("p22_nosetuptools", R)
