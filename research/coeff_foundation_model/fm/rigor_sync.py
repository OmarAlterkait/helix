"""Pin down the nonzero/sync count and RoPE recompute, + test the two cheap fixes:
  fix1: precompute RoPE cos/sin ONCE (hoist out of per-block apply_rope)
  fix2: avoid .nonzero()/.item() syncs in forward_feat + losses (use boolean
        masking that doesn't force a host sync where possible)
We measure fwd time with fix1 to confirm RoPE hoist recovers the ~8ms.
"""
import os, sys, glob
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
import data as D
from data import DEV
import model as M
from model import FMModel, losses
from torch.profiler import profile, ProfilerActivity


def load1():
    D.init_pipeline_cpu()
    return D.get_cached(sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[0], device=DEV)


def mk(B):
    g = torch.Generator(device=DEV).manual_seed(0)
    return torch.rand(B["inp"].shape[0], device=DEV, generator=g) < 0.75


def ct(fn, it=30, w=8):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/it


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    B = load1(); m = mk(B)
    model = FMModel(128, D.N_BAND, 6, d=512, blocks=12, dec_blocks=4, heads=8, dec_mode="cross").to(DEV)
    print(f"N={B['inp'].shape[0]}\n")

    def fwd_only():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(B, m)

    # count nonzero & syncs in ONE fwd (no_grad) and ONE fwd+bwd
    opt = torch.optim.AdamW(model.parameters(), 4e-4)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, m); bce, val = losses(occ, mu, lv, B, m, noisy=True)
            (bce+val).backward()
    for _ in range(3): step()
    torch.cuda.synchronize()
    for label, fn in [("fwd-only(no_grad)", fwd_only), ("fwd+loss+bwd", step)]:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            fn(); torch.cuda.synchronize()
        nz = sum(e.count for e in p.key_averages() if e.key == "aten::nonzero")
        sy = sum(e.count for e in p.key_averages() if "ynchronize" in e.key.lower())
        it = sum(e.count for e in p.key_averages() if e.key in ("aten::item","aten::is_nonzero"))
        print(f"  {label:20}: nonzero={nz}  streamSync={sy}  item/is_nonzero={it}")
    print()

    # ---- fix1: hoist RoPE cos/sin out of per-block ----
    # current apply_rope recomputes cos(ang),sin(ang),repeat_interleave every block.
    # Precompute per-(token,axis) cos/sin once; rewrite apply_rope to take them.
    base_fwd = ct(fwd_only)
    print(f"  baseline fwd            = {base_fwd:.1f} ms")

    orig = M.apply_rope
    _cache = {}
    def apply_rope_cached(x, ang_t, ang_w):
        h2 = x.shape[-1] // 2
        key = (id(ang_t), id(ang_w))
        if key not in _cache:
            def cs(ang):
                c = torch.cos(ang)[:, None, :].repeat_interleave(2, -1)
                s = torch.sin(ang)[:, None, :].repeat_interleave(2, -1)
                return c, s
            ct_, st_ = cs(ang_t)
            cw_, sw_ = cs(ang_w) if ang_w is not None else (None, None)
            _cache[key] = (ct_, st_, cw_, sw_)
        ct_, st_, cw_, sw_ = _cache[key]
        def rot(v, c, s):
            v2 = torch.stack([-v[..., 1::2], v[..., 0::2]], -1).reshape_as(v)
            return v * c + v2 * s
        xt = rot(x[..., :h2], ct_, st_)
        xw = rot(x[..., h2:], cw_, sw_) if cw_ is not None else x[..., h2:]
        return torch.cat([xt, xw], -1)

    M.apply_rope = apply_rope_cached
    def fwd_cached():
        _cache.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(B, m)
    fix1 = ct(fwd_cached)
    M.apply_rope = orig
    print(f"  fwd, RoPE cos/sin hoisted (per-fwd cache) = {fix1:.1f} ms  "
          f"(saves ~{base_fwd-fix1:.1f} ms, {100*(base_fwd-fix1)/base_fwd:.0f}%)")
    print("\nDONE")


if __name__ == "__main__":
    main()
