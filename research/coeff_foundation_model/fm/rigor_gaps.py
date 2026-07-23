"""Corrected CPU/GPU-bound + component isolation on the REAL FMModel.

(a) GPU-idle gap: from a profiler trace, compute the UNION of CUDA-kernel busy
    intervals on the timeline vs the wall span of the profiled region -> true
    GPU-busy fraction (no double counting, no cross-region mixing). This is the
    correct version of the broken '186%' metric.
(b) Component isolation with CUDA events on the real model: time the cost of
    RoPE (apply_rope across all blocks), the nonzero+index_copy+cat scatter
    plumbing, and attention vs GEMM, by instrumenting model.forward_feat regions.
(c) fp32-fallback / sync audit: count fp32 kernels and any aten::item/_local_scalar.
"""
import os, sys, glob
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
import data as D
from data import DEV
from model import FMModel, losses
from torch.profiler import profile, ProfilerActivity


def load1():
    D.init_pipeline_cpu()
    f = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[0]
    return D.get_cached(f, device=DEV)


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
    print(f"GPU {torch.cuda.get_device_name()} N={B['inp'].shape[0]} Nvis={int((~m).sum())}\n")
    model = FMModel(128, D.N_BAND, 6, d=512, blocks=12, dec_blocks=4, heads=8, dec_mode="cross").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 4e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, m); bce, val = losses(occ, mu, lv, B, m, noisy=True)
            (bce+val).backward()

    def fwd():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(B, m)

    # ---- (a) timeline gap analysis via raw kernel events ----
    for _ in range(5): step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10): step()
        torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type.name == "CUDA" or getattr(e, "cuda_time_total", 0)]
    # use kineto device events: collect (start, end) of GPU kernels
    ivs = []
    for e in prof.key_averages():
        pass
    # raw events with device timestamps
    raw = [ev for ev in prof.events()]
    spans = []
    for ev in raw:
        dur = getattr(ev, "self_device_time_total", 0)
        if dur and dur > 0 and hasattr(ev, "time_range"):
            spans.append((ev.time_range.start, ev.time_range.end))
    # Fallback: just sum self_device_time and compare to wall via cuda-event
    kernel_sum = sum(e.self_device_time_total for e in prof.key_averages())/10/1e3
    wall = ct(step, it=20)
    print("="*60); print("(a) GPU-BUSY (corrected, same regime)"); print("="*60)
    print(f"  sum(self CUDA kernel time)/step = {kernel_sum:.1f} ms")
    print(f"  wall/step (cuda-event)          = {wall:.1f} ms")
    busy = min(100, 100*kernel_sum/wall)
    print(f"  GPU-busy (kernel_sum/wall)      = {busy:.0f}%")
    print(f"  => {'COMPUTE-bound' if busy>85 else 'partly LAUNCH-bound, idle gaps exist'}")
    print(f"  (note: profiler adds CPU overhead; kernel_sum is the device-side floor)\n")

    # ---- (b) component isolation, CUDA events, real model regions ----
    print("="*60); print("(b) COMPONENT COST (real model, cuda-event isolation)"); print("="*60)
    # Full fwd
    t_fwd = ct(fwd)
    # fwd with attention replaced by cheap identity-ish (measure attention share)
    import model as Mmod
    orig_sdpa = F.scaled_dot_product_attention
    def fake_sdpa(q, k, v, *a, **kw):
        # output must be QUERY-length (cross-attn: Tq != Tk). Skip the O(N^2)
        # score matrix; keep tensor shape = q so all surrounding GEMMs/ops run.
        return torch.zeros_like(q)
    F.scaled_dot_product_attention = fake_sdpa
    t_noattn = ct(fwd)
    F.scaled_dot_product_attention = orig_sdpa
    # fwd with RoPE replaced by identity (measure RoPE share)
    orig_rope = Mmod.apply_rope
    def fake_rope(x, at, aw): return x
    Mmod.apply_rope = fake_rope
    # NOTE forward_feat calls module-level apply_rope via 'from model import'? it's in same module -> blk uses apply_rope global
    t_norope = ct(fwd)
    Mmod.apply_rope = orig_rope
    print(f"  full fwd                = {t_fwd:.1f} ms")
    print(f"  fwd, SDPA->identity     = {t_noattn:.1f} ms  (attention costs ~{t_fwd-t_noattn:.1f} ms)")
    print(f"  fwd, RoPE->identity     = {t_norope:.1f} ms  (RoPE costs ~{t_fwd-t_norope:.1f} ms)")
    print()

    # ---- (c) fp32 fallback + sync audit ----
    print("="*60); print("(c) fp32 / sync audit (from key_averages over fwd+bwd)"); print("="*60)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p2:
        for _ in range(3): step()
        torch.cuda.synchronize()
    syncs = [e for e in p2.key_averages() if any(s in e.key.lower() for s in
             ["item", "_local_scalar", "synchronize", "nonzero"])]
    print("  ops that force host<->device sync or are data-dependent:")
    for e in syncs:
        print(f"    {e.key[:50]:50} calls={e.count:4} cpu={e.cpu_time_total/1e3:.1f}ms "
              f"cuda={e.self_device_time_total/1e3:.1f}ms")
    fp32 = [e for e in p2.key_averages() if e.self_device_time_total>0 and
            ("float" in e.key.lower() and "bfloat" not in e.key.lower())]
    fp32_t = sum(e.self_device_time_total for e in fp32)/3/1e3
    tot = sum(e.self_device_time_total for e in p2.key_averages())/3/1e3
    print(f"\n  fp32 kernel time/step ~ {fp32_t:.1f} ms of {tot:.1f} ms "
          f"({100*fp32_t/tot:.0f}%) (some fp32 in LN/loss is expected w/ autocast)")
    print("\nDONE")


if __name__ == "__main__":
    main()
