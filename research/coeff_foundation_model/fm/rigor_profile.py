"""RIGOROUS skeptical kernel-level profile of the REAL FMModel.

Goals (all on the ACTUAL model.FMModel + model.losses, no reimplementation,
no analytical FLOPs):
  1. CUDA-event timing of REAL fwd, fwd+loss, fwd+loss+bwd, +opt on a fixed real
     event. Reconcile 35 vs 48 vs 162 ms. Reproduce the prior 35ms "fwd" (which
     used a FAKE loss) to PROVE what it measured.
  2. torch.profiler (CPU+CUDA, record_shapes) of one real fwd+loss+bwd step ->
     top kernels, bucketed (SDPA / GEMM / elementwise / index-scatter / reduce).
  3. CPU-vs-GPU bound: GPU-busy fraction = sum(CUDA kernel time)/wall. Gap => launch-bound.
  4. Which SDPA backend actually fires (FLASH/EFFICIENT/MATH/CUDNN) for the real
     head_dim=64 shapes, both encoder self-attn and cross-decoder.
  5. Any hidden sync / fp32 fallback / pathological op (nonzero, index_copy, RoPE
     repeat_interleave/stack).
"""
import os, sys, glob, time, contextlib
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import data as D
from data import DEV
from model import FMModel, losses
from torch.profiler import profile, ProfilerActivity, record_function
from torch.nn.attention import SDPBackend, sdpa_kernel

DEC = "cross"


def load_events(K):
    D.init_pipeline_cpu()
    fs = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[:K]
    return [D.get_cached(f, device=DEV) for f in fs]


def mk_mask(B, r=0.75, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.rand(B["inp"].shape[0], device=DEV, generator=g) < r


def cuda_time(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    evs = load_events(1)
    B = evs[0]
    m = mk_mask(B)
    N = B["inp"].shape[0]
    print(f"GPU {torch.cuda.get_device_name()} | torch {torch.__version__}")
    print(f"Real event N={N}  N_vis={int((~m).sum())}  N_mask={int(m.sum())}\n")

    model = FMModel(128, D.N_BAND, 6, d=512, blocks=12, dec_blocks=4, heads=8,
                    dec_mode=DEC).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 4e-4)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  dec_mode={DEC}\n")

    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    # ===== 1. CUDA-EVENT TIMING, REAL MODEL =====
    print("="*70)
    print("1) CUDA-EVENT TIMING (real model, fixed event, bf16 autocast)")
    print("="*70)

    def f_fwd():
        with torch.no_grad(), ac():
            model(B, m)

    def f_fwd_fakeloss():  # EXACTLY the prior bench_pack_varlen SEQ "forward"
        with torch.no_grad(), ac():
            occ, mu, lv = model(B, m)
            (occ.float().pow(2).mean() + mu.float().pow(2).mean())

    def f_fwd_realloss():
        with torch.no_grad(), ac():
            occ, mu, lv = model(B, m)
            bce, val = losses(occ, mu, lv, B, m, noisy=True)
            (bce + val)

    def f_fwd_bwd():
        opt.zero_grad(set_to_none=True)
        with ac():
            occ, mu, lv = model(B, m)
            bce, val = losses(occ, mu, lv, B, m, noisy=True)
            loss = bce + val
        loss.backward()

    def f_step():
        opt.zero_grad(set_to_none=True)
        with ac():
            occ, mu, lv = model(B, m)
            bce, val = losses(occ, mu, lv, B, m, noisy=True)
            loss = bce + val
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    t_fwd = cuda_time(f_fwd)
    t_fwd_fake = cuda_time(f_fwd_fakeloss)
    t_fwd_real = cuda_time(f_fwd_realloss)
    t_fb = cuda_time(f_fwd_bwd)
    t_step = cuda_time(f_step)
    print(f"  fwd (no_grad, no loss)        : {t_fwd:6.1f} ms")
    print(f"  fwd + FAKE loss (prior bench) : {t_fwd_fake:6.1f} ms  <- reproduces prior '35ms'?")
    print(f"  fwd + REAL losses()           : {t_fwd_real:6.1f} ms")
    print(f"  fwd + REAL loss + bwd         : {t_fb:6.1f} ms")
    print(f"  full step (+clip+opt)         : {t_step:6.1f} ms")
    print(f"  [bench_profile reported: fwd+loss=48, bwd+opt=114, REAL=162]\n")

    # grad-enabled fwd (no no_grad) — the prior fwd-only bench had grad ON (it backward()s)
    def f_fwd_grad():
        with ac():
            occ, mu, lv = model(B, m)
            occ.float().pow(2).mean() + mu.float().pow(2).mean()
    print(f"  fwd + FAKE loss, GRAD ON      : {cuda_time(f_fwd_grad):6.1f} ms  (builds autograd graph)\n")

    # ===== 2. SDPA BACKEND CHECK =====
    print("="*70)
    print("2) WHICH SDPA BACKEND FIRES (real shapes, head_dim=64)")
    print("="*70)
    Nv = int((~m).sum())
    for name, (q_t, k_t) in [("encoder self-attn (Nv x Nv)", (Nv, Nv)),
                             ("cross-decoder (Nmask x Nvis)", (int(m.sum()), Nv))]:
        q = torch.randn(1, 8, q_t, 64, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(1, 8, k_t, 64, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(1, 8, k_t, 64, device=DEV, dtype=torch.bfloat16)
        fired = []
        for be, lab in [(SDPBackend.FLASH_ATTENTION, "FLASH"),
                        (SDPBackend.EFFICIENT_ATTENTION, "EFFICIENT"),
                        (SDPBackend.CUDNN_ATTENTION, "CUDNN"),
                        (SDPBackend.MATH, "MATH")]:
            try:
                with sdpa_kernel(be):
                    F.scaled_dot_product_attention(q, k, v)
                fired.append(lab)
            except Exception:
                pass
        print(f"  {name}: available backends = {fired}")
    # what does the DEFAULT dispatcher actually pick? -> see profiler kernel names below
    print()

    # ===== 3. torch.profiler KERNEL BREAKDOWN of one real fwd+loss+bwd step =====
    print("="*70)
    print("3) torch.profiler CUDA KERNEL BREAKDOWN (real fwd+loss+bwd step)")
    print("="*70)
    for _ in range(5):
        f_fwd_bwd()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for _ in range(5):
            f_fwd_bwd()
        torch.cuda.synchronize()

    ka = prof.key_averages()
    # total device time over the 5 steps
    tot_cuda_us = sum(e.self_device_time_total for e in ka)
    print(f"  total self CUDA time over 5 steps = {tot_cuda_us/1e3:.1f} ms  "
          f"=> {tot_cuda_us/1e3/5:.1f} ms/step of pure kernel time\n")

    def bucket(name):
        n = name.lower()
        if any(s in n for s in ["flash", "attention", "fmha", "efficient_attention",
                                "scaled_dot", "mha", "sdp"]):
            return "SDPA/attn"
        if any(s in n for s in ["gemm", "cutlass", "ampere", "sgemm", "matmul",
                                "addmm", "wgrad", "dgrad", "cublas", "splitk", "gemv",
                                "s16816", "h16816", "16x8", "implicit"]):
            return "GEMM"
        if any(s in n for s in ["nonzero", "index", "scatter", "gather", "copy",
                                "masked", "take", "put"]):
            return "index/scatter"
        if any(s in n for s in ["reduce", "norm", "sum", "mean", "softmax",
                                "layer_norm", "bce", "mse", "cross_entropy", "var"]):
            return "reduce/norm"
        if any(s in n for s in ["elementwise", "vectorized", "gelu", "add", "mul",
                                "cat", "sin", "cos", "repeat", "stack", "cast",
                                "convert", "fill", "to_copy", "stride", "slice"]):
            return "elementwise"
        return "other"

    from collections import defaultdict
    bk = defaultdict(float)
    for e in ka:
        if e.self_device_time_total > 0:
            bk[bucket(e.key)] += e.self_device_time_total
    print("  --- bucketed CUDA time (per step = total/5) ---")
    print(f"  {'bucket':>14} {'ms/step':>9} {'%':>6}")
    for b, us in sorted(bk.items(), key=lambda x: -x[1]):
        print(f"  {b:>14} {us/1e3/5:>9.2f} {100*us/tot_cuda_us:>5.1f}%")
    print()
    print("  --- TOP 25 kernels by self CUDA time ---")
    print(f"  {'ms/step':>8} {'%':>5} {'#calls':>7}  kernel")
    rows = sorted([e for e in ka if e.self_device_time_total > 0],
                  key=lambda e: -e.self_device_time_total)[:25]
    for e in rows:
        print(f"  {e.self_device_time_total/1e3/5:>8.2f} "
              f"{100*e.self_device_time_total/tot_cuda_us:>4.1f} "
              f"{e.count:>7}  {e.key[:78]}")
    print()

    # ===== 4. CPU vs GPU BOUND =====
    print("="*70)
    print("4) CPU-LAUNCH-BOUND vs COMPUTE-BOUND")
    print("="*70)
    # wall time of the same 5 steps (CUDA events) vs pure kernel time
    wall = cuda_time(f_fwd_bwd, iters=20)
    pure = tot_cuda_us/1e3/5
    print(f"  wall/step (cuda-event)   = {wall:.1f} ms")
    print(f"  pure kernel time/step    = {pure:.1f} ms")
    print(f"  GPU-busy fraction        = {100*pure/wall:.0f}%  "
          f"(<<100% => launch/dispatch-bound; ~100% => compute-bound)")
    # count kernel launches/step
    n_launch = sum(e.count for e in ka if e.self_device_time_total > 0) / 5
    print(f"  CUDA kernel launches/step= {n_launch:.0f}")
    print(f"  avg kernel duration      = {pure*1e3/n_launch:.1f} us "
          f"(tiny kernels + many launches => launch-bound signature)\n")

    print("DONE")


if __name__ == "__main__":
    main()
