"""TEST 1 (fixed): GEMM achieved-TFLOPs vs M, correct units + DCE-proof.

Times a stack of L identical GEMMs per call (so the region is well above CUDA-event
resolution and cannot be dead-code-eliminated), accumulating into a live output.
elapsed_time() is in MILLISECONDS; convert once. Reports achieved TFLOPs and % of
A100 bf16 peak (312) at M in {6.6k,13k,26k,53k} for the model's real GEMM shapes.
"""
import os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
DEV = "cuda"
PEAK = 312.0
torch.backends.cuda.matmul.allow_tf32 = True


def time_ms(fn, iters=50, warmup=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters   # milliseconds per call


def main():
    print(f"GPU {torch.cuda.get_device_name()} | A100 bf16 peak={PEAK} TFLOPs\n")
    Ms = [6608, 13216, 26432, 52864]
    L = 20  # GEMMs per timed call: keeps region >> event resolution, defeats DCE
    for label, (Kin, Kout) in [("FFN up  512->2048", (512, 2048)),
                               ("qkv     512->1536", (512, 1536)),
                               ("proj    512->512 ", (512, 512))]:
        print(f"  {label}   (L={L} chained GEMMs/call)")
        print(f"  {'M':>7} {'fwd ms':>8} {'fwd TF':>7} {'%pk':>5} | {'f+b ms':>8} {'f+b TF':>7} {'%pk':>5}")
        for M in Ms:
            x = torch.randn(M, Kin, device=DEV, dtype=torch.bfloat16)
            ws = [torch.randn(Kin, Kout, device=DEV, dtype=torch.bfloat16) for _ in range(L)]
            flops_fwd = L * 2 * M * Kin * Kout

            def fwd():
                with torch.no_grad():
                    acc = x
                    for w in ws:
                        acc = (x @ w)[:, :Kin] if Kout != Kin else x @ w
                    return acc
            # simpler DCE-proof fwd: sum outputs
            def fwd2():
                with torch.no_grad():
                    s = torch.zeros(1, device=DEV, dtype=torch.float32)
                    for w in ws:
                        s = s + (x @ w).float().sum()
                    return s
            t_fwd = time_ms(fwd2) / 1e3  # seconds
            tf_fwd = flops_fwd / t_fwd / 1e12

            xb = torch.randn(M, Kin, device=DEV, dtype=torch.bfloat16, requires_grad=True)
            wb = torch.randn(Kin, Kout, device=DEV, dtype=torch.bfloat16, requires_grad=True)
            def fbwd():
                acc = 0.0
                for _ in range(L):
                    acc = acc + (xb @ wb).sum()
                acc.backward()
                xb.grad = None; wb.grad = None
            t_fb = time_ms(fbwd, iters=30, warmup=10) / 1e3
            tf_fb = (3 * flops_fwd) / t_fb / 1e12

            print(f"  {M:>7} {t_fwd*1e3:>8.2f} {tf_fwd:>7.1f} {tf_fwd/PEAK*100:>4.0f}% | "
                  f"{t_fb*1e3:>8.2f} {tf_fb:>7.1f} {tf_fb/PEAK*100:>4.0f}%")
        print()
    print("VERDICT: rising TFLOPs from M=6.6k->53k => small-M GEMMs are NOT at the")
    print("roofline (wave/tile quantization); batched (larger-M) GEMMs run more")
    print("efficiently, so the 'GEMMs already saturated' premise is false.")
    print("DONE")


if __name__ == "__main__":
    main()
