"""Localize the SDPA backend / memory blowup for the CrossMAE cross-attention
(big Tq queries x visible Tk keys). Shows default-backend peak memory and
whether FLASH / EFFICIENT run and at what memory, for bf16 and fp32."""
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
print("torch", torch.__version__, "| cuda", torch.version.cuda, flush=True)

for (Tq, Tk, dt) in [(26000, 6600, torch.bfloat16), (37000, 9400, torch.bfloat16), (37000, 9400, torch.float32)]:
    def mk():
        q = torch.randn(8, Tq, 64, device='cuda', dtype=dt, requires_grad=True)
        k = torch.randn(8, Tk, 64, device='cuda', dtype=dt)
        v = torch.randn(8, Tk, 64, device='cuda', dtype=dt)
        return q, k, v
    print(f"\nTq={Tq} Tk={Tk} dtype={dt}", flush=True)
    for name, be in [("default", None), ("flash", SDPBackend.FLASH_ATTENTION),
                     ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("math", SDPBackend.MATH)]:
        q, k, v = mk()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            if be is None:
                o = F.scaled_dot_product_attention(q, k, v)
            else:
                with sdpa_kernel([be]):
                    o = F.scaled_dot_product_attention(q, k, v)
            o.float().sum().backward()
            print(f"  {name:10}: peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
        except Exception as e:
            print(f"  {name:10}: {str(e)[:70]}", flush=True)
print("\ndone", flush=True)
