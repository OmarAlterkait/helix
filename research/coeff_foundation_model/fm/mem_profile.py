"""Thorough memory profile of the FMModel training step — WHY are we batch=1?
Answers: real N, peak mem (fwd+bwd, bf16), which SDPA backend is active (flash vs
the 32k^2 MATH materialization), activation share (fwd vs fwd+bwd), encoder-drop
saving, gradient-checkpointing headroom, and how big a batch fits."""
import glob, torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
import data as D
from model import FMModel, losses
D.init_pipeline_cpu()
dev = "cuda"

paths = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))
B = D.get_cached(paths[0], device=dev)
N = B["inp"].shape[0]
GB = 1024 ** 3
print(f"GPU: {torch.cuda.get_device_name()}  total {torch.cuda.get_device_properties(0).total_memory/GB:.1f} GB")
print(f"event N tokens = {N}\n")


def build():
    torch.manual_seed(0)
    m = FMModel(128, D.N_BAND, 6, d=512, blocks=12, dec_blocks=4, heads=8).to(dev)
    return m, torch.optim.AdamW(m.parameters(), 1e-4)


def peak():
    torch.cuda.synchronize(); return torch.cuda.max_memory_allocated() / GB


def step(m, opt, mask, fwd_only=False, amp=True):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        occ, mu, lv = m(B, mask)
        bce, val = losses(occ, mu, lv, B, mask)
        loss = bce + val
    if not fwd_only:
        loss.backward(); opt.step()
    return peak()


m, opt = build()
mask = (torch.rand(N, device=dev) < 0.75)
wt = sum(p.numel() for p in m.parameters()) * 4 / GB
print(f"weights (fp32)        : {wt:.2f} GB")
step(m, opt, mask)                                    # warmup (optimizer state alloc)
print(f"fwd+bwd+step peak     : {step(m, opt, mask):.2f} GB   <- the real per-event cost")
print(f"fwd ONLY peak         : {step(m, opt, mask, fwd_only=True):.2f} GB   (gap = backward activations)\n")

print("SDPA backend probe (is the decoder 32k^2 matrix being materialized?):")
for name, bk in [("FLASH", SDPBackend.FLASH_ATTENTION),
                 ("EFFICIENT(mem)", SDPBackend.EFFICIENT_ATTENTION),
                 ("MATH(materializes)", SDPBackend.MATH)]:
    try:
        with sdpa_kernel(bk):
            print(f"  {name:20s}: {step(m, opt, mask):.2f} GB")
    except Exception as e:
        print(f"  {name:20s}: unavailable ({str(e)[:45]})")
print()

# gradient checkpointing headroom (wrap every block)
def ckpt_step(m, opt, mask):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); opt.zero_grad(set_to_none=True)
    orig_enc = [blk.forward for blk in m.enc]; orig_dec = [blk.forward for blk in m.dec]
    for blk in list(m.enc) + list(m.dec):
        f = blk.forward
        blk.forward = (lambda *a, _f=f: checkpoint(_f, *a, use_reentrant=False))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        occ, mu, lv = m(B, mask); bce, val = losses(occ, mu, lv, B, mask); loss = bce + val
    loss.backward(); opt.step()
    p = peak()
    for blk, f in zip(list(m.enc) + list(m.dec), orig_enc + orig_dec): blk.forward = f
    return p
try:
    print(f"fwd+bwd WITH grad-checkpointing : {ckpt_step(m, opt, mask):.2f} GB  (trades ~33% compute for memory)\n")
except Exception as e:
    print(f"grad-checkpoint test failed: {str(e)[:80]}\n")

# how big a batch fits? replicate the event K times (block-diag would prevent leak;
# here we only need the MEMORY slope, so a plain concat of the per-event tensors is fine)
def concat(K):
    import torch as T
    out = {}
    for k, v in B.items():
        if k in ("cell", "slot"):  # per-row indices -> offset by cell count per copy
            continue
        out[k] = T.cat([v] * K, 0) if hasattr(v, "shape") and v.dim() >= 1 else v
    return out, N * K

print("batch headroom (K events concatenated -> peak mem; OOM = too big at batch=1 cost):")
for K in (1, 2, 3, 4):
    try:
        Bk, Nk = concat(K)
        gmask = (torch.rand(Nk, device=dev) < 0.75)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); opt.zero_grad(set_to_none=True)
        globals()["B"] = Bk
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = m(Bk, gmask)
            # value loss needs cell/slot which we dropped; use occ-only proxy for the mem test
            loss = occ.float().pow(2).mean()
        loss.backward(); opt.step()
        print(f"  K={K} (N={Nk:>6}): {peak():.2f} GB")
    except RuntimeError as e:
        print(f"  K={K} (N={N*K:>6}): OOM / {str(e)[:40]}"); break
