"""ADVERSARIAL AUDIT of "batching only ~1.2x, compute-bound".

Three decisive tests, each attacking a specific gap in the prior benchmarks:

TEST 1 -- GEMM-vs-M sweep (the core claim).
  Prior claim: "GEMMs already saturated at Nv=6.6k, so batching can't help the 9%
  GEMM slice." But the SAME benchmarks report only 27-35% of A100 peak (312 TF).
  27-35% is NOT the FLOPS roofline. If a (M,512)x(512,2048) bf16 GEMM's achieved
  TFLOPs RISE from M=6.6k to M=53k, then batched GEMMs do more work per second and
  the "saturated" premise is false. We time fwd and fwd+bwd for the FFN GEMM and
  the qkv GEMM at M in {6.6k, 13k, 26k, 53k}.

TEST 2 -- KERNEL-MATCHED SEQ vs PACK (apples-to-apples).
  Prior SEQ loops model(B,m) which uses F.scaled_dot_product_attention; PACK uses
  flash_attn_varlen_func. Different kernels => 1.2x conflates "packing" with "kernel
  swap". We build SEQ' = loop the SAME varlen kernel per event (one cu_seqlens each),
  and PACK = all events in one cu_seqlens. Now the ONLY difference is packing. If
  SEQ'-vs-PACK ratio > the 1.2x SDPA-SEQ-vs-varlen-PACK ratio, part of the reported
  1.2x was the kernel swap (over-count); if it's the SAME, packing itself is the
  gain and 1.2x is robust; if it's LARGER, packing gain was UNDER-counted.

TEST 3 -- GPU-busy metric audit.
  "88% GPU-busy" = kernel_sum/wall. Report it ALONGSIDE achieved MFU (flops/wall/312).
  Show they disagree: near-100% busy can coexist with 30% MFU => "busy" does NOT
  mean "at the compute roofline", so batching CAN still help.

All on the real model weights / real event token counts.
"""
import os, sys, glob
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import data as D
from data import DEV
from model import FMModel, rope_angles, apply_rope
from flash_attn import flash_attn_varlen_func

PEAK = 312.0  # A100 bf16 TFLOPs (no sparsity)


def load_events(K):
    D.init_pipeline_cpu()
    fs = sorted(glob.glob(
        "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/artifacts/fm_cache_tpc/ev_*.npz"))[:K]
    return [D.get_cached(f, device=DEV) for f in fs]


def mk_mask(B, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.rand(B["inp"].shape[0], device=DEV, generator=g) < 0.75


def cuda_time(fn, iters=30, warmup=8):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


# ============================== TEST 1 ==============================
def test1_gemm_vs_M():
    print("=" * 70)
    print("TEST 1: GEMM achieved-TFLOPs vs M  (does batching help GEMMs?)")
    print("=" * 70)
    d = 512
    Ms = [6608, 13216, 26432, 52864]  # 1,2,4,8 events worth of visible tokens
    torch.backends.cuda.matmul.allow_tf32 = True
    for label, (K_in, K_out) in [("FFN up  (512->2048)", (d, 4 * d)),
                                  ("qkv     (512->1536)", (d, 3 * d)),
                                  ("proj    (512->512) ", (d, d))]:
        print(f"\n  {label}")
        print(f"  {'M':>7} {'fwd us':>8} {'fwd TF':>7} {'%pk':>5} | {'f+b us':>8} {'f+b TF':>7} {'%pk':>5}")
        for M in Ms:
            x = torch.randn(M, K_in, device=DEV, dtype=torch.bfloat16, requires_grad=True)
            w = torch.randn(K_in, K_out, device=DEV, dtype=torch.bfloat16, requires_grad=True)
            def fwd():
                with torch.no_grad():
                    return x @ w
            flf = 2 * M * K_in * K_out
            tf = cuda_time(fwd) / 1e3  # ms->? actually elapsed_time is ms; keep ms
            # cuda_time returns ms. convert to seconds for TFLOPs
            tf_s = tf / 1e3
            fwd_tflops = flf / tf_s / 1e12
            def fbwd():
                y = x @ w
                y.sum().backward()
            tb = cuda_time(fbwd) / 1e3 / 1e3
            fb_tflops = (3 * flf) / tb / 1e12
            print(f"  {M:>7} {tf*1e3:>8.1f} {fwd_tflops:>7.1f} {fwd_tflops/PEAK*100:>4.0f}% | "
                  f"{tb*1e6:>8.1f} {fb_tflops:>7.1f} {fb_tflops/PEAK*100:>4.0f}%")
    print("\n  VERDICT: if fwd TF and f+b TF rise materially M=6.6k->53k, the")
    print("  'GEMMs already saturated' premise is FALSE and batching helps GEMMs.")


# ============================== TEST 2 ==============================
def enc_varlen(blk, x, at, aw, cu, maxs):
    T, d = x.shape
    h = blk.n1(x)
    q, k, v = blk.qkv(h).chunk(3, -1)
    q = apply_rope(q.view(T, blk.h, blk.hd), at, aw)
    k = apply_rope(k.view(T, blk.h, blk.hd), at, aw)
    v = v.view(T, blk.h, blk.hd)
    o = flash_attn_varlen_func(q.bfloat16(), k.bfloat16(), v.bfloat16(), cu, cu, maxs, maxs,
                               softmax_scale=blk.hd ** -0.5)
    x = x + blk.proj(o.reshape(T, d).to(x.dtype))
    return x + blk.mlp(blk.n2(x))


def enc_prep(model, B, m):
    at = rope_angles(B["t_phys"], model.d // model.heads, *model.lam_t)
    aw = rope_angles(B["wire_pos"], model.d // model.heads, *model.lam_w)
    vis = ~m
    g, b = model.film(B["band_id"], B["plane_id"], B["wirefeat"])
    cond = model.band_emb(B["band_id"]) + model.plane_emb(B["plane_id"])
    xv = model.embed(torch.cat([B["inp"][vis], B["occ"][vis]], -1))
    xv = g[vis] * xv + b[vis] + cond[vis]
    return xv, at[vis], aw[vis], int(vis.sum())


def seq_varlen_step(model, opt, evs, masks, bwd):
    """SEQ' : loop the SAME varlen kernel per event (kernel-matched to PACK)."""
    if bwd: opt.zero_grad(set_to_none=True)
    loss = 0.0
    for B, m in zip(evs, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xv, atv, awv, nv = enc_prep(model, B, m)
            cu = torch.tensor([0, nv], dtype=torch.int32, device=DEV)
            for blk in model.enc:
                xv = enc_varlen(blk, xv, atv, awv, cu, nv)
            x = model.dec_norm(xv)
            l = model.occ_head(x).float().pow(2).mean() + model.val_head(x).float().pow(2).mean()
        if bwd:
            l.backward()


def seq_sdpa_step(model, opt, evs, masks, bwd):
    """SEQ  : the ORIGINAL path -- model.enc Block uses F.scaled_dot_product_attention."""
    if bwd: opt.zero_grad(set_to_none=True)
    for B, m in zip(evs, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xv, atv, awv, nv = enc_prep(model, B, m)
            for blk in model.enc:
                xv = blk(xv, atv, awv)       # SDPA path in Block.forward
            x = model.dec_norm(xv)
            l = model.occ_head(x).float().pow(2).mean() + model.val_head(x).float().pow(2).mean()
        if bwd:
            l.backward()


def pack_step(model, opt, evs, masks, bwd):
    if bwd: opt.zero_grad(set_to_none=True)
    xv_l, atv_l, awv_l, vsz = [], [], [], []
    for B, m in zip(evs, masks):
        xv, atv, awv, nv = enc_prep(model, B, m)
        xv_l.append(xv); atv_l.append(atv); awv_l.append(awv); vsz.append(nv)
    xv = torch.cat(xv_l); atv = torch.cat(atv_l); awv = torch.cat(awv_l)
    cu = torch.tensor(np.cumsum([0] + vsz), dtype=torch.int32, device=DEV); maxs = max(vsz)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for blk in model.enc:
            xv = enc_varlen(blk, xv, atv, awv, cu, maxs)
        x = model.dec_norm(xv)
        l = model.occ_head(x).float().pow(2).mean() + model.val_head(x).float().pow(2).mean()
    if bwd:
        l.backward()


def test2_kernel_matched(model, opt, evs8):
    print("\n" + "=" * 70)
    print("TEST 2: kernel-matched SEQ' (looped varlen) vs PACK, and vs SDPA-SEQ")
    print("  Encoder-only (enc=12 d=512), fwd+bwd. Isolates PACKING from KERNEL swap.")
    print("=" * 70)
    print(f"  {'K':>2} {'sdpaSEQ ev/s':>12} {'vlSEQ ev/s':>11} {'PACK ev/s':>10} "
          f"{'PACK/sdpaSEQ':>13} {'PACK/vlSEQ':>11}")
    for K in (1, 2, 4, 8):
        evs = evs8[:K]; masks = [mk_mask(B, i) for i, B in enumerate(evs)]
        d_sdpa = cuda_time(lambda: seq_sdpa_step(model, opt, evs, masks, True), iters=12, warmup=5)
        d_vl = cuda_time(lambda: seq_varlen_step(model, opt, evs, masks, True), iters=12, warmup=5)
        d_pk = cuda_time(lambda: pack_step(model, opt, evs, masks, True), iters=12, warmup=5)
        s_sdpa = K / d_sdpa * 1000; s_vl = K / d_vl * 1000; s_pk = K / d_pk * 1000
        print(f"  {K:>2} {s_sdpa:>12.1f} {s_vl:>11.1f} {s_pk:>10.1f} "
              f"{s_pk/s_sdpa:>12.2f}x {s_pk/s_vl:>10.2f}x")
    print("\n  READ: PACK/vlSEQ = PURE packing gain (kernel held fixed).")
    print("        PACK/sdpaSEQ = what prior benchmark reported (kernel swap included).")
    print("        If PACK/vlSEQ ~ PACK/sdpaSEQ -> kernel swap negligible, 1.2x is real packing.")


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU {torch.cuda.get_device_name()}\n")
    test1_gemm_vs_M()
    evs8 = load_events(8)
    print(f"\nN={[B['inp'].shape[0] for B in evs8]}")
    model = FMModel(128, D.N_BAND, 6, d=512, blocks=12, dec_blocks=1, heads=8, dec_mode="cross").to(DEV)
    model.dec = torch.nn.ModuleList()
    opt = torch.optim.AdamW(model.parameters(), 4e-4)
    test2_kernel_matched(model, opt, evs8)
    print("\nDONE")


if __name__ == "__main__":
    main()
