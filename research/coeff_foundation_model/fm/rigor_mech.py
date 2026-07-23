"""PROVE the compute-bound -> ~1.2x mechanism by the extreme case.

Reuse the verified varlen PACK vs real SEQ harness, but strip the decoder and
shrink the encoder. Thesis: batching's multiplicative (->K x) benefit appears
ONLY when per-event GPU work is small enough that launch/overhead dominates.
As we go FULL -> ENCODER-ONLY -> EXTREME-LIGHT, per-event compute shrinks and
the PACK/SEQ ratio should climb toward K x. If encoder-only still ~1.2x, even
the encoder is compute-bound.

Configs (each K in {1,2,4,8}, fwd+bwd):
  A FULL          enc=12 dec=4 cross d=512   (reference ~1.2x)
  B ENC-ONLY      enc=12 dec=0     d=512     (no decoder: encoder over ~6.6k vis
                                              tokens + heads/loss on enc feats)
  C EXTREME-LIGHT enc=2  dec=0     d=128     (tiny per-event GPU work)

For each config at K=1 we also report GPU-busy% = kernel_sum/wall (profiler CUDA).

SEQ  = real per-event loop (model.forward path, encoder-drop; real MSE loss on
       enc features so bwd is real). For dec=0 we run only the encoder + heads.
PACK = varlen block-sparse encoder (+ decoder if present), model's own weights.
"""
import os, sys, glob
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import data as D
from data import DEV
from model import FMModel, rope_angles, apply_rope
from flash_attn import flash_attn_varlen_func
from torch.profiler import profile, ProfilerActivity


def load_events(K):
    D.init_pipeline_cpu()
    fs = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[:K]
    return [D.get_cached(f, device=DEV) for f in fs]


def mk_mask(B, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.rand(B["inp"].shape[0], device=DEV, generator=g) < 0.75


def cuda_time(fn, iters=15, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters, torch.cuda.max_memory_allocated()/1e9


# ---- encoder-only building blocks (real model weights) ----
def enc_prep_seq(model, B, m):
    """Per-event encoder-drop inputs (visible tokens only), as in FMModel.forward_feat."""
    at = rope_angles(B["t_phys"], model.d // model.heads, *model.lam_t)
    aw = rope_angles(B["wire_pos"], model.d // model.heads, *model.lam_w)
    band, plane = B["band_id"], B["plane_id"]
    vis = ~m
    g, b = model.film(band, plane, B["wirefeat"]) if model.film is not None else (None, None)
    cond = model.band_emb(band) + model.plane_emb(plane)
    xv = model.embed(torch.cat([B["inp"][vis], B["occ"][vis]], -1))
    if model.film is not None:
        xv = g[vis] * xv + b[vis]
    xv = xv + cond[vis]
    return xv, at[vis], aw[vis]


def seq_encoder_step(model, opt, evs, masks, bwd):
    if bwd: opt.zero_grad(set_to_none=True)
    for B, m in zip(evs, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xv, atv, awv = enc_prep_seq(model, B, m)
            for blk in model.enc:
                xv = blk(xv, atv, awv)
            x = model.dec_norm(xv)
            occ = model.occ_head(x); val = model.val_head(x)
            loss = occ.float().pow(2).mean() + val.float().pow(2).mean()
        if bwd:
            loss.backward()


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


def pack_encoder_step(model, opt, evs, masks, bwd):
    if bwd: opt.zero_grad(set_to_none=True)
    xv_l, atv_l, awv_l, vsz = [], [], [], []
    for B, m in zip(evs, masks):
        xv, atv, awv = enc_prep_seq(model, B, m)
        xv_l.append(xv); atv_l.append(atv); awv_l.append(awv); vsz.append(xv.shape[0])
    xv = torch.cat(xv_l); atv = torch.cat(atv_l); awv = torch.cat(awv_l)
    cuv = torch.tensor(np.cumsum([0]+vsz), dtype=torch.int32, device=DEV); maxv = max(vsz)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for blk in model.enc:
            xv = enc_varlen(blk, xv, atv, awv, cuv, maxv)
        x = model.dec_norm(xv)
        occ = model.occ_head(x); val = model.val_head(x)
        loss = occ.float().pow(2).mean() + val.float().pow(2).mean()
    if bwd:
        loss.backward()


def gpu_busy(fn):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10): fn()
        torch.cuda.synchronize()
    ksum = sum(e.self_device_time_total for e in p.key_averages())/10/1e3
    wall, _ = cuda_time(fn, iters=20)
    return min(100, 100*ksum/wall), ksum, wall


def run_config(name, blocks, dec_blocks, d, evs8, cap_vis=None):
    torch.cuda.empty_cache()
    heads = 8 if d >= 256 else 4
    model = FMModel(128, D.N_BAND, 6, d=d, blocks=blocks, dec_blocks=max(dec_blocks,1),
                    heads=heads, dec_mode="cross").to(DEV)
    if dec_blocks == 0:
        model.dec = torch.nn.ModuleList()   # encoder-only: drop decoder
    opt = torch.optim.AdamW(model.parameters(), 4e-4)
    print(f"\n### {name}: enc={blocks} dec={dec_blocks} d={d} heads={heads} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    # K=1 GPU-busy
    B0 = evs8[0]; m0 = mk_mask(B0, 0)
    busy, ks, wl = gpu_busy(lambda: pack_encoder_step(model, opt, [B0], [m0], True))
    print(f"  K=1 GPU-busy = {busy:.0f}%  (kernel_sum={ks:.1f}ms / wall={wl:.1f}ms)")
    print(f"  {'K':>2} {'SEQ ev/s':>9} {'PACK ev/s':>10} {'PACK/SEQ':>9} {'seqGB':>7} {'packGB':>7}")
    ratios = {}
    for K in (1, 2, 4, 8):
        evs = evs8[:K]; masks = [mk_mask(B, i) for i, B in enumerate(evs)]
        dts, pks = cuda_time(lambda: seq_encoder_step(model, opt, evs, masks, True))
        try:
            dtp, pkp = cuda_time(lambda: pack_encoder_step(model, opt, evs, masks, True))
            r = (K/dtp*1000)/(K/dts*1000)
            ratios[K] = r
            print(f"  {K:>2} {K/dts*1000:>9.1f} {K/dtp*1000:>10.1f} {r:>8.2f}x {pks:>7.1f} {pkp:>7.1f}")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {K:>2} {K/dts*1000:>9.1f} {'OOM':>10}")
    del model, opt; torch.cuda.empty_cache()
    return busy, ratios


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    evs8 = load_events(8)
    Ns = [B["inp"].shape[0] for B in evs8]
    print(f"GPU {torch.cuda.get_device_name()} | N={Ns}")
    print("fwd+bwd throughput; SEQ = real per-event loop, PACK = varlen block-sparse\n")
    summary = {}
    summary["FULL"] = run_config("A FULL (enc12 dec4 cross d512)", 12, 4, 512, evs8)
    summary["ENC-ONLY"] = run_config("B ENC-ONLY (enc12 dec0 d512)", 12, 0, 512, evs8)
    summary["LIGHT"] = run_config("C EXTREME-LIGHT (enc2 dec0 d128)", 2, 0, 128, evs8)

    print("\n" + "="*60)
    print("SUMMARY: config -> K=1 GPU-busy% -> PACK/SEQ ratio @K=8")
    print("="*60)
    print(f"{'config':>26} {'busy%':>6} {'r@2':>6} {'r@4':>6} {'r@8':>6}")
    for name, (busy, rat) in summary.items():
        print(f"{name:>26} {busy:>5.0f}% {rat.get(2,float('nan')):>5.2f} "
              f"{rat.get(4,float('nan')):>5.2f} {rat.get(8,float('nan')):>5.2f}")
    print("\nThesis: busy high -> ratio ~1.0-1.2x ; busy low -> ratio -> K x")
    print("DONE")


if __name__ == "__main__":
    main()
