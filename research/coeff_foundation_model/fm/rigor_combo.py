"""'Reduce cost THEN batch' compounding on REALISTIC configs (full enc12/dec4
cross, real ~26k-token events, fwd+bwd). Absolute events/sec, normalized to the
d512/K=1 baseline = 1.00x. Reuses the verified varlen PACK vs real SEQ harness.

Configs: d in {512,384,256,128}, all enc=12 dec=4 cross.
Optional timing proxy: at d512 and d256, subsample VISIBLE tokens to ~1.6k
(fewer/coarser tokens). Labeled a TIMING PROXY, not a quality config.

Per config x K in {1,2,4,8}: PACK ev/s, x-vs-d512K1, PACK/SEQ, and K=1 GPU-busy%.
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


def mk_mask(B, seed, sub_vis=None):
    """Mask 75%. If sub_vis given, additionally cap #visible to sub_vis (timing proxy)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    m = torch.rand(B["inp"].shape[0], device=DEV, generator=g) < 0.75
    if sub_vis is not None:
        vis_idx = (~m).nonzero(as_tuple=True)[0]
        if vis_idx.numel() > sub_vis:
            keep = vis_idx[torch.randperm(vis_idx.numel(), generator=g, device=DEV)[:sub_vis]]
            newm = torch.ones_like(m)
            newm[keep] = False
            m = newm
    return m


def cuda_time(fn, iters=15, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters, torch.cuda.max_memory_allocated()/1e9


# ---- real per-event forward-drop inputs ----
def prep(model, B, m):
    at = rope_angles(B["t_phys"], model.d // model.heads, *model.lam_t)
    aw = rope_angles(B["wire_pos"], model.d // model.heads, *model.lam_w)
    band, plane = B["band_id"], B["plane_id"]
    g, b = model.film(band, plane, B["wirefeat"]) if model.film is not None else (None, None)
    cond = model.band_emb(band) + model.plane_emb(plane)
    vis = ~m
    xv = model.embed(torch.cat([B["inp"][vis], B["occ"][vis]], -1))
    if model.film is not None: xv = g[vis]*xv + b[vis]
    xv = xv + cond[vis]
    qm = model.mask_tok.expand(int(m.sum()), model.d)
    if model.film is not None: qm = g[m]*qm + b[m]
    qm = qm + cond[m]
    return (xv, at[vis], aw[vis]), (qm.to(xv.dtype), at[m], aw[m])


def seq_step(model, opt, evs, masks):
    opt.zero_grad(set_to_none=True)
    for B, m in zip(evs, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (xv, atv, awv), (qm, atm, awm) = prep(model, B, m)
            for blk in model.enc: xv = blk(xv, atv, awv)
            for blk in model.dec: qm = blk(qm, xv, atm, awm, atv, awv)
            x = model.dec_norm(torch.cat([xv, qm]))
            occ = model.occ_head(x); val = model.val_head(x)
            (occ.float().pow(2).mean() + val.float().pow(2).mean()).backward()


def enc_varlen(blk, x, at, aw, cu, maxs):
    T, d = x.shape
    q, k, v = blk.qkv(blk.n1(x)).chunk(3, -1)
    q = apply_rope(q.view(T, blk.h, blk.hd), at, aw)
    k = apply_rope(k.view(T, blk.h, blk.hd), at, aw)
    v = v.view(T, blk.h, blk.hd)
    o = flash_attn_varlen_func(q.bfloat16(), k.bfloat16(), v.bfloat16(), cu, cu, maxs, maxs,
                               softmax_scale=blk.hd**-0.5)
    x = x + blk.proj(o.reshape(T, d).to(x.dtype))
    return x + blk.mlp(blk.n2(x))


def cross_varlen(blk, q, kv, qat, qaw, kat, kaw, cuq, cuk, maxq, maxk):
    Tq, Tk = q.shape[0], kv.shape[0]
    qh = apply_rope(blk.q(blk.nq(q)).view(Tq, blk.h, blk.hd), qat, qaw)
    k, v = blk.kv(blk.nk(kv)).chunk(2, -1)
    kh = apply_rope(k.view(Tk, blk.h, blk.hd), kat, kaw)
    vh = v.view(Tk, blk.h, blk.hd)
    o = flash_attn_varlen_func(qh.bfloat16(), kh.bfloat16(), vh.bfloat16(), cuq, cuk, maxq, maxk,
                               softmax_scale=blk.hd**-0.5)
    q = q + blk.proj(o.reshape(Tq, blk.h*blk.hd).to(q.dtype))
    return q + blk.mlp(blk.n2(q))


def pack_step(model, opt, evs, masks):
    opt.zero_grad(set_to_none=True)
    xvL, atvL, awvL, vsz = [], [], [], []
    qmL, atmL, awmL, msz = [], [], [], []
    for B, m in zip(evs, masks):
        (xv, atv, awv), (qm, atm, awm) = prep(model, B, m)
        xvL.append(xv); atvL.append(atv); awvL.append(awv); vsz.append(xv.shape[0])
        qmL.append(qm); atmL.append(atm); awmL.append(awm); msz.append(qm.shape[0])
    xv = torch.cat(xvL); atv = torch.cat(atvL); awv = torch.cat(awvL)
    qm = torch.cat(qmL); atm = torch.cat(atmL); awm = torch.cat(awmL)
    cuv = torch.tensor(np.cumsum([0]+vsz), dtype=torch.int32, device=DEV); maxv = max(vsz)
    cum = torch.tensor(np.cumsum([0]+msz), dtype=torch.int32, device=DEV); maxm = max(msz)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for blk in model.enc: xv = enc_varlen(blk, xv, atv, awv, cuv, maxv)
        for blk in model.dec: qm = cross_varlen(blk, qm, xv, atm, awm, atv, awv, cum, cuv, maxm, maxv)
        x = model.dec_norm(torch.cat([xv, qm]))
        occ = model.occ_head(x); val = model.val_head(x)
        (occ.float().pow(2).mean() + val.float().pow(2).mean()).backward()


def gpu_busy(fn):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10): fn()
        torch.cuda.synchronize()
    ksum = sum(e.self_device_time_total for e in p.key_averages())/10/1e3
    wall, _ = cuda_time(fn, iters=20)
    return min(100, 100*ksum/wall)


def run(name, d, evs8, sub_vis=None):
    torch.cuda.empty_cache()
    heads = 8 if d >= 256 else 4
    model = FMModel(128, D.N_BAND, 6, d=d, blocks=12, dec_blocks=4, heads=heads,
                    dec_mode="cross").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 4e-4)
    B0 = evs8[0]; m0 = mk_mask(B0, 0, sub_vis)
    busy = gpu_busy(lambda: pack_step(model, opt, [B0], [m0]))
    rows = []
    for K in (1, 2, 4, 8):
        evs = evs8[:K]; masks = [mk_mask(B, i, sub_vis) for i, B in enumerate(evs)]
        try:
            dtp, pkp = cuda_time(lambda: pack_step(model, opt, evs, masks))
            evps = K/dtp*1000
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); evps = float('nan'); pkp = float('nan')
        try:
            dts, _ = cuda_time(lambda: seq_step(model, opt, evs, masks))
            ratio = evps/(K/dts*1000)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); ratio = float('nan')
        rows.append((K, evps, ratio, pkp))
    del model, opt; torch.cuda.empty_cache()
    return name, d, busy, rows


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    evs8 = load_events(8)
    print(f"GPU {torch.cuda.get_device_name()} | N={[B['inp'].shape[0] for B in evs8]}")
    print("full enc12 dec4 cross; PACK varlen fwd+bwd; ev/s absolute\n")

    configs = [("d512", 512, None), ("d384", 384, None), ("d256", 256, None), ("d128", 128, None),
               ("d512 vis~1.6k [PROXY]", 512, 1600), ("d256 vis~1.6k [PROXY]", 256, 1600)]
    results = [run(nm, d, evs8, sv) for nm, d, sv in configs]

    base = None
    for nm, d, busy, rows in results:
        if nm == "d512":
            base = rows[0][1]  # d512 K=1 ev/s
    print(f"d512/K=1 baseline = {base:.1f} ev/s  (== 1.00x)\n")
    print(f"{'config':>22} {'busy%':>5} | " + " ".join(f"K={k}:ev/s  xB   P/S" for k in (1,2,4,8)))
    print("-"*104)
    for nm, d, busy, rows in results:
        cells = []
        for (K, evps, ratio, pk) in rows:
            cells.append(f"{evps:6.1f} {evps/base:4.2f} {ratio:4.2f}")
        print(f"{nm:>22} {busy:>4.0f}% | " + "  ".join(cells))

    # best combined
    best = max(((nm, K, evps/base) for nm, d, busy, rows in results for (K, evps, r, pk) in rows
                if evps == evps), key=lambda x: x[2])
    print(f"\nBEST combined throughput: {best[0]} @ K={best[1]} -> {best[2]:.2f}x over d512/batch-1")
    print("\ncols per K: [ev/s]  [x vs d512-K1]  [PACK/SEQ ratio]")
    print("DONE")


if __name__ == "__main__":
    main()
