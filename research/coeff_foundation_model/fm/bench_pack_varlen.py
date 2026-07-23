"""DECISIVE packing benchmark v2 — TRUE block-sparse packing via flash_attn_varlen.

v1 (dense additive block-diagonal SDPA mask) was a FLAWED test: a (T,T) float mask
forces SDPA onto the MATH backend, materializing the full T^2 score matrix -> slow +
OOM. That is NOT what real varlen packing does. flash_attn_varlen keeps attention
BLOCK-SPARSE (only intra-event blocks computed): attention work = sum(N_i^2), memory
linear in tokens. This is the honest test of whether packing K events raises ev/s.

SEQ : K events each separate fwd(+bwd)  (current regime; what grad-accum tests).
PACK: K events packed; encoder = flash_attn_varlen (one cu_seqlens), cross-decoder =
      flash_attn_varlen with separate q/k cu_seqlens. GEMMs become K x bigger; launches /K.

Production config: d=512, enc=12, dec=4 cross, heads=8, head_dim=64, ffn_mult=4.
"""
import sys, os, time, glob, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import data as D
from data import DEV
from model import FMModel, rope_angles, apply_rope
from flash_attn import flash_attn_varlen_func


def load_events(K):
    D.init_pipeline_cpu()
    fs = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[:K]
    return [D.get_cached(f, device=DEV) for f in fs]


def make_mask(B, r=0.75):
    return torch.rand(B["inp"].shape[0], device=DEV) < r


# ---- SEQ: the real model forward ----
def seq_forward(model, evs, masks):
    outs = []
    for B, m in zip(evs, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, m)
            outs.append(occ.float().pow(2).mean() + mu.float().pow(2).mean())
    return sum(outs)


# ---- PACK: encoder-drop + cross-decoder, flash varlen ----
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


def cross_varlen(blk, q, kv, qa_t, qa_w, ka_t, ka_w, cuq, cuk, maxq, maxk):
    Tq, Tk = q.shape[0], kv.shape[0]
    qh = apply_rope(blk.q(blk.nq(q)).view(Tq, blk.h, blk.hd), qa_t, qa_w)
    k, v = blk.kv(blk.nk(kv)).chunk(2, -1)
    kh = apply_rope(k.view(Tk, blk.h, blk.hd), ka_t, ka_w)
    vh = v.view(Tk, blk.h, blk.hd)
    o = flash_attn_varlen_func(qh.bfloat16(), kh.bfloat16(), vh.bfloat16(), cuq, cuk, maxq, maxk,
                               softmax_scale=blk.hd ** -0.5)
    q = q + blk.proj(o.reshape(Tq, blk.h * blk.hd).to(q.dtype))
    return q + blk.mlp(blk.n2(q))


def packed_forward(model, evs, masks):
    d = model.d
    xv_l, atv_l, awv_l, vis_sz = [], [], [], []
    qm_l, atm_l, awm_l, msk_sz = [], [], [], []
    for B, tok_mask in zip(evs, masks):
        at = rope_angles(B["t_phys"], model.d // model.heads, *model.lam_t)
        aw = rope_angles(B["wire_pos"], model.d // model.heads, *model.lam_w)
        band, plane = B["band_id"], B["plane_id"]
        g, b = model.film(band, plane, B["wirefeat"]) if model.film is not None else (None, None)
        cond = model.band_emb(band) + model.plane_emb(plane)
        vis = ~tok_mask
        xv = model.embed(torch.cat([B["inp"][vis], B["occ"][vis]], -1))
        if model.film is not None:
            xv = g[vis] * xv + b[vis]
        xv = xv + cond[vis]
        xv_l.append(xv); atv_l.append(at[vis]); awv_l.append(aw[vis]); vis_sz.append(int(vis.sum()))
        qm = model.mask_tok.expand(int(tok_mask.sum()), d)
        if model.film is not None:
            qm = g[tok_mask] * qm + b[tok_mask]
        qm = qm + cond[tok_mask]
        qm_l.append(qm); atm_l.append(at[tok_mask]); awm_l.append(aw[tok_mask]); msk_sz.append(int(tok_mask.sum()))

    xv = torch.cat(xv_l); atv = torch.cat(atv_l); awv = torch.cat(awv_l)
    qm = torch.cat(qm_l); atm = torch.cat(atm_l); awm = torch.cat(awm_l)
    cuv = torch.tensor(np.cumsum([0] + vis_sz), dtype=torch.int32, device=DEV); maxv = max(vis_sz)
    cum = torch.tensor(np.cumsum([0] + msk_sz), dtype=torch.int32, device=DEV); maxm = max(msk_sz)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for blk in model.enc:
            xv = enc_varlen(blk, xv, atv, awv, cuv, maxv)
        for blk in model.dec:
            qm = cross_varlen(blk, qm, xv, atm, awm, atv, awv, cum, cuv, maxm, maxv)
        x = model.dec_norm(torch.cat([xv, qm]))
        return model.occ_head(x).float().pow(2).mean() + model.val_head(x).float().pow(2).mean()


def timed(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters, torch.cuda.max_memory_allocated() / 1e9


def fwd_flops(evs, masks, d=512, enc=12, dec=4):
    fl = 0.0
    for B, m in zip(evs, masks):
        N = B["inp"].shape[0]; Nv = int((~m).sum()); Nm = int(m.sum())
        fl += 2 * Nv * 256 * d
        for _ in range(enc):
            fl += 2 * Nv * d * 3 * d + 2 * Nv * d * d + 2 * (2 * Nv * d * 4 * d) + 2 * (2 * Nv * Nv * d)
        for _ in range(dec):
            fl += 2 * Nm * d * d + 2 * Nv * d * 2 * d + 2 * Nm * d * d + 2 * (2 * Nm * d * 4 * d) + 2 * (2 * Nm * Nv * d)
        fl += 2 * N * d * 256
    return fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", default="1,2,4,8")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--bwd", action="store_true")
    args = ap.parse_args()
    Kmax = max(int(k) for k in args.Ks.split(","))
    evs_all = load_events(Kmax)
    Ns = [B["inp"].shape[0] for B in evs_all]
    print(f"GPU {torch.cuda.get_device_name()} | {len(evs_all)} events N={Ns}")
    model = FMModel(128, D.N_BAND, 6, d=512, blocks=12, dec_blocks=4, heads=8, dec_mode="cross").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 4e-4)
    print(f"FMModel d=512 enc=12 dec=4 cross heads=8 | params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"| {'fwd+bwd' if args.bwd else 'fwd-only'}\n")
    print(f"{'K':>2} {'regime':>6} {'ms/step':>8} {'ms/event':>9} {'ev/s':>7} {'peakGB':>7} {'TFLOPs':>7} {'%peak':>6}")
    seq_base = {}
    for K in [int(k) for k in args.Ks.split(",")]:
        evs = evs_all[:K]; masks = [make_mask(B) for B in evs]
        flops = fwd_flops(evs, masks) * (3.0 if args.bwd else 1.0)
        row = {}
        for regime, fwd in (("seq", seq_forward), ("pack", packed_forward)):
            def step():
                opt.zero_grad(set_to_none=True)
                out = fwd(model, evs, masks)
                if args.bwd:
                    out.backward()
            try:
                dt, pk = timed(step, args.iters)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); print(f"{K:>2} {regime:>6} {'OOM':>8}"); continue
            tflops = flops / dt / 1e12
            print(f"{K:>2} {regime:>6} {dt*1000:>8.1f} {dt*1000/K:>9.1f} {K/dt:>7.1f} {pk:>7.1f} {tflops:>7.1f} {tflops/312*100:>5.1f}%")
            row[regime] = K / dt
        if "seq" in row and "pack" in row:
            print(f"   -> PACK/SEQ ev/s speedup = {row['pack']/row['seq']:.2f}x")
        print()


if __name__ == "__main__":
    main()
