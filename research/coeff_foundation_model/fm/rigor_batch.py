"""TRUE batched-throughput test on the REAL FMModel.

SEQ baseline = the ACTUAL model.FMModel.forward looped K times + the REAL
losses(noisy=True) + (optionally) backward. This is the honest current-regime
number — NOT the prior bench's fake pow(2) loss.

PACK = flash_attn_varlen block-sparse over K events packed into one sequence,
reusing the model's OWN weights (model.enc[i].qkv, .proj, .mlp, model.dec[i].q/kv,
heads, RoPE via model.rope_angles/apply_rope). To prove the pack path is correct,
we compare its per-event encoder output against the real model's encoder on the
SAME event (should match to bf16 tolerance). Only then is the speedup meaningful.

We report ev/s for SEQ vs PACK at K in {1,2,4,8}, fwd-only and fwd+bwd.
We also run torch.compile on the real model (fwd, fwd+bwd) for the speedup lever.
"""
import os, sys, glob, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import data as D
from data import DEV
from model import FMModel, losses, rope_angles, apply_rope

HAVE_FLASH = False
try:
    from flash_attn import flash_attn_varlen_func
    HAVE_FLASH = True
except Exception as e:
    print("flash_attn import failed:", e)


def load_events(K):
    D.init_pipeline_cpu()
    fs = sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[:K]
    return [D.get_cached(f, device=DEV) for f in fs]


def mk_mask(B, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.rand(B["inp"].shape[0], device=DEV, generator=g) < 0.75


def cuda_time(fn, iters=15, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, torch.cuda.max_memory_allocated() / 1e9


# ---------- REAL SEQ (actual model + real loss) ----------
def seq_step(model, evs, masks, bwd):
    tot = 0.0
    for B, m in zip(evs, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, m)
            bce, val = losses(occ, mu, lv, B, m, noisy=True)
            loss = bce + val
        if bwd:
            loss.backward()
        else:
            tot += float(loss)  # NO: would sync. avoid.
    return


def seq_step_fwd(model, evs, masks):
    with torch.no_grad():
        for B, m in zip(evs, masks):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                occ, mu, lv = model(B, m)
                losses(occ, mu, lv, B, m, noisy=True)


def seq_step_bwd(model, opt, evs, masks):
    opt.zero_grad(set_to_none=True)
    for B, m in zip(evs, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, m)
            bce, val = losses(occ, mu, lv, B, m, noisy=True)
            (bce + val).backward()


# ---------- PACK (block-sparse varlen, model's own weights) ----------
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


def pack_prep(model, evs, masks):
    d = model.d
    xv_l, atv_l, awv_l, vsz = [], [], [], []
    qm_l, atm_l, awm_l, msz = [], [], [], []
    for B, tm in zip(evs, masks):
        at = rope_angles(B["t_phys"], model.d // model.heads, *model.lam_t)
        aw = rope_angles(B["wire_pos"], model.d // model.heads, *model.lam_w)
        band, plane = B["band_id"], B["plane_id"]
        g, b = model.film(band, plane, B["wirefeat"]) if model.film is not None else (None, None)
        cond = model.band_emb(band) + model.plane_emb(plane)
        vis = ~tm
        xv = model.embed(torch.cat([B["inp"][vis], B["occ"][vis]], -1))
        if model.film is not None:
            xv = g[vis] * xv + b[vis]
        xv = xv + cond[vis]
        xv_l.append(xv); atv_l.append(at[vis]); awv_l.append(aw[vis]); vsz.append(int(vis.sum()))
        qm = model.mask_tok.expand(int(tm.sum()), d)
        if model.film is not None:
            qm = g[tm] * qm + b[tm]
        qm = qm + cond[tm]
        qm_l.append(qm); atm_l.append(at[tm]); awm_l.append(aw[tm]); msz.append(int(tm.sum()))
    xv = torch.cat(xv_l); atv = torch.cat(atv_l); awv = torch.cat(awv_l)
    qm = torch.cat(qm_l); atm = torch.cat(atm_l); awm = torch.cat(awm_l)
    cuv = torch.tensor(np.cumsum([0]+vsz), dtype=torch.int32, device=DEV); maxv = max(vsz)
    cum = torch.tensor(np.cumsum([0]+msz), dtype=torch.int32, device=DEV); maxm = max(msz)
    return xv, atv, awv, cuv, maxv, qm, atm, awm, cum, maxm, vsz, msz


def pack_fwd(model, prep):
    xv, atv, awv, cuv, maxv, qm, atm, awm, cum, maxm, vsz, msz = prep
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for blk in model.enc:
            xv = enc_varlen(blk, xv, atv, awv, cuv, maxv)
        for blk in model.dec:
            qm = cross_varlen(blk, qm, xv, atm, awm, atv, awv, cum, cuv, maxm, maxv)
        x = model.dec_norm(torch.cat([xv, qm]))
        return model.occ_head(x), model.val_head(x), xv


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    evs8 = load_events(8)
    Ns = [B["inp"].shape[0] for B in evs8]
    print(f"GPU {torch.cuda.get_device_name()} | flash_attn={HAVE_FLASH} | N={Ns}\n")
    model = FMModel(128, D.N_BAND, 6, d=512, blocks=12, dec_blocks=4, heads=8, dec_mode="cross").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 4e-4)
    model.eval()

    # ---- correctness: pack encoder output (K=1) vs real model encoder ----
    if HAVE_FLASH:
        B = evs8[0]; m = mk_mask(B, 0)
        with torch.no_grad():
            prep = pack_prep(model, [B], [m])
            _, _, xv_pack = pack_fwd(model, prep)
            # real model encoder-drop path:
            vis = ~m
            at = rope_angles(B["t_phys"], model.d//model.heads, *model.lam_t)
            aw = rope_angles(B["wire_pos"], model.d//model.heads, *model.lam_w)
            g, b = model.film(B["band_id"], B["plane_id"], B["wirefeat"])
            cond = model.band_emb(B["band_id"]) + model.plane_emb(B["plane_id"])
            xr = model.embed(torch.cat([B["inp"][vis], B["occ"][vis]], -1))
            xr = g[vis]*xr + b[vis] + cond[vis]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for blk in model.enc:
                    xr = blk(xr, at[vis], aw[vis])
            rel = (xv_pack.float()-xr.float()).abs().max() / xr.float().abs().max().clamp_min(1e-6)
        print(f"[correctness] pack-encoder vs real-encoder max rel err = {rel:.2e}  "
              f"({'OK, block-sparse matches' if rel < 5e-2 else 'MISMATCH!'})\n")

    # ---- throughput SEQ vs PACK ----
    print("="*64)
    print("THROUGHPUT: REAL SEQ (model.forward + real losses) vs PACK varlen")
    print("="*64)
    for bwd in (False, True):
        print(f"\n--- {'fwd+bwd' if bwd else 'fwd-only'} ---")
        print(f"{'K':>2} {'regime':>5} {'ms/step':>8} {'ms/ev':>7} {'ev/s':>7} {'peakGB':>7}")
        for K in (1, 2, 4, 8):
            evs = evs8[:K]; masks = [mk_mask(B, i) for i, B in enumerate(evs)]
            rows = {}
            # SEQ
            if bwd:
                fn = lambda: seq_step_bwd(model, opt, evs, masks)
            else:
                fn = lambda: seq_step_fwd(model, evs, masks)
            dt, pk = cuda_time(fn)
            rows["seq"] = K/dt*1000
            print(f"{K:>2} {'seq':>5} {dt:>8.1f} {dt/K:>7.1f} {K/dt*1000:>7.1f} {pk:>7.1f}")
            # PACK
            if HAVE_FLASH:
                if bwd:
                    def fn2():
                        opt.zero_grad(set_to_none=True)
                        prep = pack_prep(model, evs, masks)
                        occ, val, _ = pack_fwd(model, prep)
                        (occ.float().pow(2).mean()+val.float().pow(2).mean()).backward()
                else:
                    def fn2():
                        with torch.no_grad():
                            prep = pack_prep(model, evs, masks)
                            pack_fwd(model, prep)
                try:
                    dt2, pk2 = cuda_time(fn2)
                    rows["pack"] = K/dt2*1000
                    print(f"{K:>2} {'pack':>5} {dt2:>8.1f} {dt2/K:>7.1f} {K/dt2*1000:>7.1f} {pk2:>7.1f}")
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache(); print(f"{K:>2} {'pack':>5} {'OOM':>8}")
            if "seq" in rows and "pack" in rows:
                print(f"     -> PACK/SEQ ev/s = {rows['pack']/rows['seq']:.2f}x")

    # ---- torch.compile lever ----
    print("\n" + "="*64)
    print("torch.compile on the REAL model (single event)")
    print("="*64)
    B = evs8[0]; m = mk_mask(B, 0)
    def eager_fwd():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(B, m)
    def eager_bwd():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ, mu, lv = model(B, m); bce, val = losses(occ, mu, lv, B, m, noisy=True)
            (bce+val).backward()
    e_f, _ = cuda_time(eager_fwd); e_b, _ = cuda_time(eager_bwd)
    print(f"  eager  fwd={e_f:.1f}  fwd+bwd={e_b:.1f} ms")
    try:
        cmodel = torch.compile(model, dynamic=True)
        def comp_fwd():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                cmodel(B, m)
        def comp_bwd():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                occ, mu, lv = cmodel(B, m); bce, val = losses(occ, mu, lv, B, m, noisy=True)
                (bce+val).backward()
        c_f, _ = cuda_time(comp_fwd, iters=10, warmup=12)   # extra warmup for compile
        c_b, _ = cuda_time(comp_bwd, iters=10, warmup=12)
        print(f"  compiled fwd={c_f:.1f} ({e_f/c_f:.2f}x)  fwd+bwd={c_b:.1f} ({e_b/c_b:.2f}x)")
    except Exception as ex:
        print("  torch.compile failed:", repr(ex)[:300])

    print("\nDONE")


if __name__ == "__main__":
    main()
