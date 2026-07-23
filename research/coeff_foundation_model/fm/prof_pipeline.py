"""Full ENCODER + DECODER pipeline time, grouped vs full attention, for both masking regimes.
Encoder-drop MAE: encoder runs on VISIBLE tokens (12 blocks); CrossMAE decoder = masked queries
cross-attend visible keys (4 blocks). Regimes: random-0.75 (vis=25%) and plane-mask (hide 1 plane, vis=5/6).
"""
import sys, os, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from prof_serial import SBlock, uniform_attn
from prof_batch_dec import grouped_cross
from model import rope_angles
import data as D; D.init_pipeline_cpu()
dev = "cuda"


class DecBlock(nn.Module):
    def __init__(self, d, heads, ffn_mult=4):
        super().__init__(); self.h, self.hd = heads, d // heads
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, q, kv, oq, okv, g, grouped):
        Tq = q.shape[0]; qh = self.q(self.nq(q)).view(Tq, self.h, self.hd)
        k, v = self.kv(self.nk(kv)).chunk(2, -1); Tk = kv.shape[0]
        k = k.view(Tk, self.h, self.hd); v = v.view(Tk, self.h, self.hd)
        if grouped:
            # grouped_cross does k==v; pass v via k slot (values), fine for timing
            o = grouped_cross(qh, v, oq, okv, g)
        else:
            o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])[0].transpose(0, 1)
        q = q + self.proj(o.reshape(Tq, -1)); return q + self.mlp(self.n2(q))


def timed(fn, iters=6, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / iters * 1e3, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", type=int, default=512); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--enc", type=int, default=12); ap.add_argument("--dec", type=int, default=4)
    ap.add_argument("--gp", type=int, default=1024); ap.add_argument("--gd", type=int, default=2048)
    a = ap.parse_args()
    B = D.get_cached(sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/fm_cache_tpc/ev_*.npz")))[3], device=dev)
    pa, ta, wa = B["plane_id"].to(dev), B["t_phys"].to(dev), B["wire_pos"].to(dev)
    Nf = len(pa); lam_t, lam_w = (8.0, 4336.0), (32.0, 2048.0); H, hd = a.heads, a.d // a.heads
    enc = nn.ModuleList([SBlock(a.d, a.heads) for _ in range(a.enc)]).to(dev)
    dec = nn.ModuleList([DecBlock(a.d, a.heads) for _ in range(a.dec)]).to(dev)

    def sched(pl, tp, wp, gp, gd):
        at = rope_angles(tp, hd, *lam_t); aw = rope_angles(wp, hd, *lam_w)
        o_pt = torch.argsort(pl.double() * 1e9 + tp.double()); o_pw = torch.argsort(pl.double() * 1e13 + wp.double() * 1e6 + tp.double())
        o_t = torch.argsort(tp.double()); o_ts = torch.roll(o_t, gd // 2)
        return ([(o_pt, gp, at, aw), (o_t, gd, at, None), (o_pw, gp, at, aw), (o_ts, gd, at, None)] * (a.enc // 4))

    for name, vis in [("random-0.75", torch.rand(Nf, device=dev) < 0.25), ("plane-mask(1)", pa != 0)]:
        msk = ~vis; iv = torch.where(vis)[0]; im = torch.where(msk)[0]
        pv, tv, wv = pa[iv], ta[iv], wa[iv]; tm = ta[im]
        Nv, Nm = len(iv), len(im)
        sc = sched(pv, tv, wv, a.gp, a.gd)
        atv = rope_angles(tv, hd, *lam_t); awv = rope_angles(wv, hd, *lam_w)
        oq = torch.argsort(tm.double()); okv = torch.argsort(tv.double())

        def run(grouped):
            xe = torch.randn(Nv, a.d, device=dev, requires_grad=True); h = xe
            for i, blk in enumerate(enc):
                if grouped: o, g, angt, angw = sc[i]; h = blk(h, angt, angw, o, g)
                else:       h = blk(h, atv, awv, None, 0)
            xq = torch.randn(Nm, a.d, device=dev, requires_grad=True); q = xq
            for blk in dec: q = blk(q, h, oq, okv, a.gd, grouped)
            (h.sum() + q.sum()).backward()

        def run_enc_only(grouped):                                    # to split enc vs dec
            xe = torch.randn(Nv, a.d, device=dev, requires_grad=True); h = xe
            for i, blk in enumerate(enc):
                if grouped: o, g, angt, angw = sc[i]; h = blk(h, angt, angw, o, g)
                else:       h = blk(h, atv, awv, None, 0)
            h.sum().backward()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            ff, mf = timed(lambda: run(False)); gg, mg = timed(lambda: run(True))
            fe, _ = timed(lambda: run_enc_only(False)); ge, _ = timed(lambda: run_enc_only(True))
        print(f"\n== {name}:  N_vis={Nv}  N_mask={Nm} ==")
        print(f"  {'':10s} {'total ms':>9} {'(enc':>7} {'dec)':>7} {'peak GB':>8}")
        print(f"  {'FULL':10s} {ff:>9.1f} {fe:>7.1f} {ff-fe:>7.1f} {mf:>8.2f}")
        print(f"  {'GROUPED':10s} {gg:>9.1f} {ge:>7.1f} {gg-ge:>7.1f} {mg:>8.2f}  -> {ff/gg:.2f}x total", flush=True)


if __name__ == "__main__":
    main()
