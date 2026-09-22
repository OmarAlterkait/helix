"""P4 — interventions that raise the per-GPU token ceiling, measured against
the production step, with equivalence checks.

  base            production forward (categorical K=128, dense CE over the grid)
  head_chunk      val_head + CE computed in cell chunks under checkpointing
  head_active     val_head + CE computed ONLY at (cell, slot) pairs the loss
                  actually weights (occ & valid & masked) -- numerically the
                  same sum, ~1/14 the elements
  blk_ckpt        activation checkpointing on the 12 encoder + 4 decoder blocks
  active+ckpt     both
  compiled        torch.compile(model)
"""
from __future__ import annotations
import argparse, gc, json, os, sys, traceback
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, peak_mem, synth_from, timeit, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--iters", type=int, default=8)
ap.add_argument("--ceiling", action="store_true")
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")

evs, _ = load_events(4)
src = {k: v for k, v in evs[0].items() if torch.is_tensor(v)}
model = build(device=dev)
K = model.n_bins
NS = model.n_slot

# ---------------------------------------------------------------- variants
from helix.model.loss import bucketize_bins, losses_cat


def _bce(occ, B, mrow):
    m_occ = mrow & B["valid"]
    e = F.binary_cross_entropy_with_logits(occ, B["occ"], reduction="none")
    return (e * m_occ).sum() / m_occ.sum().clamp(min=1)


def head_loss_base(model, feat, B, tok_mask):
    occ = model.occ_head(feat) * model.readout_mult
    val = (model.val_head(feat) * model.readout_mult).view(-1, NS, K)
    return losses_cat(occ, val, B, tok_mask, model.bin_edges, vis_w=0.0)


def head_loss_chunk(model, feat, B, tok_mask, chunk=4096):
    """Same arithmetic, cell-chunked, logits recomputed in backward."""
    occ = model.occ_head(feat) * model.readout_mult
    mrow = tok_mask[:, None]
    bce = _bce(occ, B, mrow)
    act = B["occ"].bool() & B["valid"] & mrow
    denom = act.sum().clamp(min=1).float()
    binid = bucketize_bins(B["tgt"], B["band_id"], model.bin_edges, K)
    tot = feat.new_zeros((), dtype=torch.float32)
    N = feat.shape[0]

    def part(f, bi, ac):
        lg = (model.val_head(f) * model.readout_mult).view(-1, NS, K)
        ce = F.cross_entropy(lg.reshape(-1, K), bi.reshape(-1), reduction="none").view_as(bi)
        return (ce * ac).sum()

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        ac = act[s:e]
        if not bool(ac.any()):
            continue
        tot = tot + checkpoint(part, feat[s:e], binid[s:e], ac, use_reentrant=False)
    return bce, tot / denom


def _active_pairs(model, feat, B, tok_mask):
    mrow = tok_mask[:, None]
    act = B["occ"].bool() & B["valid"] & mrow
    ci, si = act.nonzero(as_tuple=True)
    binid = bucketize_bins(B["tgt"][ci, si][:, None], B["band_id"][ci],
                           model.bin_edges, K).squeeze(1)
    return act, ci, si, binid


def head_loss_active(model, feat, B, tok_mask):
    """Slot-grouped loop: one small GEMM per slot, no padding."""
    occ = model.occ_head(feat) * model.readout_mult
    bce = _bce(occ, B, tok_mask[:, None])
    act, ci, si, binid = _active_pairs(model, feat, B, tok_mask)
    P = ci.numel()
    if P == 0:
        return bce, feat.sum() * 0
    W = model.val_head.weight.view(NS, K, -1)
    bv = model.val_head.bias.view(NS, K)
    order = torch.argsort(si)
    ci_s, bin_s = ci[order], binid[order]
    bl = torch.cat([torch.zeros(1, dtype=torch.long, device=feat.device),
                    torch.bincount(si[order], minlength=NS).cumsum(0)]).tolist()
    tot = feat.new_zeros((), dtype=torch.float32)
    for s in range(NS):
        a, b = bl[s], bl[s + 1]
        if b <= a:
            continue
        lg = F.linear(feat[ci_s[a:b]], W[s], bv[s]) * model.readout_mult
        tot = tot + F.cross_entropy(lg.float(), bin_s[a:b], reduction="sum")
    return bce, tot / float(P)


def head_loss_active_bmm(model, feat, B, tok_mask):
    """Same pairs, one padded bmm + one cross_entropy: 4 kernels, not 4*n_slot."""
    occ = model.occ_head(feat) * model.readout_mult
    bce = _bce(occ, B, tok_mask[:, None])
    act, ci, si, binid = _active_pairs(model, feat, B, tok_mask)
    P = ci.numel()
    if P == 0:
        return bce, feat.sum() * 0
    order = torch.argsort(si)
    ci_s, si_s, bin_s = ci[order], si[order], binid[order]
    cnt = torch.bincount(si_s, minlength=NS)
    mx = int(cnt.max())
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=feat.device),
                        cnt.cumsum(0)[:-1]])
    col = torch.arange(mx, device=feat.device)[None, :]
    valid = col < cnt[:, None]                            # (NS, mx)
    flat = (starts[:, None] + col).clamp(max=max(P - 1, 0))
    idx = ci_s[flat]                                      # (NS, mx) cell per slot-slot
    f = feat[idx]                                         # (NS, mx, d)
    W = model.val_head.weight.view(NS, K, -1)
    bv = model.val_head.bias.view(NS, K)
    lg = (torch.baddbmm(bv[:, None, :].to(f.dtype), f, W.transpose(1, 2).to(f.dtype))
          * model.readout_mult)                           # (NS, mx, K)
    tgtb = bin_s[flat]
    ce = F.cross_entropy(lg.reshape(-1, K).float(), tgtb.reshape(-1), reduction="none")
    tot = (ce.view(NS, mx) * valid).sum()
    return bce, tot / float(P)


def run_fwd(model, B, m, head, blk_ckpt=False):
    if blk_ckpt:
        feat = forward_feat_ckpt(model, B, m)
    else:
        feat = model.forward_feat(B, m)
    bce, val = head(model, feat, B, m)
    return bce + val


# ------------------------------------- forward_feat with per-block checkpointing
from helix.model.fm import rope_angles
from helix.model.serial import _cross, _self


def forward_feat_ckpt(model, B, tok_mask):
    """SerialFMModel.forward_feat with each block wrapped in checkpoint().

    Mirrors serial.py exactly; the only delta is the checkpoint() wrapper, so a
    drift in serial.py shows up as a numerical mismatch in the equivalence check
    below rather than silently.
    """
    N = B["inp"].shape[0]
    at = rope_angles(B["t_phys"], model.d // model.heads, *model.lam_t)
    aw = rope_angles(B["wire_pos"], model.d // model.heads, *model.lam_w)
    vis = ~tok_mask
    vis_idx = vis.nonzero(as_tuple=True)[0]
    mask_idx = tok_mask.nonzero(as_tuple=True)[0]
    xv = model._emb(B, vis_idx); atv, awv = at[vis], aw[vis]
    sched = model._sched(B["plane_id"][vis], B["t_phys"][vis], B["wire_pos"][vis])
    c = model._cond(B) if model.cond == "adaln" else None
    cv = c[vis] if c is not None else None
    for blk, (o, g, uw) in zip(model.enc, sched):
        xv = checkpoint(_self, blk, xv, atv, awv if uw else None, o, g, cv,
                        use_reentrant=False)
    qm = model.mask_tok.expand(mask_idx.numel(), model.d)
    if c is not None:
        qm = qm.to(xv.dtype)
    else:
        if model.film is not None:
            g_, b_ = model.film(B["band_id"][tok_mask], B["plane_id"][tok_mask],
                                B["wirefeat"][tok_mask])
            qm = g_ * qm + b_
        qm = (qm + model.band_emb(B["band_id"][tok_mask])
              + model.plane_emb(B["plane_id"][tok_mask])).to(xv.dtype)
    atm, awm = at[tok_mask], aw[tok_mask]
    oq = torch.argsort(B["t_phys"][tok_mask].double())
    okv = torch.argsort(B["t_phys"][vis].double())
    cm = c[tok_mask] if c is not None else None
    for blk in model.dec:
        qm = checkpoint(_cross, blk, qm, xv, atm, awm, atv, awv, oq, okv, model.gd, cm,
                        use_reentrant=False)
    x = torch.zeros(N, model.d, dtype=xv.dtype, device=xv.device)
    x = x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm)
    return model.dec_norm(x)


# ---------------------------------------------------------------- measure
R = {"variants": {}, "chunk": A.chunk}
B = to_device(src, dev)
B["n_cells"] = B["plane_id"].shape[0]
N0 = B["n_cells"]
gen = torch.Generator(device=dev); gen.manual_seed(0)
m = model.make_mask(B, mode="random", gen=gen)
print(f"event n_cells={N0} masked={float(m.float().mean()):.3f}")
act = (B["occ"].bool() & B["valid"] & m[:, None])
R["grid"] = dict(n_cells=N0, grid_cells=N0 * NS, active_pairs=int(act.sum()),
                 active_frac=float(act.float().mean()),
                 logit_elems_dense=N0 * NS * K, logit_elems_active=int(act.sum()) * K,
                 saving=N0 * NS / max(int(act.sum()), 1))
print(json.dumps(R["grid"], indent=1))

opt = torch.optim.AdamW(model.param_groups(1.1e-3, weight_decay=0.05), betas=(0.9, 0.95))

VAR = {
    "base":         dict(head=head_loss_base, blk_ckpt=False),
    "head_chunk":   dict(head=lambda mo, f, b, t: head_loss_chunk(mo, f, b, t, A.chunk), blk_ckpt=False),
    "head_active":  dict(head=head_loss_active, blk_ckpt=False),
    "head_bmm":     dict(head=head_loss_active_bmm, blk_ckpt=False),
    "blk_ckpt":     dict(head=head_loss_base, blk_ckpt=True),
    "bmm+ckpt":     dict(head=head_loss_active_bmm, blk_ckpt=True),
    "chunk+ckpt":   dict(head=lambda mo, f, b, t: head_loss_chunk(mo, f, b, t, A.chunk), blk_ckpt=True),
}

ref_loss = ref_grad = None
for name, v in VAR.items():
    try:
        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = run_fwd(model, B, m, v["head"], v["blk_ckpt"])
            loss.backward()
            return loss
        l = step()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
        if ref_loss is None:
            ref_loss, ref_grad = float(l), gn
        t = timeit(lambda: step(), warmup=2, iters=A.iters)
        with peak_mem() as pm:
            step()
        opt.zero_grad(set_to_none=True)
        row = dict(ms=t["ms_med"], peak_MiB=pm["peak_alloc_MiB"],
                   loss=float(l), grad_norm=gn,
                   d_loss=float(l) - ref_loss, d_grad_rel=(gn - ref_grad) / max(ref_grad, 1e-9),
                   MiB_per_cell=pm["peak_alloc_MiB"] / N0,
                   speed_vs_base=None, mem_vs_base=None)
        R["variants"][name] = row
        print(f"  {name:14s} {row['ms']:8.2f} ms  peak {row['peak_MiB']:9.1f} MiB  "
              f"loss {row['loss']:.6f} (d={row['d_loss']:+.2e})  |g| {gn:.5f} "
              f"(rel d={row['d_grad_rel']:+.2e})", flush=True)
    except Exception as e:
        R["variants"][name] = dict(error=repr(e), tb=traceback.format_exc()[-800:])
        print(f"  {name:14s} FAILED {e}", flush=True)
    gc.collect(); torch.cuda.empty_cache()

b = R["variants"].get("base", {})
for name, row in R["variants"].items():
    if "ms" in row and "ms" in b:
        row["speed_vs_base"] = b["ms"] / row["ms"]
        row["mem_vs_base"] = row["peak_MiB"] / b["peak_MiB"]

# ------------------------------------------------------------ torch.compile
print("== torch.compile ==", flush=True)
try:
    cm = torch.compile(model, dynamic=True)
    def cstep():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            o = cm(B, tok_mask=m)
        o["loss"].backward()
    t = timeit(cstep, warmup=5, iters=A.iters)
    with peak_mem() as pm:
        cstep()
    R["variants"]["compiled"] = dict(ms=t["ms_med"], peak_MiB=pm["peak_alloc_MiB"],
                                     speed_vs_base=b.get("ms", float("nan")) / t["ms_med"])
    print(f"  compiled {t['ms_med']:.2f} ms peak {pm['peak_alloc_MiB']:.1f} MiB")
except Exception as e:
    R["variants"]["compiled"] = dict(error=repr(e), tb=traceback.format_exc()[-1500:])
    print("  compile FAILED", e)
opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()

# ------------------------------------------------------- ceiling per variant
if A.ceiling:
    print("== token ceiling per variant ==", flush=True)
    ceil = {}
    for name in ("base", "head_bmm", "bmm+ckpt", "chunk+ckpt"):
        v = VAR[name]
        lo, hi, best = 8000, 600000, 0
        while hi - lo > 8000:
            mid = (lo + hi) // 2
            try:
                Bs = to_device(synth_from(src, mid), dev)
                Bs["n_cells"] = mid
                g2 = torch.Generator(device=dev); g2.manual_seed(0)
                ms = model.make_mask(Bs, mode="random", gen=g2)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", torch.bfloat16):
                    loss = run_fwd(model, Bs, ms, v["head"], v["blk_ckpt"])
                loss.backward()
                torch.cuda.synchronize()
                best = mid; lo = mid
            except torch.cuda.OutOfMemoryError:
                hi = mid
            finally:
                Bs = ms = loss = None
                opt.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache()
        ceil[name] = dict(max_cells=best, events=best / 32500.0)
        print(f"  {name:14s} max_cells={best:7d}  ~{best/32500.0:.1f} events", flush=True)
    R["ceiling"] = ceil

emit("p4_interventions", R)
