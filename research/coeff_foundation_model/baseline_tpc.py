#!/usr/bin/env python
"""Simplest baseline (SIMPLEST_BASELINE.md): masked CROSS-PLANE coefficient AE,
RoPE on physical time (+ wire), plain full attention over a whole event.

Per event: all 6 planes' per-band 2D patch tokens go in ONE token set ->
full self-attention (cross-plane enabled). Mask a fraction (random + whole-
plane curriculum); reconstruct masked tokens' CLEAN coefficients from visible.
Positional info = RoPE(physical_time) on all tokens + RoPE(wire) on TPC;
learned plane/level embeddings added. No hierarchy, no bias, no pooling.

Input = production pipeline on the fly (star_tpc). Run:
  python baseline_tpc.py --blocks 4 --steps 2000
"""
import sys, os, json, time, argparse, math

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import hdf5plugin  # noqa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import star_tpc as stp
import vit_tpc as vtp
import star_model as sm
from star_model import DEV

N_SLOT = vtp.N_SLOT          # 128 (16 wires x 8 band-ticks)
NB_T = stp.NB_T


def rope_angles(pos, dim, base=10000.0):
    """pos: (T,) float -> (T, dim) interleaved cos/sin angles for RoPE on `dim` dims."""
    half = dim // 2
    inv = base ** (-torch.arange(0, half, 2, device=pos.device).float() / half)  # (half/2,)
    ang = pos[:, None] * inv[None, :]                          # (T, half/2)
    return ang


def apply_rope(x, ang_t, ang_w):
    """x: (T, H, hd). First half of hd rotated by time, second half by wire."""
    T, H, hd = x.shape
    h2 = hd // 2
    def rot(v, ang):                                           # v: (T,H,h2), ang:(T,h2/2)
        c = torch.cos(ang)[:, None, :].repeat_interleave(2, -1)
        s = torch.sin(ang)[:, None, :].repeat_interleave(2, -1)
        v2 = torch.stack([-v[..., 1::2], v[..., 0::2]], -1).reshape_as(v)
        return v * c + v2 * s
    xt = rot(x[..., :h2], ang_t)
    xw = rot(x[..., h2:], ang_w) if ang_w is not None else x[..., h2:]
    return torch.cat([xt, xw], -1)


class RoPEBlock(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, ang_t, ang_w):
        T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        q = apply_rope(q.view(T, self.h, self.hd), ang_t, ang_w)
        k = apply_rope(k.view(T, self.h, self.hd), ang_t, ang_w)
        v = v.view(T, self.h, self.hd)
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                           v.transpose(0, 1)[None])[0].transpose(0, 1)
        x = x + self.proj(o.reshape(T, d))
        return x + self.mlp(self.n2(x))


class BaselineViT(nn.Module):
    def __init__(self, d=192, blocks=4):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(2 * N_SLOT, d)
        self.band_emb = nn.Embedding(NB_T, d)
        self.gid_emb = nn.Embedding(6, d)
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(RoPEBlock(d) for _ in range(blocks))
        self.dec = nn.Linear(d, 2 * N_SLOT)

    def forward(self, B, tok_mask):
        x = self.embed(torch.cat([B["inp"], B["occ"]], -1))
        x = torch.where(tok_mask[:, None], self.mask_tok.expand_as(x), x)   # MAE: hide masked
        x = x + self.band_emb(B["cell_band"]) + self.gid_emb(B["cell_gid"])
        hd = self.d // 4
        ang_t = rope_angles(B["cell_t"], hd)            # physical-time RoPE (all)
        ang_w = rope_angles(B["cell_wire"], hd)         # wire RoPE (TPC)
        for blk in self.blocks:
            x = blk(x, ang_t, ang_w)
        out = self.dec(x).view(-1, N_SLOT, 2)
        return out[..., 0], out[..., 1]


def event_batch(ev_idx, cap=30000):
    """One event = all planes -> one token set (cross-plane)."""
    ev = stp.prep_tpc(ev_idx)
    B = vtp.assemble_tpc_band(ev, list(range(ev["n_chunks"])))
    if B["n_cells"] > cap:                              # pilot: subsample tokens
        keep = torch.randperm(B["n_cells"], device=DEV)[:cap].sort().values
        remap = -torch.ones(B["n_cells"], dtype=torch.long, device=DEV)
        remap[keep] = torch.arange(cap, device=DEV)
        rowkeep = remap[B["cell"]] >= 0
        for k in ("inp", "occ", "tgt", "valid", "cell_band", "cell_gid", "cell_t", "cell_wire"):
            B[k] = B[k][keep]
        for k in ("band", "val", "target", "cell", "slot"):
            B[k] = B[k][rowkeep]
        B["cell"] = remap[B["cell"]]
        B["n_cells"] = cap
    return B


def make_mask(B, frac=0.5, p_plane=0.4, gen=None):
    n = B["n_cells"]
    m = (torch.rand(n, generator=gen, device=DEV) < frac)
    if gen is None and torch.rand(1, device=DEV).item() < p_plane:
        g = int(torch.randint(0, 6, (1,), device=DEV))
        m = m | (B["cell_gid"] == g)                    # whole-plane curriculum
    return m


def loss_on_masked(ol, vp, B, tok_mask):
    mrow = tok_mask[B["cell"]]
    occm = tok_mask[:, None] & B["valid"]
    bce = F.binary_cross_entropy_with_logits(ol[occm], B["occ"][occm]) if occm.any() else ol.sum()*0
    act = mrow & (B["occ"][B["cell"], B["slot"]] > 0 if False else True)
    actm = tok_mask[B["cell"]]
    err = vp[B["cell"], B["slot"]][actm]
    tgt = B["target"][actm]
    mse = F.mse_loss(err, tgt) if actm.any() else ol.sum() * 0
    return bce, mse


@torch.no_grad()
def evaluate(model, events, frac=0.5):
    model.eval()
    se = np.zeros(NB_T); cnt = np.zeros(NB_T); base = np.zeros(NB_T)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for ei in events:
        B = event_batch(ei)
        m = make_mask(B, frac, p_plane=0.0, gen=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ol, vp = model(B, m)
        mrow = m[B["cell"]].cpu().numpy()
        err = ((vp[B["cell"], B["slot"]] - B["target"]) ** 2).cpu().numpy()[mrow]
        bl = ((B["val"] - B["target"]) ** 2).cpu().numpy()[mrow]
        band = B["band"].cpu().numpy()[mrow]
        np.add.at(se, band, err); np.add.at(cnt, band, 1); np.add.at(base, band, bl)
    pb = se / np.maximum(cnt, 1); pbase = base / np.maximum(cnt, 1)
    model.train()
    return dict(primary=float(pb.mean()), per_band={str(b): float(pb[b]) for b in range(NB_T)},
                baseline_primary=float(pbase.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--events", type=int, default=120)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    P = stp._pipeline(); P["nw"] = np.array([P["geom"][g]["n_wires"] for g in range(6)], np.int64)

    evs = list(range(args.events))
    test = [e for i, e in enumerate(evs) if i % 3 == args.fold]
    train = [e for i, e in enumerate(evs) if i % 3 != args.fold]
    if args.quick:
        train, test, args.steps = train[:4], test[:2], 30

    model = BaselineViT(d=args.d, blocks=args.blocks).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"baseline_tpc d={args.d} blocks={args.blocks} params={npar/1e6:.2f}M "
          f"train={len(train)} steps={args.steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    step, t0 = 0, time.time()
    while step < args.steps:
        for ei in np.random.permutation(train):
            B = event_batch(int(ei))
            m = make_mask(B, args.frac)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ol, vp = model(B, m)
                bce, mse = loss_on_masked(ol, vp, B, m)
            loss = bce + mse
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 200 == 0:
                print(f"  step {step}: bce {float(bce):.4f} mse {float(mse):.4f} "
                      f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)
            if step >= args.steps:
                break
    res = evaluate(model, test, args.frac)
    res.update(model="baseline_tpc", blocks=args.blocks, d=args.d, steps=args.steps,
               frac=args.frac, params=npar, ms_per_step=(time.time()-t0)/max(step,1)*1000)
    print(json.dumps(res, indent=1))
    with open(os.path.join(sm.HERE, "artifacts", "star_results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
