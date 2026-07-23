#!/usr/bin/env python
"""The simplified architecture (post step-back): ViT-style patch tokenizer.

  asinh values -> HYBRID patchify (column cells A10..D4 + per-band P=64
  windows for D3/D2) -> LINEAR patch embedding ([values, occupancy bits] ->
  d_model) + token-type/band embedding + physical-time PE
  -> optional transformer blocks (full attention within chunk)
  -> linear decode: d_model -> n_slot x (occupancy logit, clean value)

No deep encoder, no attention pooling, no tree ops — the tokenizer phase
showed a linear map reaches the per-patch floors. Tasks:
  ae  : denoising AE (acceptance test vs the deep-substrate hybrid: 0.434)
  mae : masked-token denoising (MAE-style trunk pilot — mask 30% of TOKENS,
        predict their clean slots from context; needs blocks > 0)

Run:  python vit_model.py --task ae --blocks 0 --steps 10000
"""
import sys, os, json, time, argparse, math

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import star_model as sm
from star_model import DEV, losses

sm.N_SLOT = 128
sm.Packer.assemble_fn = sm.assemble_hybrid
import onfly_optical

N_SLOT = 128
N_TYPES = sm.NBANDS + 1            # per-band token types + column type


class FullBlock(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, amask):
        Bn, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        sh = (Bn, T, self.h, d // self.h)
        q, k, v = (a.view(sh).transpose(1, 2) for a in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=amask)
        x = x + self.proj(o.transpose(1, 2).reshape(Bn, T, d))
        return x + self.mlp(self.n2(x))


class ViTTok(nn.Module):
    def __init__(self, d=256, blocks=0):
        super().__init__()
        self.d, self.n_blocks = d, blocks
        self.embed = nn.Linear(2 * N_SLOT, d)          # [values, occ bits]
        self.type_emb = nn.Embedding(N_TYPES, d)
        nf = d // 2
        self.register_buffer("freqs", torch.exp(
            torch.linspace(math.log(1.0), math.log(65536.0), nf)))
        self.pe_proj = nn.Linear(2 * nf, d)
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(FullBlock(d) for _ in range(blocks))
        self.dec = nn.Linear(d, 2 * N_SLOT)

    def pe(self, t):
        a = t[:, None] / self.freqs[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    def forward(self, B, tok_mask=None):
        x = self.embed(torch.cat([B["inp"], B["occ"]], -1))
        if tok_mask is not None:
            x = torch.where(tok_mask[:, None], self.mask_tok.expand_as(x), x)
        # token position: max tphys of the token's coefficients
        cell, n = B["cell"], B["n_cells"]
        cell_t = torch.zeros(n, device=x.device).index_reduce(
            0, cell, B["tphys"], "amax", include_self=True)
        x = x + self.pe_proj(self.pe(cell_t)) + self.type_emb(B["cell_band"])
        if self.n_blocks:
            x = self._attend(x, B["cell_chunk"])
        out = self.dec(x).view(n, N_SLOT, 2)
        return out[..., 0], out[..., 1]

    def _attend(self, x, grp):
        n_g = int(grp.max()) + 1
        counts = torch.bincount(grp, minlength=n_g)
        mx = int(counts.max())
        pad = torch.zeros(n_g, mx, x.shape[1], device=x.device)
        keep = torch.zeros(n_g, mx, dtype=torch.bool, device=x.device)
        order = torch.argsort(grp, stable=True)      # cells need not be group-contiguous
        gsort = grp[order]
        start = torch.searchsorted(gsort, torch.arange(n_g, device=grp.device))
        pos = torch.empty_like(grp)
        pos[order] = torch.arange(len(grp), device=grp.device) - start[gsort]
        pad[grp, pos] = x
        keep[grp, pos] = True
        amask = keep[:, None, None, :]
        for blk in self.blocks:
            pad = blk(pad, amask)
        return pad[grp, pos]


@torch.no_grad()
def evaluate(model, files, task, max_batches=60):
    model.eval()
    nb = sm.NBANDS
    se = np.zeros(nb); cnt = np.zeros(nb); base = np.zeros(nb)
    bce_t = []
    g = torch.Generator(device=DEV).manual_seed(7)
    for bi, B in enumerate(onfly_optical.OnflyPacker(files, seed=123)):
        if bi >= max_batches:
            break
        tok_mask = None
        if task == "mae":
            tok_mask = torch.rand(B["n_cells"], generator=g, device=DEV) < 0.3
        ol, vp = model(B, tok_mask)
        bce, _ = losses(ol, vp, B)
        bce_t.append(float(bce))
        err = ((vp[B["cell"], B["slot"]] - B["target"]) ** 2).cpu().numpy()
        bl = ((B["val"] - B["target"]) ** 2).cpu().numpy()
        band = B["band"].cpu().numpy()
        if task == "mae":                       # score masked tokens only
            mrow = tok_mask[B["cell"]].cpu().numpy()
            err, bl, band = err[mrow], bl[mrow], band[mrow]
        np.add.at(se, band, err); np.add.at(cnt, band, 1); np.add.at(base, band, bl)
    pb = se / np.maximum(cnt, 1)
    pbase = base / np.maximum(cnt, 1)
    model.train()
    return dict(primary=float(pb.mean()),
                per_band={str(b): float(pb[b]) for b in range(nb)},
                baseline_classical={str(b): float(pbase[b]) for b in range(nb)},
                baseline_primary=float(pbase.mean()), bce=float(np.mean(bce_t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ae", choices=["ae", "mae"])
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--budget", type=int, default=60000)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    files = onfly_optical.event_keys()
    test = [f for i, f in enumerate(files) if i % 3 == args.fold]
    train = [f for i, f in enumerate(files) if i % 3 != args.fold]
    if args.quick:
        train, test, args.steps = train[:6], test[:3], 50

    model = ViTTok(d=args.d, blocks=args.blocks).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"ViTTok task={args.task} blocks={args.blocks} d={args.d} "
          f"params={nparam/1e6:.2f}M train={len(train)} test={len(test)} steps={args.steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    step, t0 = 0, time.time()
    while step < args.steps:
        for B in onfly_optical.OnflyPacker(train, budget=args.budget,
                                           seed=args.seed + step):
            tok_mask = None
            if args.task == "mae":
                tok_mask = torch.rand(B["n_cells"], device=DEV) < 0.3
            ol, vp = model(B, tok_mask)
            bce, mse = losses(ol, vp, B)
            loss = bce + mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            if step % 500 == 0:
                print(f"  step {step}: bce {float(bce):.4f} mse {float(mse):.4f} "
                      f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)
            if step >= args.steps:
                break
    res = evaluate(model, test, args.task)
    res.update(model="vit", task=args.task, blocks=args.blocks, d=args.d,
               steps=args.steps, seed=args.seed, params=nparam,
               ms_per_step=(time.time() - t0) / max(step, 1) * 1000)
    print(json.dumps(res, indent=1))
    with open(os.path.join(sm.HERE, "artifacts", "star_results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
