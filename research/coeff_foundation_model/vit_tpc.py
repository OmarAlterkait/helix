#!/usr/bin/env python
"""TPC twin of the simplified ViT tokenizer (closes part one).

Per-band 2D patches (PW wires x PT band-ticks) in each band's native grid,
linear patch embedding ([noisy values, occupancy bits, dead bits] -> d_model)
+ band-type embedding + (time, wire, plane) PE -> optional within-plane
attention blocks -> linear decode (occupancy + clean value).

Input = production pipeline on the fly (star_tpc): densify -> GPU
coherent+intrinsic noise -> digitize -> coif3 L4 DWT -> smart removal kgate=4
-> per-band threshold; clean targets from the no-noise no-removal path.
Dead-wire robustness: random wire-kill augmentation (--deadfrac), dead bits
appended to each patch (sim has no dead channels, so this trains the
property; dead bit distinguishes 'dead' from 'thresholded-quiet').

Acceptance: beat the deep-substrate TPC AE (2.48) and the classical baseline
(0.56), mirroring the optical result.

Run:  python vit_tpc.py --task ae --blocks 0 --steps 10000
"""
import sys, os, json, time, argparse, math

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import star_tpc as stp
import star_model as sm
from star_model import DEV, losses
from vit_model import FullBlock

PW, PT = 16, 8                       # patch: wires x band-ticks
N_SLOT = PW * PT                     # 128
NB_T = stp.NB_T                      # 4
LENS_T = stp.LENS_T
DEAD_FRAC = 0.0                      # set by CLI for wire-kill augmentation


def assemble_tpc_band(ev, unit_ids, device=DEV):
    """Per-band 2D patch tokens for selected planes.

    device="cpu" keeps everything on host (for DataLoader worker assembly that
    is later moved to GPU in the main process); default DEV preserves old callers.
    """
    cs = ev["cstart"]
    rows = np.concatenate([np.arange(cs[c], cs[c + 1]) for c in unit_ids])
    band = ev["band"][rows]
    gid = ev["gid"][rows]
    wire = ev["wire"][rows]
    idx = ev["idx"][rows]
    tau = idx % LENS_T[band]
    val = ev["val"][rows]
    target = ev["val_clean"][rows]
    charge = ev["val_charge"][rows] if ev.get("val_charge") is not None else None

    wb, tb = wire // PW, tau // PT
    key = (gid.astype(np.int64) << 40) | (band.astype(np.int64) << 36) \
        | (wb.astype(np.int64) << 18) | tb
    uniq, cell = np.unique(key, return_inverse=True)
    n_cells = len(uniq)
    slot = (wire % PW) * PT + (tau % PT)
    cell_band = ((uniq >> 36) & 0xF).astype(np.int64)
    cell_gid = (uniq >> 40).astype(np.int64)
    cell_wb = ((uniq >> 18) & 0x3FFFF).astype(np.int64)
    cell_tb = (uniq & 0x3FFFF).astype(np.int64)

    occ = np.zeros((n_cells, N_SLOT), bool)
    inp = np.zeros((n_cells, N_SLOT), np.float32)
    tgt = np.zeros((n_cells, N_SLOT), np.float32)
    occ[cell, slot] = True
    inp[cell, slot] = np.arcsinh(val / sm.SIGMA)
    tgt[cell, slot] = np.arcsinh(target / sm.SIGMA)

    # valid slots: wire-in-block < plane n_wires, tick-in-block < band length
    nw = stp._pipeline()["nw"][cell_gid]
    Lb = LENS_T[cell_band]
    wi = np.arange(PW)[None, :]
    ti = np.arange(PT)[None, :]
    wok = (cell_wb[:, None] * PW + wi) < nw[:, None]           # (n_cells, PW)
    tok = (cell_tb[:, None] * PT + ti) < Lb[:, None]           # (n_cells, PT)
    valid = (wok[:, :, None] & tok[:, None, :]).reshape(n_cells, N_SLOT)

    # wire-kill augmentation: zero whole wire rows, flag as dead
    dead = np.zeros((n_cells, PW), bool)
    if DEAD_FRAC > 0:
        kill = np.random.random((n_cells, PW)) < DEAD_FRAC
        dead = kill & (cell_wb[:, None] * PW + wi < nw[:, None])
        ks = np.repeat(dead, PT, axis=1)                       # (n_cells, N_SLOT)
        inp[ks] = 0.0
        occ[ks] = False                                        # killed -> not active input
        # targets/valid unchanged: model must still predict, knowing wire is dead

    # RoPE time coord = canonical delay-corrected patch-CENTER DRIFT time.
    # Was np.maximum.at over survivor tphys, which is (a) band-width-biased (coarse patches
    # span 128 ticks vs 32 for fine -> max pushes coarse tokens ~100 ticks late, misaligning
    # bands), (b) occupancy-dependent (jumps with which coeffs survive), (c) zero-init clamps
    # negative A4 times to 0. And tphys=(tau+DELTA)*DEC carries per-plane TOFF, so "same drift
    # time" wasn't relative-0 across planes. Fix: grid-center time per (tb,band) minus per-plane
    # TOFF -> band-aligned, occupancy-independent, plane-aligned. (audit bugs 1/1b/3)
    _DEC = (1 << stp.LEV_T).astype(np.float32)              # [16,16,8,4] = 2^level
    _TOFF = np.array([-17.4, 2.6, 5.5], np.float32)         # U,V,Y sensor->drift (pb_labels.TOFF)
    _center_tau = cell_tb.astype(np.float32) * PT + PT / 2.0
    cell_t = ((_center_tau + stp.DELTA_T[cell_band]) * _DEC[cell_band]
              - _TOFF[cell_gid % 3]).astype(np.float32)     # plane orientation = gid%3 (U/V/Y)
    if os.environ.get("FM_CELLT") == "centroid":            # debiased survivor-max: amplitude-weighted
        # MEAN drift-time (not max -> no band-width bias), TOFF-corrected, no zero-clamp. Restores the
        # sub-patch occupancy timing that cross-plane triangulation uses (probed 3D 0.60 vs 0.42 canonical).
        _tp = ((tau.astype(np.float32) + stp.DELTA_T[band]) * _DEC[band] - _TOFF[gid % 3])
        _w = np.abs(val).astype(np.float32) + 1e-6
        _ws = np.zeros(n_cells, np.float32); _ts = np.zeros(n_cells, np.float32)
        np.add.at(_ws, cell, _w); np.add.at(_ts, cell, _w * _tp)
        cell_t = (_ts / np.maximum(_ws, 1e-6)).astype(np.float32)

    t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=device)
    out = dict(
        band=t(band.astype(np.int64), torch.long),
        val=t(np.arcsinh(val / sm.SIGMA), torch.float32),
        target=t(np.arcsinh(target / sm.SIGMA), torch.float32),
        cell=t(cell, torch.long), slot=t(slot, torch.long),
        n_cells=n_cells, occ=t(occ, torch.float32), inp=t(inp, torch.float32),
        tgt=t(tgt, torch.float32), valid=t(valid, torch.bool),
        dead=t(dead.astype(np.float32), torch.float32),
        cell_band=t(cell_band, torch.long), cell_gid=t(cell_gid, torch.long),
        cell_t=t(cell_t, torch.float32),
        cell_wire=t((cell_wb * PW).astype(np.float32), torch.float32))
    if charge is not None:                                # deconv-probe target (raw charge/row)
        out["target_charge"] = t(charge.astype(np.float32), torch.float32)
    return out


class TPCPacker:
    def __init__(self, events, budget=60000, seed=0, sort_size=False):
        self.events, self.budget = events, budget
        self.rng = np.random.default_rng(seed)
        self.sort_size = sort_size

    def __iter__(self):
        for ei in self.rng.permutation(len(self.events)):
            ev = stp.prep_tpc(self.events[ei])
            yield from sm.pack_event(ev, self.rng, self.budget, self.sort_size,
                                     assemble_fn=assemble_tpc_band)


class ViTTpc(nn.Module):
    def __init__(self, d=256, blocks=0):
        super().__init__()
        self.d, self.n_blocks = d, blocks
        self.embed = nn.Linear(3 * N_SLOT, d)          # [values, occ, dead-by-wire->slot]
        self.band_emb = nn.Embedding(NB_T, d)
        self.gid_emb = nn.Embedding(6, d)
        nf = d // 4
        self.register_buffer("tf", torch.exp(torch.linspace(math.log(1.), math.log(8192.), nf)))
        self.register_buffer("wf", torch.exp(torch.linspace(math.log(1.), math.log(2048.), nf)))
        self.pe = nn.Linear(4 * nf, d)
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(FullBlock(d) for _ in range(blocks))
        self.dec = nn.Linear(d, 2 * N_SLOT)

    def _pe(self, tt, ww):
        at = tt[:, None] / self.tf[None, :]
        aw = ww[:, None] / self.wf[None, :]
        return torch.cat([torch.sin(at), torch.cos(at), torch.sin(aw), torch.cos(aw)], -1)

    def forward(self, B, tok_mask=None):
        dead_slot = B["dead"].repeat_interleave(PT, dim=1)         # (n_cells, N_SLOT)
        x = self.embed(torch.cat([B["inp"], B["occ"], dead_slot], -1))
        if tok_mask is not None:
            x = torch.where(tok_mask[:, None], self.mask_tok.expand_as(x), x)
        x = x + self.pe(self._pe(B["cell_t"], B["cell_wire"])) \
            + self.band_emb(B["cell_band"]) + self.gid_emb(B["cell_gid"])
        if self.n_blocks:
            x = self._attend(x, B["cell_gid"])                     # within-plane
        out = self.dec(x).view(B["n_cells"], N_SLOT, 2)
        return out[..., 0], out[..., 1]

    def _attend(self, x, grp):
        n_g = int(grp.max()) + 1
        counts = torch.bincount(grp, minlength=n_g)
        mx = int(counts.max())
        pad = torch.zeros(n_g, mx, x.shape[1], device=x.device)
        keep = torch.zeros(n_g, mx, dtype=torch.bool, device=x.device)
        order = torch.argsort(grp, stable=True)
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
def evaluate(model, events, task, max_batches=40):
    model.eval()
    se = np.zeros(NB_T); cnt = np.zeros(NB_T); base = np.zeros(NB_T)
    bce_t = []
    g = torch.Generator(device=DEV).manual_seed(7)
    for bi, B in enumerate(TPCPacker(events, seed=123)):
        if bi >= max_batches:
            break
        tok_mask = (torch.rand(B["n_cells"], generator=g, device=DEV) < 0.3
                    if task == "mae" else None)
        ol, vp = model(B, tok_mask)
        bce, _ = losses(ol, vp, B)
        bce_t.append(float(bce))
        err = ((vp[B["cell"], B["slot"]] - B["target"]) ** 2).cpu().numpy()
        bl = ((B["val"] - B["target"]) ** 2).cpu().numpy()
        band = B["band"].cpu().numpy()
        if task == "mae":
            mrow = tok_mask[B["cell"]].cpu().numpy()
            err, bl, band = err[mrow], bl[mrow], band[mrow]
        np.add.at(se, band, err); np.add.at(cnt, band, 1); np.add.at(base, band, bl)
    pb = se / np.maximum(cnt, 1); pbase = base / np.maximum(cnt, 1)
    model.train()
    return dict(primary=float(pb.mean()),
                per_band={str(b): float(pb[b]) for b in range(NB_T)},
                baseline_classical={str(b): float(pbase[b]) for b in range(NB_T)},
                baseline_primary=float(pbase.mean()), bce=float(np.mean(bce_t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ae", choices=["ae", "mae"])
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--events", type=int, default=120)
    ap.add_argument("--budget", type=int, default=60000)
    ap.add_argument("--deadfrac", type=float, default=0.0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    global DEAD_FRAC
    DEAD_FRAC = args.deadfrac

    # plane n_wires lookup for the assemble validity mask
    P = stp._pipeline()
    P["nw"] = np.array([P["geom"][g]["n_wires"] for g in range(6)], np.int64)

    events = list(range(args.events))
    test = [e for i, e in enumerate(events) if i % 3 == args.fold]
    train = [e for i, e in enumerate(events) if i % 3 != args.fold]
    if args.quick:
        train, test, args.steps = train[:4], test[:2], 50

    model = ViTTpc(d=args.d, blocks=args.blocks).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"ViTTpc task={args.task} blocks={args.blocks} dead={args.deadfrac} "
          f"params={npar/1e6:.2f}M train={len(train)} test={len(test)} steps={args.steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    step, t0 = 0, time.time()
    while step < args.steps:
        for B in TPCPacker(train, budget=args.budget, seed=args.seed + step):
            tok_mask = (torch.rand(B["n_cells"], device=DEV) < 0.3
                        if args.task == "mae" else None)
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
    res.update(model="vit_tpc", task=args.task, blocks=args.blocks, d=args.d,
               deadfrac=args.deadfrac, steps=args.steps, params=npar,
               ms_per_step=(time.time() - t0) / max(step, 1) * 1000)
    print(json.dumps(res, indent=1))
    with open(os.path.join(sm.HERE, "artifacts", "star_results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
