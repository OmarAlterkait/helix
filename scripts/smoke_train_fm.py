#!/usr/bin/env python
"""A few real training steps on the coeff corpus — the "does the loop work" check.

Not a training run and not a benchmark. It answers exactly three questions, on
real corpus events with the real tokenizer and the real muP param groups:

  1. do the muP groups come out right?      hidden lr = base/m, hidden wd = wd*m
  2. does the loss actually go down?
  3. is the categorical head sane at init?  CE should start at ln(n_bins)

That last one is the useful sanity check: an untrained 128-bin head must sit at
ln(128) = 4.852, so a run that starts far from it has a broken target, not a
broken model.

The optimizer is built exactly as the research trainer built it
(``fm/mae_ddp.py``)::

    opt = AdamW(model.param_groups(lr, weight_decay=wd), lr=lr, betas=(0.9, 0.95))
    ratio = [pg["lr"] / lr for pg in opt.param_groups]     # muP per-group ratio
    ...     pg["lr"] = lr_at(step) * ratio[i]              # scheduler must preserve it

That last line is the subtlety any trainer integration has to reproduce: a
scheduler that sets one LR for all groups silently discards muP. pimm's
OneCycleLR takes a per-group ``max_lr`` list, which is how it is expressed there.

Usage::

    python scripts/smoke_train_fm.py [--events 4] [--steps 16] [--n-cells 6000]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

CORPUS = "/sdf/data/neutrino/omara/coeff_tpc/run_0027575715"
ARCHIVE = "/sdf/data/neutrino/omara/archive/fm_m113_converted.pt"

# Per-cell keys the dense (fused / categorical) loss path needs. The sparse
# cell/slot view is deliberately not carried: it indexes the per-event cell axis
# and would need rebasing under any batching. See MULTI_EVENT_BATCHING.md.
PER_CELL = ("inp", "occ", "valid", "tgt", "band_id", "plane_id", "t_phys",
            "wire_pos", "wirefeat")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--bins-from", default=ARCHIVE,
                    help="converted checkpoint to take categorical bin edges from")
    ap.add_argument("--events", type=int, default=4)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--n-cells", type=int, default=6000,
                    help="cells per step. losses_cat allocates an "
                         "(n_cells, n_slot, n_bins) intermediate, which is ~4 GiB "
                         "at a full event on the m113 config — cap it to fit.")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    a = ap.parse_args(argv)

    from pimm_data import CoeffTPCDataset
    from helix.model import build_fm
    from helix.model.tokenize import CoeffTokenize

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CoeffTPCDataset(data_root=a.corpus, dataset_name="sim_wire",
                         modalities=("coeff", "coeff_clean"), transform=None)
    tk = CoeffTokenize(part="coeff", clean_part="coeff_clean")

    def batch(i):
        B = tk(ds.get_data(i))["coeff"]
        return {k: torch.from_numpy(np.ascontiguousarray(B[k][:a.n_cells])).to(dev)
                for k in PER_CELL}

    torch.manual_seed(0)
    cfg = dict(n_slot=128, n_band=4, n_plane=6, d=a.d, blocks=a.blocks,
               dec_blocks=2, heads=8, n_bins=128, dec_mode="cross",
               mup=True, d_base=128)
    # rope_split is explicit for the same reason the configs pin it: it leaves no
    # trace in the weights, and defaulting it is what made m113 unevaluable.
    model = build_fm(cfg, serial=True, rope_split=False).to(dev)
    edges = torch.load(a.bins_from, map_location=dev,
                       weights_only=False)["bins"]["edges"]
    model.set_bins(edges)

    opt = torch.optim.AdamW(model.param_groups(a.lr, weight_decay=a.wd),
                            lr=a.lr, betas=(0.9, 0.95))
    ratio = [pg["lr"] / a.lr for pg in opt.param_groups]
    m_width = a.d // cfg["d_base"]
    print(f"muP width multiplier m = {a.d}/{cfg['d_base']} = {m_width}")
    for i, pg in enumerate(opt.param_groups):
        n = sum(p.numel() for p in pg["params"])
        print(f"  group {i}: {n/1e6:>6.2f}M  lr={pg['lr']:.2e}  ratio={ratio[i]:.2f}  "
              f"wd={pg.get('weight_decay')}")
    hidden = [i for i, r in enumerate(ratio) if abs(r - 1.0 / m_width) < 1e-9]
    assert hidden, f"no group scaled by 1/m={1/m_width}: muP is not active"

    print(f"\n{'step':>4} {'loss':>8} {'bce':>8} {'val':>8} {'|g|':>7}")
    hist = []
    for step in range(a.steps):
        B = batch(step % a.events)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            out = model(B)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(float(out["loss"]))
        if step % max(a.steps // 5, 1) == 0 or step == a.steps - 1:
            print(f"{step:>4} {float(out['loss']):>8.4f} {float(out['bce']):>8.4f} "
                  f"{float(out['val']):>8.4f} {float(gn):>7.2f}")

    k = max(a.steps // 4, 1)
    first, last = float(np.mean(hist[:k])), float(np.mean(hist[-k:]))
    print(f"\nloss {first:.4f} -> {last:.4f}   "
          f"{'DECREASING' if last < first else 'NOT DECREASING'}")
    print(f"ln(128) = {np.log(128):.4f}  (untrained categorical CE; the head should "
          f"start here)")
    return 0 if last < first else 1


if __name__ == "__main__":
    raise SystemExit(main())
