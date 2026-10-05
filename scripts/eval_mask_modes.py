#!/usr/bin/env python3
"""Masked reconstruction of one checkpoint under several mask modes, per band.

    python scripts/eval_mask_modes.py --checkpoint <artifact> --tag T --out res.jsonl
        [--corpus C] [--split probe] [--events 60] [--modes random location]

Same events, same ratio (0.75), only the mask unit differs. Under ``random`` a
masked token's co-located tokens in the OTHER bands are each visible with
probability 0.25; under ``location`` (every band of a plane x 16-wire x 128-tick
location together) none is. For A4 the location IS its token's footprint, so
the A4 rows differ only in whether the other bands at the same place are seen:
the gap is how much of the reconstruction is cross-scale interpolation.

Per band and mode: explained variance of the decoded value (token space) and the
value cross-entropy, over occupied valid slots of masked tokens. Mask draws are
seeded per event, so two checkpoints see identical masks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, help="JSONL, appended")
    ap.add_argument("--corpus", default=None, help="default: HELIX_CORPUS")
    ap.add_argument("--split", default="probe")
    ap.add_argument("--events", type=int, default=60)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--modes", nargs="+", default=["random", "location"])
    a = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.model.artifact import build, load
    from helix.model.loss import bucketize_bins
    from helix.model.mask import make_mask
    from helix.model.tokenize import PatchConfig, assemble, to_fm
    from helix.paths import root

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = a.corpus or str(root("HELIX_CORPUS"))
    art = load(a.checkpoint)
    model = build(art, device=dev, eval_mode=True)
    pw, pt, nb = art.op.pw or 16, art.op.pt or 8, art.op.n_bands or 4
    cfg = PatchConfig(cell_t="grid_center", pw=pw, pt=pt, n_bands=nb)
    cell = (16, 128)                                   # the same physical unit for every pw
    events = json.load(open(os.path.join(corpus, "holdout.json")))[a.split][:a.events]

    acc = {m: np.zeros((nb, 5)) for m in a.modes}      # sse, sum y, sum y^2, n, ce
    for i, ent in enumerate(events):
        tag = ent["source_file"].replace("sim_wire_sensor_", "").replace(".h5", "")
        shard = os.path.join(corpus, f"sim_wire_coeff_{tag}.h5")
        with h5py.File(shard, "r") as f:
            ids = f["ident"]["event"][:]
            gids, nw, bl, ns = (f["config"][x][:] for x in ("gids", "n_wires", "band_lengths", "norm_sigma"))
        pos = int(np.searchsorted(ids, ent["event"]))
        ce = read_coeff_event(shard, pos)
        cl = read_coeff_event(shard.replace("sim_wire_coeff_", "sim_wire_coeff_clean_"), pos, coords_from=ce)
        tok = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, gids=gids, n_wires=nw,
                       band_lengths=bl, norm_sigma=ns, cfg=cfg, value_clean=cl.value)
        B = {k: torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v for k, v in to_fm(tok).items()}
        B["n_cells"] = B["plane_id"].shape[0]
        for mode in a.modes:
            g = torch.Generator(device=dev).manual_seed(1000 + i)
            m = make_mask(B, mode, a.ratio, 1, gen=g, cell=cell)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                feat, rows = model.forward_feat(B, m, masked_only=True)
            NS, K = model.n_slot, model.n_bins
            for s in range(0, rows.numel(), 2048):
                r, f_ = rows[s:s + 2048], feat[s:s + 2048].float()
                with torch.no_grad():
                    lp = torch.log_softmax(model.val_head(f_.to(model.val_head.weight.dtype)).float()
                                           .view(-1, NS, K) * model.readout_mult, -1)
                occ, valid, tgt, band = B["occ"][r].bool(), B["valid"][r].bool(), B["tgt"][r].float(), B["band_id"][r].long()
                rec = (lp.exp() * model.bin_cent_asinh[band][:, None, :].float()).sum(-1)
                ce_ = -lp.gather(-1, bucketize_bins(tgt, band, model.bin_edges, K)[..., None]).squeeze(-1)
                sel = occ & valid
                for b in range(nb):
                    sb = sel & (band == b)[:, None]
                    y = tgt[sb]; d_ = rec[sb] - y
                    acc[mode][b] += [float((d_ * d_).sum()), float(y.sum()), float((y * y).sum()),
                                     float(sb.sum()), float(ce_[sb].sum())]
        if i % 10 == 0:
            print(f"  event {i}/{len(events)}", flush=True)

    names = ["A4", "D4", "D3", "D2"][:nb]
    res = {"tag": a.tag, "checkpoint": a.checkpoint, "events": len(events), "ratio": a.ratio, "cell": cell}
    for mode, A in acc.items():
        out = {}
        for b, nm in enumerate(names):
            sse, sy, syy, n, ce = A[b]
            var = syy / max(n, 1) - (sy / max(n, 1)) ** 2
            out[nm] = {"var_expl": round(1 - sse / max(n, 1) / max(var, 1e-12), 4), "ce": round(ce / max(n, 1), 4), "n": int(n)}
        sse, sy, syy, n, ce = A.sum(0)
        var = syy / n - (sy / n) ** 2
        out["all"] = {"var_expl": round(1 - sse / n / var, 4), "ce": round(ce / n, 4), "n": int(n)}
        res[mode] = out
        print(mode, json.dumps(out), flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
