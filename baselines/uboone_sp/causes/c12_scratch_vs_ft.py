#!/usr/bin/env python3
"""C12: does PRETRAINING set the high-charge floor? Matched pairs, same recipe, pretrained-then-fine-tuned vs from
scratch: M4n_ft vs M2n_scratch (full scale, CellHead), le3_ft_N16384 vs le3_scratch_N16384 (16k labels, CellHead).
Bright cells (training-label complete truth) on 24 test events, fp32: hw68 / median of y - log1p(T/Q0) per plane
and truth bin. Scratch ~= ft at high charge => the pretrained start does not set the floor.

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/c12_scratch_vs_ft.py --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
W = "/pscratch/sd/o/oalter/helix_work/denoise"
CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
Q0 = 2043.2578108909702
MODELS = {"M4n_ft (full)": f"{W}/M4n_ft_full_near/final.pt", "M2n_scratch (full)": f"{W}/M2n_scratch_full_near/final.pt",
          "le3_ft_N16384": f"{W}/le3_ft_N16384/final.pt", "le3_scratch_N16384": f"{W}/le3_scratch_N16384/final.pt"}
BINS = [(4e3, 16e3), (16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-283")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import CellDecoder, build_denoise
    from helix.model.tokenize import assemble, to_fm
    from helix.probe import resolution as R_
    dev = torch.device("cuda")
    ds = DenoiseEvents(CORPUS, [], CT, items=[]); cfg = ds.cfg
    evd = []
    lo, hi = map(int, a.events.split("-"))
    for ev in range(lo, hi + 1):
        z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
        run = os.path.basename(os.path.dirname(str(z["shard"])))
        path = os.path.join(CORPUS, run, os.path.basename(str(z["shard"])))
        with h5py.File(path, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], int(z["event"])))
            c = {k: fh["config"][k][:] for k in ("gids", "n_wires", "band_lengths", "norm_sigma")}
        tk, tq, _ = ds._truth(run, path, pos)
        s = tq >= 4e3
        ce = read_coeff_event(path, pos)
        B = to_fm(assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, gids=c["gids"], n_wires=c["n_wires"],
                           band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg))
        evd.append((B, c, tk[s].astype(np.int64), tq[s]))
    out = {}
    for name, ckpt in MODELS.items():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
        model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta.get("head_kw") or {})
        model.load_state_dict(ck["model"]); model.to(dev).eval(); del ck
        decoder = isinstance(model.head, CellDecoder)
        E, P, T = [], [], []
        for B, c, keys, tq in evd:
            idx, aux = R_.cell_inputs(keys, B["cell_key"], c["band_lengths"], cfg)
            ok = (idx >= 0).any(1)
            Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
            Bt["n_cells"] = Bt["plane_id"].shape[0]
            with torch.no_grad():
                feats = model.fm.encode(Bt)
                ti, ta = torch.as_tensor(idx[ok]).to(dev), torch.as_tensor(aux[ok]).to(dev)
                o = model.head(feats, ti, ta, nbr=torch.as_tensor(R_.cell_neighbors(keys[ok], B["cell_key"], c["band_lengths"], cfg)).to(dev),
                               B=Bt) if decoder else model.head(feats, ti, ta)
                o = o[0] if isinstance(o, tuple) else o
            g, _, _ = R_.unkey(keys[ok])
            E.append(o.float().cpu().numpy() - np.log1p(tq[ok] / Q0)); P.append(g % 3); T.append(tq[ok])
        E, P, T = map(np.concatenate, (E, P, T))
        r = {}
        for pl in range(3):
            for lo_, hi_ in BINS:
                m = (P == pl) & (T >= lo_) & (T < hi_)
                a16, a50, a84 = np.percentile(E[m], [16, 50, 84])
                r[f"{'UVY'[pl]}_{lo_:.0f}"] = dict(hw68=float((a84 - a16) / 2), med=float(a50), n=int(m.sum()))
        out[name] = r
        print(name, {k: round(v["hw68"], 4) for k, v in r.items()}, flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
