#!/usr/bin/env python3
"""T2d: where M7dc's sublinear gain response (dy ~ 0.67 dlog(1+e) for a GLOBAL scale; T2a) comes from.
Within tracks it transmits cell-to-cell fluctuations at slope 1.00 (T7), so the damping acts at a coarser scale.
Scale (1 + e), e = 0.05, only part of the input and measure the response of bright cells inside vs outside it:
  all      every coefficient (reference; repeats T2a)
  wstrip   wires with (wire // 64) % 2 == 0, every plane (cells in the middle 32 wires of a strip = inside,
           middle of an unscaled strip = control)
  tblock   coefficients whose drift tick (tick_of_tau) has (tick // 512) % 2 == 0 (middle 256 ticks = inside)
  plane<g> one plane gid g (cells of g = inside, the other gids = control), g = 0..5
  vol<v>   one TPC volume (gids 0-2 or 3-5)
  band<b>  one wavelet band b (A4, D4, D3, D2)
All in fp32. Prediction if the model normalises amplitude over a broad context: local slopes (wstrip, tblock)
~1 while 'all' is ~0.67; if the sublinearity is in the local response: local slopes ~0.67 too.

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/t2d_local.py --events 260-283 --out <dir>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
CKPT = f"{R}/denoise/M7dc_ft_full_dec_cond_ct25/final.pt"
EPS = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-283")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    a = ap.parse_args()
    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.model.tokenize import assemble, tick_of_tau, to_fm
    from helix.probe import resolution as R_
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta["head_kw"])
    model.load_state_dict(ck["model"]); model.to(dev).eval(); del ck
    ds = DenoiseEvents(CORPUS, [], CT, items=[]); cfg = ds.cfg
    os.makedirs(a.out, exist_ok=True)
    lo, hi = map(int, a.events.split("-"))
    for ev in range(lo, hi + 1):
        out = os.path.join(a.out, f"ev{ev:03d}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
        shard = str(z["shard"]); event = int(z["event"]); run = os.path.basename(os.path.dirname(shard))
        path = os.path.join(CORPUS, run, os.path.basename(shard))
        with h5py.File(path, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
            c = {k: fh["config"][k][:] for k in ("gids", "n_wires", "band_lengths", "norm_sigma")}
        tkeys, tq, tev = ds._truth(run, path, pos)
        assert tev == event
        sel = tq >= 16e3
        keys, Tq = tkeys[sel].astype(np.int64), tq[sel].astype(np.float64)
        ce = read_coeff_event(path, pos)
        tick = tick_of_tau(ce.tau.astype(np.float64), ce.plane_gid, np.minimum(ce.band, cfg.n_bands - 1), cfg)
        masks = {"base": np.zeros(len(ce.value), bool), "all": np.ones(len(ce.value), bool),
                 "wstrip": (ce.wire // 64) % 2 == 0, "tblock": (np.floor(tick).astype(np.int64) // 512) % 2 == 0}
        for gsel in range(6):
            masks[f"plane{gsel}"] = ce.plane_gid == gsel
        for v in range(2):
            masks[f"vol{v}"] = (ce.plane_gid // 3) == v
        for b in range(cfg.n_bands):
            masks[f"band{b}"] = ce.band == b
        res = dict(keys=keys, Tq=Tq)
        idx = aux = nbr = None
        for name, m in masks.items():
            v = ce.value.astype(np.float64) * np.where(m, 1 + EPS, 1.0)
            tok = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, v.astype(np.float32), gids=c["gids"], n_wires=c["n_wires"],
                           band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg)
            B = to_fm(tok)
            if idx is None:
                idx, aux = R_.cell_inputs(keys, B["cell_key"], c["band_lengths"], cfg)
                ok = (idx >= 0).any(1)
                nbr = R_.cell_neighbors(keys, B["cell_key"], c["band_lengths"], cfg)
                res["covered"] = ok
            Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
            Bt["n_cells"] = Bt["plane_id"].shape[0]
            y = np.zeros(len(keys), np.float32)
            with torch.no_grad():
                feats = model.fm.encode(Bt)
                s_all = np.nonzero(ok)[0]
                for s in range(0, len(s_all), 65536):
                    ss = s_all[s:s + 65536]
                    o = model.head(feats, torch.as_tensor(idx[ss]).to(dev), torch.as_tensor(aux[ss]).to(dev),
                                   nbr=torch.as_tensor(nbr[ss]).to(dev), B=Bt)
                    y[ss] = o[0].float().cpu().numpy()
            res[f"y_{name}"] = y
        np.savez(out + ".tmp.npz", **res)
        os.replace(out + ".tmp.npz", out)
        print(f"ev{ev:03d}: {len(keys)} bright cells, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
