#!/usr/bin/env python3
"""C14: how much of the bright cells' input lies in the pretrained FM's OPEN-ENDED value bins?

The FM's 128 categorical bins (uniform in arcsinh(v / sigma)) end at a finite top edge per band, followed by an
edge at +-1e18: the masked objective cannot tell apart any two values beyond it. Per plane, truth bin and band, on
the 48 floor test events (training-label complete truth): the fraction of the covering tokens' occupied slots
whose |value| lies beyond the last finite edge, and the fraction of cells whose covering token has any such slot.

    scripts/helix_run.sh python baselines/uboone_sp/causes/c14_overflow.py --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
ART = "/pscratch/sd/o/oalter/helix_work/probe/artifact_nz_pw16_s1"
BINS = [(4e3, 16e3), (16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    from helix.data.denoise import DenoiseEvents
    from helix.model.artifact import load
    from helix.probe import resolution as R_
    E = np.asarray(load(ART).state_dict["bin_edges"], np.float64)
    E = E if E.ndim == 2 else E[None]
    top = np.array([e[np.isfinite(e) & (np.abs(e) < 1e17)].max() for e in E])
    bot = np.array([e[np.isfinite(e) & (np.abs(e) < 1e17)].min() for e in E])
    print("finite edge range per band:", [(round(float(lo), 2), round(float(hi), 2)) for lo, hi in zip(bot, top)])
    ds0 = DenoiseEvents(CORPUS, [], CT, items=[]); cfg = ds0.cfg
    nb = cfg.n_bands
    acc = {}                                      # (plane, bin, band) -> [over slots, occ slots, cells any-over, cells]
    allc = np.zeros((3, nb, 2))                   # every occupied slot of the event, per plane and band
    for ev in range(260, 308):
        z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
        run = os.path.basename(os.path.dirname(str(z["shard"])))
        path = os.path.join(CORPUS, run, os.path.basename(str(z["shard"])))
        with h5py.File(path, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], int(z["event"])))
        tkeys, tq, _ = ds0._truth(run, path, pos)
        sel = tq >= 4e3; keys, q = tkeys[sel].astype(np.int64), tq[sel]
        _, B, bl = ds0.tokens(path, pos)
        inp, occ, band, pl = B["inp"], B["occ"] > 0, B["band_id"], B["plane_id"] % 3
        over = occ & ((inp > top[band][:, None]) | (inp < bot[band][:, None]))
        for p in range(3):
            for b in range(nb):
                m = (pl == p) & (band == b)
                allc[p, b] += [over[m].sum(), occ[m].sum()]
        idx, _ = R_.cell_inputs(keys, B["cell_key"], bl, cfg)
        g, _, _ = R_.unkey(keys); P = g % 3
        for p in range(3):
            for k, (lo, hi) in enumerate(BINS):
                m = (P == p) & (q >= lo) & (q < hi)
                for b in range(nb):
                    ix = idx[m, b]; ix = ix[ix >= 0]
                    r = acc.setdefault((p, k, b), np.zeros(4))
                    r += [over[ix].sum(), occ[ix].sum(), over[ix].any(1).sum(), len(ix)]
        print(f"ev{ev}", flush=True)
    res = dict(top_edge=top.tolist(), bottom_edge=bot.tolist(), cells={}, all_slots={})
    for (p, k, b), r in acc.items():
        res["cells"][f"{'UVY'[p]}_{BINS[k][0]:.0f}_b{b}"] = dict(slot_frac=float(r[0] / max(r[1], 1)),
                                                                 cell_frac=float(r[2] / max(r[3], 1)), n=int(r[3]))
    for p in range(3):
        for b in range(nb):
            res["all_slots"][f"{'UVY'[p]}_b{b}"] = float(allc[p, b, 0] / max(allc[p, b, 1], 1))
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res["cells"].items():
        print(k, {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()})
    print("all occupied slots:", {k: round(v, 5) for k, v in res["all_slots"].items()})


if __name__ == "__main__":
    main()
