#!/usr/bin/env python3
"""C15: are the distributional charge heads' per-cell uncertainties CALIBRATED, and how sharp are they?

For each checkpoint with a distributional CellDecoder (dist gauss / hlgauss), on the 48 floor test events
(training-label complete truth, cells >= 1 k e-), per plane and truth bin:
  gauss    z = (y_true - mu) / sigma: fraction with |z| < 1 (0.683 if calibrated) and < 2 (0.954); median sigma vs
           the actual error's hw68 (sharpness: equal if the width matches the spread of the errors)
  hlgauss  PIT = predictive CDF at y_true: fraction inside [0.16, 0.84] (0.68) and [0.025, 0.975] (0.95); median
           predictive std vs the actual error's hw68
y = log1p(q / Q0) throughout (a width of 0.01 is ~1% in charge at bright cells).

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/c15_dist_calib.py \
        --ckpt M14g05=<final.pt> --ckpt M14hl=<final.pt> --out <json>
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
Q0 = 2043.2578108909702
BINS = [(1e3, 4e3), (4e3, 16e3), (16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="TAG=path")
    ap.add_argument("--events", default="260-307")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
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
        tk, tq, _ = ds._truth(run, path, pos)
        s = tq >= BINS[0][0]
        evd.append((path, pos, tk[s].astype(np.int64), tq[s]))
    out = {}
    for spec in a.ckpt:
        tag, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
        model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta.get("head_kw") or {})
        model.load_state_dict(ck["model"]); model.to(dev).eval(); del ck
        dist = model.head.dist
        E, S, C, P, T = [], [], [], [], []                        # error, width, calibration stat, plane, truth
        for path_, pos, keys, tq in evd:
            _, B, bl = ds.tokens(path_, pos)
            idx, aux = R_.cell_inputs(keys, B["cell_key"], bl, cfg)
            ok = (idx >= 0).any(1)
            Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
            Bt["n_cells"] = Bt["plane_id"].shape[0]
            yT = torch.as_tensor(np.log1p(tq[ok] / Q0), device=dev, dtype=torch.float32)
            with torch.no_grad():
                feats = model.fm.encode(Bt)
                nbr = torch.as_tensor(R_.cell_neighbors(keys[ok], B["cell_key"], bl, cfg)).to(dev)
                yp, _, extra = model.head(feats, torch.as_tensor(idx[ok]).to(dev), torch.as_tensor(aux[ok]).to(dev), nbr=nbr, B=Bt)
                yp = yp.float()
                if dist == "gauss":
                    sig = torch.exp(extra.float().clamp(-7.0, 3.0))
                    stat = (yT - yp) / sig                            # z
                else:
                    pr = extra.float().softmax(-1); c = model.head.centers; e = model.head.edges
                    w = (e[1:] - e[:-1])
                    frac = ((yT[:, None] - e[None, :-1]) / w[None, :]).clamp(0, 1)   # share of each bin below y_true
                    stat = (pr * frac).sum(-1)                         # PIT
                    sig = torch.sqrt((pr * (c[None, :] - yp[:, None]) ** 2).sum(-1))
            g, _, _ = R_.unkey(keys[ok])
            E.append((yp - yT).cpu().numpy()); S.append(sig.cpu().numpy()); C.append(stat.cpu().numpy())
            P.append(g % 3); T.append(tq[ok])
        E, S, C, P, T = map(np.concatenate, (E, S, C, P, T))
        r = {"dist": dist}
        for pl in range(3):
            for lo_, hi_ in BINS:
                m = (P == pl) & (T >= lo_) & (T < hi_)
                if m.sum() < 50:
                    continue
                e16, e84 = np.percentile(E[m], [16, 84])
                d = dict(n=int(m.sum()), err_hw68=float((e84 - e16) / 2), width_med=float(np.median(S[m])))
                if dist == "gauss":
                    d.update(cov1=float(np.mean(np.abs(C[m]) < 1)), cov2=float(np.mean(np.abs(C[m]) < 2)))
                else:
                    d.update(cov68=float(np.mean((C[m] > 0.16) & (C[m] < 0.84))),
                             cov95=float(np.mean((C[m] > 0.025) & (C[m] < 0.975))))
                r[f"{'UVY'[pl]}_{lo_:.0f}"] = d
        out[tag] = r
        print(tag, dist, {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                          for k, v in r.items() if k != "dist"}, flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
