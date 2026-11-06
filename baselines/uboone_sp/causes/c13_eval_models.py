#!/usr/bin/env python3
"""C13: score the encoder ablations against their control on the SAME cells (paired), fp32.

Per model: bright cells (training-label complete truth >= 16 k e-) of the test events; per plane and bin
(16-64 k, 64-256 k, >= 256 k) hw68 / median of e = y - log1p(T/Q0); the same for the cathode window (drift tick
3400-3560); the global gain response (all coefficients x1.05, median dy / dlog). Paired vs the --ref model:
event-bootstrap 95% interval of hw68(model) / hw68(ref) per plane and bin.

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/c13_eval_models.py \\
        --ckpt M8c=<path> --ckpt M9lin=<path> ... --ref M8c --out <json>
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
BINS = [(16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="TAG=path")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--events", default="260-307")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", default="16000,64000,256000", help="lower edges of the truth bins (the last is open)")
    a = ap.parse_args()
    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import CellDecoder, build_denoise
    from helix.model.tokenize import assemble, to_fm
    from helix.probe import resolution as R_
    dev = torch.device("cuda")
    global BINS
    lo_e = [float(x) for x in a.bins.split(",")]
    BINS = list(zip(lo_e, lo_e[1:] + [np.inf]))
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
        s = tq >= BINS[0][0]
        evd.append((read_coeff_event(path, pos), c, tk[s].astype(np.int64), tq[s]))

    def tokens(ce, c, scale):
        return to_fm(assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, (ce.value.astype(np.float64) * scale).astype(np.float32),
                              gids=c["gids"], n_wires=c["n_wires"], band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg))

    pred = {}
    for spec in a.ckpt:
        tag, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
        model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta.get("head_kw") or {})
        model.load_state_dict(ck["model"]); model.to(dev).eval(); del ck
        Y0, Y1 = [], []
        for ce, c, keys, T in evd:
            ys = []
            for scale in (1.0, 1.05):
                B = tokens(ce, c, scale)
                idx, aux = R_.cell_inputs(keys, B["cell_key"], c["band_lengths"], cfg)
                ok = (idx >= 0).any(1)
                Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
                Bt["n_cells"] = Bt["plane_id"].shape[0]
                y = np.full(len(keys), np.nan)
                with torch.no_grad():
                    feats = model.fm.encode(Bt)
                    nbr = torch.as_tensor(R_.cell_neighbors(keys[ok], B["cell_key"], c["band_lengths"], cfg)).to(dev)
                    o = model.head(feats, torch.as_tensor(idx[ok]).to(dev), torch.as_tensor(aux[ok]).to(dev), nbr=nbr, B=Bt)
                    y[ok] = (o[0] if isinstance(o, tuple) else o).float().cpu().numpy()
                ys.append(y)
            Y0.append(ys[0]); Y1.append(ys[1])
        pred[tag] = (np.concatenate(Y0), np.concatenate(Y1))
        print("predicted", tag, flush=True)
        del model; torch.cuda.empty_cache()
    keys = np.concatenate([e[2] for e in evd]); T = np.concatenate([e[3] for e in evd])
    evi = np.concatenate([np.full(len(e[2]), i) for i, e in enumerate(evd)])
    g, _, ft = R_.unkey(keys); p = g % 3; yT = np.log1p(T / Q0)
    cath = (ft * R_.FT >= 3400) & (ft * R_.FT < 3560)
    ok = np.all([np.isfinite(v[0]) for v in pred.values()], axis=0)
    hw = lambda x: float(np.subtract(*np.percentile(x, [84, 16])) / 2)
    rng = np.random.default_rng(0)
    out = {}
    for tag, (y0, y1) in pred.items():
        e = y0 - yT
        q0 = Q0 * np.expm1(np.maximum(y0, 0)); lin = np.log1p(q0 * 1.05 / Q0) - np.log1p(q0 / Q0)
        r = {"gain_response_med": float(np.nanmedian(((y1 - y0) / lin)[ok]))}
        for region, rm in (("all", np.ones_like(ok)), ("cathode", cath)):
            for pl in range(3):
                for lo_, hi_ in BINS:
                    m = ok & rm & (p == pl) & (T >= lo_) & (T < hi_)
                    if m.sum() < 50:
                        continue
                    k = f"{region}_{'UVY'[pl]}_{lo_:.0f}"
                    d = dict(hw68=hw(e[m]), med=float(np.median(e[m])), n=int(m.sum()))
                    if tag != a.ref:
                        er = pred[a.ref][0] - yT
                        evs = np.unique(evi[m]); idx = {v: np.nonzero(m & (evi == v))[0] for v in evs}
                        bs = []
                        for _ in range(1000):
                            sel = np.concatenate([idx[v] for v in rng.choice(evs, len(evs))])
                            bs.append(hw(e[sel]) / hw(er[sel]))
                        d["ratio_vs_ref"] = [hw(e[m]) / hw(er[m]), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                    r[k] = d
        out[tag] = r
        print(tag, "gain", round(r["gain_response_med"], 3),
              {k: (round(v["hw68"], 4), [round(x, 3) for x in v.get("ratio_vs_ref", [])]) for k, v in r.items() if k.startswith("all_")}, flush=True)
        print("   cathode", {k: (round(v["hw68"], 4), [round(x, 3) for x in v.get("ratio_vs_ref", [])]) for k, v in r.items() if k.startswith("cathode_")}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
