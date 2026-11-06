#!/usr/bin/env python3
"""C4: does M7dc get anything useful from the OTHER TPC volume? Encode each volume as its own event (attention
cannot cross volumes; an intervention, not a training change) and compare with the normal full-event encoding:
  * bright-cell accuracy (hw68 / median of y - log1p(T/Q0), fp32) per plane and truth bin;
  * the within-volume coupling: scale plane gid 0 (U of volume 0) by 1.05, response of U cells (inside) and of the
    V / Y cells of volume 0 (control), full vs split encoding.

    CUDA_VISIBLE_DEVICES=3 scripts/helix_run.sh python baselines/uboone_sp/causes/c4_split_volumes.py --out <json>
"""
import argparse
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
CKPT = f"{R}/denoise/M7dc_ft_full_dec_cond_ct25/final.pt"
Q0 = 2043.2578108909702
BINS = [(16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-283")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.model.tokenize import assemble, to_fm
    from helix.probe import resolution as R_
    dev = torch.device("cuda")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta["head_kw"])
    model.load_state_dict(ck["model"]); model.to(dev).eval(); del ck
    ds = DenoiseEvents(CORPUS, [], CT, items=[]); cfg = ds.cfg

    @torch.no_grad()
    def predict(ce, c, keys, scale_mask=None):
        v = ce.value.astype(np.float64) * (1.0 if scale_mask is None else np.where(scale_mask, 1.05, 1.0))
        B = to_fm(assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, v.astype(np.float32), gids=c["gids"], n_wires=c["n_wires"],
                           band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg))
        idx, aux = R_.cell_inputs(keys, B["cell_key"], c["band_lengths"], cfg)
        ok = (idx >= 0).any(1)
        y = np.full(len(keys), np.nan)
        if not ok.any():
            return y
        Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
        Bt["n_cells"] = Bt["plane_id"].shape[0]
        feats = model.fm.encode(Bt)
        nbr = torch.as_tensor(R_.cell_neighbors(keys[ok], B["cell_key"], c["band_lengths"], cfg)).to(dev)
        o = model.head(feats, torch.as_tensor(idx[ok]).to(dev), torch.as_tensor(aux[ok]).to(dev), nbr=nbr, B=Bt)
        y[ok] = o[0].float().cpu().numpy()
        return y

    def sub(ce, m):
        return types.SimpleNamespace(band=ce.band[m], plane_gid=ce.plane_gid[m], wire=ce.wire[m], tau=ce.tau[m], value=ce.value[m])

    rec = {k: [] for k in ("g", "T", "full", "split", "full_g0", "split_g0")}
    lo, hi = map(int, a.events.split("-"))
    for ev in range(lo, hi + 1):
        z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
        run = os.path.basename(os.path.dirname(str(z["shard"])))
        path = os.path.join(CORPUS, run, os.path.basename(str(z["shard"])))
        with h5py.File(path, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], int(z["event"])))
            c = {k: fh["config"][k][:] for k in ("gids", "n_wires", "band_lengths", "norm_sigma")}
        tkeys, tq, _ = ds._truth(run, path, pos)
        sel = tq >= 16e3
        keys, T = tkeys[sel].astype(np.int64), tq[sel]
        g, _, _ = R_.unkey(keys)
        ce = read_coeff_event(path, pos)
        full = predict(ce, c, keys); full_g0 = predict(ce, c, keys, ce.plane_gid == 0)
        split = np.full(len(keys), np.nan); split_g0 = np.full(len(keys), np.nan)
        for vol in (0, 1):
            mv = (ce.plane_gid // 3) == vol; kv = (g // 3) == vol
            cv = sub(ce, mv)
            split[kv] = predict(cv, c, keys[kv])
            split_g0[kv] = predict(cv, c, keys[kv], cv.plane_gid == 0)
        for k, v in (("g", g), ("T", T), ("full", full), ("split", split), ("full_g0", full_g0), ("split_g0", split_g0)):
            rec[k].append(v)
        print(f"ev{ev:03d} {len(keys)} cells", flush=True)
    D = {k: np.concatenate(v) for k, v in rec.items()}
    yT = np.log1p(D["T"] / Q0); p = D["g"] % 3
    st = lambda x: (lambda a16, a50, a84: dict(med=float(a50), hw68=float((a84 - a16) / 2), n=int(len(x))))(*np.percentile(x, [16, 50, 84]))
    out = {"accuracy": {}, "coupling_g0": {}}
    for pl in range(3):
        for lo_, hi_ in BINS:
            m = (p == pl) & (D["T"] >= lo_) & (D["T"] < hi_) & np.isfinite(D["full"]) & np.isfinite(D["split"])
            out["accuracy"][f"{'UVY'[pl]}_{lo_:.0f}"] = dict(full=st(D["full"][m] - yT[m]), split=st(D["split"][m] - yT[m]),
                                                            split_minus_full=st(D["split"][m] - D["full"][m]))
    for mode in ("full", "split"):
        y0, y1 = D[mode], D[f"{mode}_g0"]
        q0 = Q0 * np.expm1(np.maximum(y0, 0)); lin = np.log1p(q0 * 1.05 / Q0) - np.log1p(q0 / Q0)
        s = (y1 - y0) / lin
        fin = np.isfinite(s)
        out["coupling_g0"][mode] = dict(inside_U_vol0=st(s[fin & (D["g"] == 0)]), control_VY_vol0=st(s[fin & np.isin(D["g"], [1, 2])]),
                                        control_vol1=st(s[fin & (D["g"] >= 3)]))
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in out["accuracy"].items():
        print(k, "full", round(v["full"]["hw68"], 4), round(v["full"]["med"], 4), "split", round(v["split"]["hw68"], 4), round(v["split"]["med"], 4),
              "split-full med", round(v["split_minus_full"]["med"], 4), "hw", round(v["split_minus_full"]["hw68"], 4))
    for mode, v in out["coupling_g0"].items():
        print("coupling", mode, {k: round(x["med"], 3) for k, x in v.items()})


if __name__ == "__main__":
    main()
