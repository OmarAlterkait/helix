#!/usr/bin/env python3
"""T2 / T3a of results/m7_tests_plan.md: the model as a function, and train-set vs test-set error.

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/t2t3_model.py --set test  --out <dir>
    CUDA_VISIBLE_DEVICES=1 scripts/helix_run.sh python baselines/uboone_sp/causes/t2t3_model.py --set train --out <dir>

Cells: every complete-truth cell >= 1 k e- (training-label truth, denoise_truth_complete, for BOTH sets, so the
two sets are scored identically). Per cell the raw charge-head output y of M7dc under:
  base   bf16 autocast (the eval's precision)            fp32   no autocast (T2b)
  g<e>   every coefficient value scaled by (1 + e) (T2a; a linear map gives dy = log1p(q(1+e)/Q0) - log1p(q/Q0))
  j<k>   every value multiplied by (1 + 1e-3 N(0,1)), k = 0..2 (T2c: local jitter)
and M7d base. test = the 48 floor events ev260-307 (probe split); train = 48 events from M7dc's training
split (seed 0). Writes per-event npz; t2t3_summary.py aggregates.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BASE))
sys.path.insert(0, ROOT)
R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
CK = {"M7dc": f"{R}/denoise/M7dc_ft_full_dec_cond_ct25/final.pt", "M7d": f"{R}/denoise/M7d_ft_full_dec_ct25/final.pt"}
GAINS = (-0.02, -0.01, 0.005, 0.01, 0.02, 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["test", "train"], required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", default=None, help="TAG=path (T3b: other models, base precision only)")
    ap.add_argument("--fp32-variants", action="store_true",
                    help="gain / jitter variants without autocast (bf16 rounds y to 0.031 steps above y = 4, which "
                         "hides a 1% response); writes y_M7dc_fp32 as the reference")
    a = ap.parse_args()
    ck_map = dict(c.split("=", 1) for c in a.ckpt) if a.ckpt else CK
    base_only = bool(a.ckpt)
    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents, split_events
    from helix.model.denoise import build_denoise
    from helix.model.tokenize import assemble, to_fm
    from helix.probe import resolution as R_
    dev = torch.device("cuda")
    models = {}
    runs = None
    for tag, path in ck_map.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
        m = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta["head_kw"])
        m.load_state_dict(ck["model"]); m.to(dev).eval()
        models[tag] = m
        if runs is None:
            runs = meta["runs"]
        del ck
    ds = DenoiseEvents(CORPUS, [], CT, items=[])
    cfg = ds.cfg
    if a.set == "test":
        items = []
        for ev in range(260, 308):
            z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
            shard = str(z["shard"]); event = int(z["event"])
            run = os.path.basename(os.path.dirname(shard))
            path = os.path.join(CORPUS, run, os.path.basename(shard))
            with h5py.File(path, "r") as fh:
                pos = int(np.searchsorted(fh["ident"]["event"][:], event))
            items.append((run, path, pos, event, f"ev{ev:03d}"))
    else:
        tr = split_events(CORPUS, runs, "train")
        pick = np.random.default_rng(0).permutation(len(tr))[:a.n]
        items = [(*tr[i], f"tr{i:06d}") for i in sorted(pick)]
    os.makedirs(a.out, exist_ok=True)
    json.dump([list(map(str, it)) for it in items], open(os.path.join(a.out, "items.json"), "w"))
    shard_cfg = {}

    def tokens(path, pos, scale):
        if path not in shard_cfg:
            with h5py.File(path, "r") as f:
                shard_cfg[path] = {k: f["config"][k][:] for k in ("gids", "n_wires", "band_lengths", "norm_sigma")}
        c = shard_cfg[path]
        ce = read_coeff_event(path, pos)
        v = ce.value.astype(np.float64) * scale
        tok = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, v.astype(np.float32), gids=c["gids"], n_wires=c["n_wires"],
                       band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg)
        return to_fm(tok), c["band_lengths"], ce.value.shape[0]

    for run, path, pos, event, name in items:
        out = os.path.join(a.out, f"{name}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        tkeys, tq, tev = ds._truth(run, path, int(pos))
        if tev != int(event):
            raise SystemExit(f"{path}#{pos}: truth event {tev} != {event}")
        sel = tq >= 1000
        keys, Tq = tkeys[sel].astype(np.int64), tq[sel].astype(np.float64)
        res = dict(keys=keys, Tq=Tq)
        variants = [("base", 1.0, True), ("fp32", 1.0, False)] + [(f"g{e:+g}", 1 + e, True) for e in GAINS] + \
                   [(f"j{k}", ("jit", k), True) for k in range(3)]
        if base_only:
            variants = variants[:1]
        if a.fp32_variants:
            variants = [("fp32", 1.0, False)] + [(f"g{e:+g}", 1 + e, False) for e in GAINS] + \
                       [(f"j{k}", ("jit", k), False) for k in range(3)]
        idx = aux = nbr = None
        for vname, scale, bf16 in variants:
            if isinstance(scale, tuple):
                rs = np.random.default_rng((int(pos), scale[1], 77))
                ce_n = read_coeff_event(path, int(pos)).value.shape[0]
                scale = 1.0 + 1e-3 * rs.standard_normal(ce_n)
            B, bl, _ = tokens(path, int(pos), scale)
            # the token grid (cell keys) does not depend on the values: idx / nbr computed once
            if idx is None:
                idx, aux = R_.cell_inputs(keys, B["cell_key"], bl, cfg)
                ok = (idx >= 0).any(1)
                nbr = R_.cell_neighbors(keys, B["cell_key"], bl, cfg)
                res["covered"] = ok
            Bt = {k: torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v for k, v in B.items()}
            Bt["n_cells"] = Bt["plane_id"].shape[0]
            for tag, model in models.items():
                if tag != "M7dc" and vname not in ("base", "fp32"):
                    continue
                y = np.zeros(len(keys), np.float32); pr = np.zeros(len(keys), np.float32)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    feats = model.fm.encode(Bt)
                    s_all = np.nonzero(ok)[0]
                    for s in range(0, len(s_all), 65536):
                        ss = s_all[s:s + 65536]
                        o = model.head(feats, torch.as_tensor(idx[ss]).to(dev), torch.as_tensor(aux[ss]).to(dev),
                                       nbr=torch.as_tensor(nbr[ss]).to(dev), B=Bt)
                        pr[ss] = torch.sigmoid(o[1].float()).cpu().numpy(); y[ss] = o[0].float().cpu().numpy()
                res[f"y_{tag}_{vname}"] = y
                if vname in ("base", "fp32"):
                    res[f"p_{tag}"] = pr
        np.savez(out + ".tmp.npz", **res)
        os.replace(out + ".tmp.npz", out)
        print(f"{name}: {len(keys)} cells >= 1k, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
