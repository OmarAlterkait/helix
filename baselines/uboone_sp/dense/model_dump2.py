#!/usr/bin/env python3
"""Full-plane per-cell outputs of trained denoisers (incl. the CellDecoder head),
exactly as scripts/eval_denoise.py predicts them (same tokens, cell_inputs,
cell_neighbors for decoder heads, bf16 autocast; cells no token covers -> 0).
v2/model_dump.py extended to decoder heads; it reads the corpus from its CFS copy
(the /pscratch copy sits partly on a hung OST) and writes to CFS.

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/dense/model_dump2.py \
        --events 260-291 --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/dense2/model \
        --ckpt M5d=/global/cfs/.../M5d_ft_full_dec/final.pt --ckpt M4n=...

Per event one npz, per plane gid g and checkpoint tag T:
  q_<T>_g<g>   raw charge map expm1(max(y, 0)) * Q0   (gated = q where p > 0.5)
  p_<T>_g<g>   presence probability sigmoid(logit)
  cov_g<g>     covered by a token;  toff: map[fw, ft + toff]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.dirname(BASE)))
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"           # test files checked off OST 61
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"
PS_CORPUS, CFS_CORPUS = "/pscratch/sd/o/oalter/coeff_tpc_r1", "/global/cfs/cdirs/m5238/users/oalter/coeff_tpc_r1"
SHAPE = {0: (1010, 296), 1: (1010, 296), 2: (740, 296)}
TOFF = 4


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="TAG=path")
    ap.add_argument("--chunk", type=int, default=65536)
    a = ap.parse_args()

    import h5py
    import torch
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.probe import resolution as R
    from model_cells import covered_cells_fn

    dev = torch.device("cuda")
    models = {}
    for spec in a.ckpt:
        tag, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["meta"]
        ov = dict(meta["overrides"]); ov["compile_blocks"] = False
        hk = meta.get("head_kw") or {}
        m = build_denoise(meta["arch"], None, overrides=ov, head_kw=hk)
        m.load_state_dict(ck["model"]); m.to(dev).eval()
        models[tag] = (m, bool(hk.get("presence")), hk.get("kind") == "decoder", float(meta["q0"]))
        print(tag, path, "presence", bool(hk.get("presence")), "decoder", hk.get("kind") == "decoder", flush=True)
        del ck
    ds = DenoiseEvents("", [], "", items=[]); cfg = ds.cfg
    covered_cells = covered_cells_fn(cfg, R)
    os.makedirs(a.out, exist_ok=True)
    lo, hi = (int(x) for x in a.events.split("-"))
    for ev in range(lo, hi + 1):
        out = os.path.join(a.out, f"ev{ev:03d}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        z = np.load(os.path.join(TRUTH, f"ev{ev:03d}.npz"), allow_pickle=True)
        zn = np.load(os.path.join(NEAR, f"ev{ev:03d}.npz"), allow_pickle=True)
        shard, event = str(z["shard"]).replace(PS_CORPUS, CFS_CORPUS), int(z["event"])
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        ce, B, bl = ds.tokens(shard, pos)
        keys = np.unique(np.concatenate([covered_cells(ce), z["mrows"], z["wkey"], zn["wkey"]]).astype(np.int64))
        g, fw, ft = R.unkey(keys)
        inside = np.zeros(len(keys), bool)
        for p in range(6):
            sh = SHAPE[p % 3]
            inside |= (g == p) & (fw >= 0) & (fw < sh[0]) & (ft >= -TOFF) & (ft + TOFF < sh[1])
        keys, g, fw, ft = keys[inside], g[inside], fw[inside], ft[inside]
        idx, aux = R.cell_inputs(keys, B["cell_key"], bl, cfg)
        ok = (idx >= 0).any(1)
        Bt = {k: torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v for k, v in B.items()}
        Bt["n_cells"] = Bt["plane_id"].shape[0]
        res = {}
        for p in range(6):
            cov = np.zeros(SHAPE[p % 3], bool)
            m = (g == p) & ok
            cov[fw[m], ft[m] + TOFF] = True
            res[f"cov_g{p}"] = cov
        sel_all = np.nonzero(ok)[0]
        nbr_all = R.cell_neighbors(keys[sel_all], B["cell_key"], bl, cfg) if any(v[2] for v in models.values()) else None
        for tag, (model, presence, decoder, Q0) in models.items():
            y = np.zeros(len(keys), np.float32); pr = np.zeros(len(keys), np.float32)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                feats = model.fm.encode(Bt)
                for s in range(0, len(sel_all), a.chunk):
                    sel = sel_all[s:s + a.chunk]
                    ti, ta = torch.as_tensor(idx[sel]).to(dev), torch.as_tensor(aux[sel]).to(dev)
                    if decoder:
                        o = model.head(feats, ti, ta, nbr=torch.as_tensor(nbr_all[s:s + a.chunk]).to(dev), B=Bt)
                    else:
                        o = model.head(feats, ti, ta)
                    if presence:
                        pr[sel] = torch.sigmoid(o[1].float()).cpu().numpy(); o = o[0]
                    y[sel] = o.float().cpu().numpy()
            q = (np.expm1(np.maximum(y, 0)) * Q0).astype(np.float32)
            for p in range(6):
                m = g == p
                Mq = np.zeros(SHAPE[p % 3], np.float32); Mq[fw[m], ft[m] + TOFF] = q[m]
                res[f"q_{tag}_g{p}"] = Mq
                if presence:
                    Mp = np.zeros(SHAPE[p % 3], np.float32); Mp[fw[m], ft[m] + TOFF] = pr[m]
                    res[f"p_{tag}_g{p}"] = Mp
            del feats
        np.savez(out + ".tmp.npz", toff=TOFF, **res)
        os.replace(out + ".tmp.npz", out)
        print(f"ev{ev:03d}: {int(ok.sum())} cells, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
