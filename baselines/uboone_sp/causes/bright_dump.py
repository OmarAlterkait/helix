#!/usr/bin/env python3
"""Full-plane per-cell outputs of the decoder denoisers under different inference
precisions (dense/model_dump2.py, the eval's predict, with a precision switch).

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/bright_dump.py \
        --events 260-291 --precision head32 --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/bright/model/head32 \
        --ckpt M6d=.../M6d_ft_full_dec/final.pt --ckpt M6dc=.../M6dc_ft_full_dec_cond/final.pt

--precision
  bf16    as eval_denoise: encoder AND head under torch.autocast(bfloat16)  (= dense3 maps)
  head32  encoder under bf16 autocast, head in fp32 (features cast to fp32)
  fp32    encoder and head in fp32 (no autocast; TF32 matmuls disabled)
Same tokens, cell_inputs, cell_neighbors; cells no token covers -> 0. Only token-covered
cells are predicted (no window/map-row cells), so no truth_v2_near file is read.
Output per event: q_<T>_g<g>, p_<T>_g<g>, cov_g<g>, toff (map[fw, ft + toff]).
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
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
PS_CORPUS, CFS_CORPUS = "/pscratch/sd/o/oalter/coeff_tpc_r1", "/global/cfs/cdirs/m5238/users/oalter/coeff_tpc_r1"
SHAPE = {0: (1010, 296), 1: (1010, 296), 2: (740, 296)}
TOFF = 4


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", required=True, help="a-b or comma list")
    ap.add_argument("--skip", default="", help="comma list of events not to read (truth file on OST 61)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--precision", default="bf16", choices=("bf16", "head32", "fp32"))
    ap.add_argument("--chunk", type=int, default=65536)
    a = ap.parse_args()
    import contextlib
    import h5py
    import torch
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.probe import resolution as R
    from model_cells import covered_cells_fn

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
        del ck
    ds = DenoiseEvents("", [], "", items=[]); cfg = ds.cfg
    covered_cells = covered_cells_fn(cfg, R)
    os.makedirs(a.out, exist_ok=True)
    if "-" in a.events and "," not in a.events:
        lo, hi = (int(x) for x in a.events.split("-")); evs = list(range(lo, hi + 1))
    else:
        evs = [int(x) for x in a.events.split(",")]
    skip = {int(x) for x in a.skip.split(",") if x}
    ac_enc = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if a.precision != "fp32" else contextlib.nullcontext
    ac_head = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if a.precision == "bf16" else contextlib.nullcontext
    for ev in evs:
        out = os.path.join(a.out, f"ev{ev:03d}.npz")
        if ev in skip or os.path.exists(out):
            continue
        t0 = time.time()
        z = np.load(os.path.join(TRUTH, f"ev{ev:03d}.npz"), allow_pickle=True)
        shard, event = str(z["shard"]).replace(PS_CORPUS, CFS_CORPUS), int(z["event"])
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        ce, B, bl = ds.tokens(shard, pos)
        keys = covered_cells(ce).astype(np.int64)
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
        nbr_all = R.cell_neighbors(keys[sel_all], B["cell_key"], bl, cfg)
        for tag, (model, presence, decoder, Q0) in models.items():
            y = np.zeros(len(keys), np.float32); pr = np.zeros(len(keys), np.float32)
            with torch.no_grad():
                with ac_enc():
                    feats = model.fm.encode(Bt)
                if a.precision != "bf16":
                    feats = feats.float()
                for s in range(0, len(sel_all), a.chunk):
                    sel = sel_all[s:s + a.chunk]
                    ti, ta = torch.as_tensor(idx[sel]).to(dev), torch.as_tensor(aux[sel]).to(dev)
                    with ac_head():
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
        np.savez(out + ".tmp.npz", toff=TOFF, precision=a.precision, **res)
        os.replace(out + ".tmp.npz", out)
        print(f"ev{ev:03d}: {int(ok.sum())} cells, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
