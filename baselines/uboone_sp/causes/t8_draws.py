#!/usr/bin/env python3
"""T8: is M7dc's per-cell bright error noise-driven or a deterministic approximation error?

Same events, 5 fresh noise draws (scripts/eval_draws.py cache; same clean signal, independent noise). Per bright
cell (training-label complete truth T >= 16 k e-), error e_k = log q_k - log T in draw k, for M7dc (fp32, so the
bf16 output grid adds no spurious draw-to-draw noise) and for the exact-kernel corpus Tikhonov on the same draw's
coefficients (lambda 1e-4, 1e-3). Decomposition with K draws:
  noise part      v_n = mean over cells of the across-draw sample variance of e_k
  systematic part v_s = var over cells of mean_k e_k  -  v_n / K      (the part that repeats in every draw)
Prediction: a noise-limited estimator has v_s << v_n (the inversion should); a model whose error is a fixed
function of the signal configuration (approximation error) has v_s >> v_n.

    CUDA_VISIBLE_DEVICES=1 scripts/helix_run.sh python baselines/uboone_sp/causes/t8_draws.py --events 260-283 --out <dir>
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BASE))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "v2")); sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
DRAWS = f"{R}/eval_draws/coeffs"
CKPT = f"{R}/denoise/M7dc_ft_full_dec_cond_ct25/final.pt"
LAMS = (1e-4, 1e-3)
PADW, PADT = 24, 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-283")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    import floor as FL
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.model.tokenize import assemble, to_fm
    from helix.probe import resolution as R_
    from sp.data import registry, PLANES
    from sp.pipeline import fine_cells
    from sp.response import load_kernel, kernel_full
    dev = torch.device("cuda")
    reg = registry()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta["head_kw"])
    model.load_state_dict(ck["model"]); model.to(dev).eval(); del ck
    ds = DenoiseEvents(CORPUS, [], CT, items=[]); cfg = ds.cfg
    kern = {}
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
        g, fw, ft = R_.unkey(keys)
        res = dict(keys=keys, Tq=Tq)
        for k in range(a.draws):
            d = np.load(os.path.join(DRAWS, f"ev{ev:03d}_d{k}.npz"))
            ce = types.SimpleNamespace(band=d["band"], plane_gid=d["plane_gid"], wire=d["wire"], tau=d["tau"], value=d["value"])
            tok = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value.astype(np.float32), gids=c["gids"], n_wires=c["n_wires"],
                           band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg)
            B = to_fm(tok)
            idx, aux = R_.cell_inputs(keys, B["cell_key"], c["band_lengths"], cfg)
            ok = (idx >= 0).any(1)
            nbr = R_.cell_neighbors(keys, B["cell_key"], c["band_lengths"], cfg)
            Bt = {kk: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for kk, x in B.items()}
            Bt["n_cells"] = Bt["plane_id"].shape[0]
            y = np.full(len(keys), np.nan, np.float32)
            with torch.no_grad():
                feats = model.fm.encode(Bt)
                s_all = np.nonzero(ok)[0]
                for s in range(0, len(s_all), 65536):
                    ss = s_all[s:s + 65536]
                    o = model.head(feats, torch.as_tensor(idx[ss]).to(dev), torch.as_tensor(aux[ss]).to(dev),
                                   nbr=torch.as_tensor(nbr[ss]).to(dev), B=Bt)
                    y[ss] = o[0].float().cpu().numpy()
            res[f"y_d{k}"] = y
            # exact-kernel Tikhonov on this draw's coefficients
            for lam in LAMS:
                res[f"tik{lam:g}_d{k}"] = np.zeros(len(keys))
            for gg in range(6):
                m = g == gg
                if not m.any():
                    continue
                p = gg % 3
                nw, nt = reg[gg]["n_wires"], reg[gg]["n_ticks"]
                NW, NT = nw + 2 * PADW, nt + 2 * PADT
                if p not in kern:
                    R0, wo, tzb = load_kernel(PLANES[p], "exact")
                    Kf = torch.as_tensor(np.fft.rfft2(kernel_full(R0, wo, tzb, NW, NT)), device=dev)
                    kern[p] = {l: Kf.conj() / (Kf.abs() ** 2 + l * (Kf.abs() ** 2).max()) for l in LAMS}
                S = FL.corpus_image(ce, c["band_lengths"], gg, nw, nt)
                Y = torch.fft.rfft2(torch.as_tensor(np.pad(S, ((PADW, PADW), (PADT, PADT))), device=dev))
                for lam in LAMS:
                    x = torch.fft.irfft2(Y * kern[p][lam], s=(NW, NT))[PADW:PADW + nw, PADT:PADT + nt].cpu().numpy()
                    Fc = fine_cells(x)
                    res[f"tik{lam:g}_d{k}"][m] = Fc[fw[m], ft[m]]
        np.savez(out + ".tmp.npz", **res)
        os.replace(out + ".tmp.npz", out)
        print(f"ev{ev:03d}: {len(keys)} bright cells, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
