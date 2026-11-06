#!/usr/bin/env python3
"""T1f of results/m7_tests_plan.md: how much of the corpus input a bright cell's charge needs, measured with
the exact-kernel Tikhonov inversion restricted to token neighbourhoods of the cell.

    CUDA_VISIBLE_DEVICES=3 scripts/helix_run.sh python baselines/uboone_sp/causes/t1f_local.py --events 260-307 --out <json>

Per bright cell (complete truth >= 16 k e-, floor.py truth) and per variant, only the coefficients of these tokens
(per band b, the token covering the cell's centre pixel has wire block wb and time block tb_b):
  cover   (wb, tb_b)                          -- exactly the decoder's direct view (q0 input) of the cell
  time3   (wb, tb_b - 1 .. tb_b + 1)
  nbr3x3  (wb - 1 .. wb + 1, tb_b - 1 .. tb_b + 1)
inverse DWT of those coefficients only, Tikhonov with the exact kernel (lambda 1e-4, 1e-3), read at the cell.
Scored per plane and truth bin against the floor's complete truth, beside the full-image inversion.
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
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "v2")); sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
FLOOR = f"{R}/bright/floor"
BINS = [(16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]
LAMS = (1e-4, 1e-3)
VARIANTS = {"cover": (0, 0), "time3": (0, 1), "nbr3x3": (1, 1)}      # (wire-block radius, time-block radius)
PL = "UVY"
PADW, PADT = 24, 400


def stats(r):
    if len(r) < 20:
        return dict(n=int(len(r)))
    a, m, b = np.percentile(r, [16, 50, 84])
    return dict(n=int(len(r)), med=float(m), hw68=float((b - a) / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-307")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    import build as BD
    from helix.core.coeff_io import read_coeff_event
    from helix.core.wavelet_ops_numpy import reconstruct
    from helix.model.tokenize import PatchConfig, cell_key, pixel_cells, unpack_cell_key
    from helix.probe.resolution import FT, FW
    from helix.tpc.config import DetectorConfig
    from sp.data import Event, registry, PLANES
    from sp.response import load_kernel, kernel_full
    dev = torch.device("cuda")
    dc = DetectorConfig()
    reg = registry()
    cfg = PatchConfig(cell_t="grid_center")
    lo, hi = map(int, a.events.split("-"))
    evs = [e for e in range(lo, hi + 1) if os.path.exists(f"{FLOOR}/ev{e:03d}.npz")]
    kern = {}
    rows = {p: [] for p in range(3)}
    names = ["Tf", "full_tik0.0001", "full_tik0.001"] + [f"{v}_tik{l:g}" for v in VARIANTS for l in LAMS]
    for ev in evs:
        t0 = time.time()
        E = Event(ev)
        shard = E.shard.replace(BD.PS_CORPUS, BD.CFS_CORPUS)
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], E.event))
            bl = fh["config"]["band_lengths"][:]
        ce = read_coeff_event(shard, pos)
        kk = ce.band < cfg.n_bands
        cb, cg, cw, ct, cv = ce.band[kk], ce.plane_gid[kk], ce.wire[kk].astype(np.int64), ce.tau[kk].astype(np.int64), ce.value[kk]
        _, _, cwb, ctb = unpack_cell_key(cell_key(cg, cb, cw, ct, cfg))
        F = np.load(f"{FLOOR}/ev{ev:03d}.npz")
        for g in range(6):
            p = g % 3
            nw, nt = reg[g]["n_wires"], reg[g]["n_ticks"]
            Tf = F[f"truth_g{g}"].astype(np.float64)
            fw, ft = np.nonzero(Tf >= 16e3)
            if not len(fw):
                continue
            pc = pixel_cells(np.full(len(fw), g), fw * FW, ft * FT + FT // 2, bl, cfg)        # (n, n_bands)
            _, _, pwb, ptb = unpack_cell_key(pc)
            wb = pwb[:, 0]
            grp, inv = np.unique(np.concatenate([wb[:, None], ptb], 1), axis=0, return_inverse=True)
            inv = inv.ravel()
            mg = cg == g
            out = {n: np.zeros(len(fw)) for n in names[3:]}
            for vname, (rw, rt) in VARIANTS.items():
                nW = 16 * (2 * rw + 1)
                NW, NT = nW + 2 * PADW, nt + 2 * PADT
                if (p, NW) not in kern:
                    R_, wo, tzb = load_kernel(PLANES[p], "exact")
                    Kf = torch.as_tensor(np.fft.rfft2(kernel_full(R_, wo, tzb, NW, NT)), device=dev)
                    kern[(p, NW)] = {l: Kf.conj() / (Kf.abs() ** 2 + l * (Kf.abs() ** 2).max()) for l in LAMS}
                den = kern[(p, NW)]
                imgs, w0s = [], []
                for gi in range(len(grp)):
                    gwb, gtb = grp[gi, 0], grp[gi, 1:]
                    m = mg & (cwb >= gwb - rw) & (cwb <= gwb + rw)
                    tsel = np.zeros(len(cb), bool)
                    for b in range(cfg.n_bands):
                        tsel |= (cb == b) & (ctb >= gtb[b] - rt) & (ctb <= gtb[b] + rt)
                    m &= tsel
                    w0 = 16 * (gwb - rw)
                    bands = [np.zeros((nW, int(L)), np.float64) for L in bl]
                    for b in range(cfg.n_bands):
                        mb = m & (cb == b)
                        wi = cw[mb] - w0
                        ok = (wi >= 0) & (wi < nW)
                        bands[b][wi[ok], ct[mb][ok]] = cv[mb][ok]
                    img = np.asarray(reconstruct(bands, dc.wavelet, dc.dwt_level, dc.dwt_mode, nt), np.float64)
                    if w0 < 0:                                   # plane edge: wires below 0 do not exist
                        img[:-w0] = 0
                    if w0 + nW > nw:
                        img[nw - w0:] = 0
                    imgs.append(img); w0s.append(w0)
                for l in LAMS:
                    est = np.zeros(len(fw))
                    for s in range(0, len(imgs), 64):
                        X = torch.zeros((len(imgs[s:s + 64]), NW, NT), dtype=torch.float64, device=dev)
                        for j, img in enumerate(imgs[s:s + 64]):
                            X[j, PADW:PADW + nW, PADT:PADT + nt] = torch.as_tensor(img, device=dev)
                        Y = torch.fft.irfft2(torch.fft.rfft2(X) * den[l], s=(NW, NT))[:, PADW:PADW + nW, PADT:PADT + nt]
                        # fine cells (2 wires x 16 ticks) of the window
                        ntc = -(-nt // FT)                                  # pad time like sp.pipeline.fine_cells
                        Y = torch.nn.functional.pad(Y, (0, ntc * FT - nt))
                        Yc = Y.reshape(Y.shape[0], nW // FW, FW, ntc, FT).sum((2, 4)).cpu().numpy()
                        for j in range(Yc.shape[0]):
                            gi = s + j
                            cells = np.nonzero(inv == gi)[0]
                            lw = fw[cells] - w0s[gi] // FW
                            est[cells] = np.maximum(Yc[j, lw, ft[cells]], 0)
                    out[f"{vname}_tik{l:g}"] = est
            cols = [Tf[fw, ft], F[f"corpus_tik0.0001_g{g}"][fw, ft], F[f"corpus_tik0.001_g{g}"][fw, ft]] + [out[n] for n in names[3:]]
            rows[p].append(np.stack(cols, 1))
        print(f"ev{ev:03d} {time.time() - t0:.0f}s", flush=True)
    res = dict(events=evs, names=names, by_bin={})
    for p in range(3):
        A = np.concatenate(rows[p]); Tf = A[:, 0]
        res["by_bin"][PL[p]] = {f"{lo_:.0f}-{hi_:.0f}": {n: stats(A[(Tf >= lo_) & (Tf < hi_), j] / Tf[(Tf >= lo_) & (Tf < hi_)] - 1)
                                                         for j, n in enumerate(names) if j}
                                for lo_, hi_ in BINS}
    json.dump(res, open(a.out, "w"), indent=1)
    for p in PL:
        for b, d in res["by_bin"][p].items():
            print(p, b, {n: (round(v.get("med", 0), 4), round(v.get("hw68", 0), 4)) for n, v in d.items()})


if __name__ == "__main__":
    main()
