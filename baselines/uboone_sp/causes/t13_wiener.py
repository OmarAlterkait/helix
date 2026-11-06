#!/usr/bin/env python3
"""T13: is the bright charge a LEARNABLE LINEAR function of the input -- with no physics knowledge?

floor.py's Tikhonov inversion uses the exact simulated response kernel. Here the linear map is LEARNED from
(input, truth) pairs, as a network would: per plane type, the minimum-mean-square translation-invariant filter
H(f) = sum conj(S_f) T_f / sum |S_f|^2 over the fit events' corpus images S (the coefficients the model reads,
inverse-DWT'd, D1 dropped) and complete-truth pixel images T (causes/build.py's rebuild). Fit on ev000..ev(N-1)
(two estimates: the first half and all N, to see whether H is converged), applied to the 48 floor test events.
Scored on fine cells (2 wires x 16 ticks) against the same truth, alongside floor.py's Tikhonov (lam 1e-3, 1e-4)
recomputed here so every number shares one metric: hw68 and median of log(pred / truth) per plane and truth bin.

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/t13_wiener.py --fit 0-127 --out <json>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "v2")); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(BASE)))
BINS = [(4e3, 16e3), (16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]
PADW, PADT = 24, 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="0-127")
    ap.add_argument("--test", default="260-307")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    import yaml
    from helix.core.coeff_io import read_coeff_event
    from sp.data import Event, registry
    from sp.pipeline import fine_cells
    from sp.response import load_kernel, kernel_full
    import build as BD
    import resim as RS
    from floor import corpus_image
    sys.path.insert(0, RS.JAXTPC)
    from tools.config import create_sim_config
    cfg = create_sim_config(yaml.safe_load(open(os.path.join(RS.JAXTPC, "config", "cubic_wireplane_config.yaml"))),
                            total_pad=1000, include_digitize=True, num_s=16)
    dev = torch.device("cuda")
    reg = registry()
    nshape = {g: (reg[g]["n_wires"], reg[g]["n_ticks"]) for g in range(6)}

    def load(ev):
        E = Event(ev)
        Tfull, _, _ = BD.rebuild(ev, E, cfg, nshape)
        shard = E.shard.replace(BD.PS_CORPUS, BD.CFS_CORPUS)
        with h5py.File(shard, "r") as fh:
            bl = fh["config"]["band_lengths"][:]
        ce = read_coeff_event(shard, E.pos)
        return {g: (corpus_image(ce, bl, g, *nshape[g]), Tfull[g]) for g in range(6)}

    def spec(X, nw, nt):
        return torch.fft.rfft2(torch.as_tensor(np.pad(X, ((PADW, PADW), (PADT, PADT))), device=dev, dtype=torch.float64))

    lo, hi = (int(x) for x in a.fit.split("-"))
    fit = list(range(lo, hi + 1)); half = set(fit[:len(fit) // 2])
    N = {}; D = {}
    t0 = time.time(); n_ok = 0
    for ev in fit:
        try:
            pairs = load(ev)
        except Exception as e:                                   # a fit event that cannot be rebuilt is skipped
            print(f"fit ev{ev:03d} skipped: {type(e).__name__}: {e}", flush=True); continue
        n_ok += 1
        for g, (S, T) in pairs.items():
            p = g % 3; nw, nt = nshape[g]
            Sf, Tf = spec(S, nw, nt), spec(T, nw, nt)
            for k in (("all",) + (("half",) if ev in half else ())):
                if (k, p) not in N:
                    N[(k, p)] = torch.zeros_like(Sf); D[(k, p)] = torch.zeros(Sf.shape, device=dev, dtype=torch.float64)
                N[(k, p)] += Sf.conj() * Tf; D[(k, p)] += Sf.abs() ** 2
        print(f"fit ev{ev:03d} ({time.time() - t0:.0f}s)", flush=True)
    H = {kp: N[kp] / D[kp].clamp_min(1e-30) for kp in N}
    del N, D
    kern = {}
    for p in range(3):
        nw, nt = nshape[p]
        R, wo, tzb = load_kernel("UVY"[p], "exact")
        kern[p] = torch.as_tensor(np.fft.rfft2(kernel_full(R, wo, tzb, nw + 2 * PADW, nt + 2 * PADT)), device=dev)

    lo, hi = (int(x) for x in a.test.split("-"))
    E_ = {}                                                      # method -> plane -> (log ratio, truth)
    for ev in range(lo, hi + 1):
        pairs = load(ev)
        for g, (S, T) in pairs.items():
            p = g % 3; nw, nt = nshape[g]; NW, NT = nw + 2 * PADW, nt + 2 * PADT
            Sf = spec(S, nw, nt)
            crop = lambda x: x[PADW:PADW + nw, PADT:PADT + nt]
            Kf = kern[p]; mx = (Kf.abs() ** 2).max()
            est = {f"wiener_{k}": torch.fft.irfft2(Sf * H[(k, p)], s=(NW, NT)) for k in ("half", "all")}
            for lam in (1e-3, 1e-4):
                est[f"tik{lam:g}"] = torch.fft.irfft2(Sf * Kf.conj() / (Kf.abs() ** 2 + lam * mx), s=(NW, NT))
            tc = fine_cells(T)
            m = tc >= 4e3
            for name, x in est.items():
                pc = fine_cells(crop(x.cpu().numpy()))
                r = E_.setdefault(name, {}).setdefault(p, [[], []])
                r[0].append(np.log(np.maximum(pc[m], 1.0) / tc[m])); r[1].append(tc[m])
        print(f"test ev{ev:03d}", flush=True)
    res = dict(fit=a.fit, n_fit_events=n_ok, scores={})
    for name, per in E_.items():
        res["scores"][name] = {}
        for p, (lr, tq) in per.items():
            lr, tq = np.concatenate(lr), np.concatenate(tq)
            for blo, bhi in BINS:
                mm = (tq >= blo) & (tq < bhi)
                a16, a50, a84 = np.percentile(lr[mm], [16, 50, 84])
                res["scores"][name][f"{'UVY'[p]}_{blo:.0f}"] = dict(hw68=float((a84 - a16) / 2), med=float(a50), n=int(mm.sum()))
    json.dump(res, open(a.out, "w"), indent=1)
    for name, s in res["scores"].items():
        print(name, {k: round(v["hw68"], 4) for k, v in s.items()})


if __name__ == "__main__":
    main()
