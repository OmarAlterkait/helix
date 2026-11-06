#!/usr/bin/env python3
"""The high-charge accuracy FLOOR: least-squares inversion of a sensor image with the exact
(JAXTPC-derived, drift-averaged, per-wire) response kernel, on GPU.

    CUDA_VISIBLE_DEVICES=3 scripts/helix_run.sh python baselines/uboone_sp/causes/floor.py --events 260-291 \
        --cells /global/cfs/cdirs/m5238/users/oalter/helix_runs/causes/cells \
        --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/bright/floor

Inputs per plane (pixel grid wires x ticks):
  syn     Tf_pix (*) R_exact, float: a sensor that the per-wire kernel describes EXACTLY
          (no sub-wire / drift information missing) -- the inversion's own limit
  clean   the stored clean sensor (integer ADC, |v| < 2 dropped, clipped at the ADC range)
  noisy   digitize(clean + incoherent + coherent noise; v2 stream seed 1000 + ev) -> our gate
  corpus  the corpus coefficients the model reads (bands A4-D2, D1 dropped), inverse DWT
Inversions:
  oracle  x supported on the complete-truth pixels (Tf_pix >= 50 e-, from causes/build.py's rebuild),
          min ||S - R (*) x||^2 by conjugate gradients on the normal equations (300 iterations)
  tik     no support knowledge: Fourier Tikhonov x = conj(R) S / (|R|^2 + lam * max|R|^2), lam in
          {1e-4, 1e-3, 1e-2}  (all three stored; the choice is made on calibration events)
Output per event: fine-cell maps (2 x 16) of every (input, inversion) and the complete truth.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "v2")); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(BASE)))
LAMS = (1e-4, 1e-3, 1e-2)
SUPP_MIN = 50.0          # oracle support: complete-truth pixels holding >= 50 e- (tails below carry < 1% of the charge)


def corpus_image(ce, bl, g, nw, nt):
    from helix.core.wavelet_ops_numpy import reconstruct
    from helix.tpc.config import DetectorConfig
    dc = DetectorConfig()
    bands = [np.zeros((nw, int(L)), np.float64) for L in bl]
    m = ce.plane_gid == g
    for b in range(len(bl)):
        mb = m & (ce.band == b)
        if b >= 4:                                   # D1: the model does not read it
            continue
        bands[b][ce.wire[mb].astype(int), ce.tau[mb].astype(int)] = ce.value[mb]
    return np.asarray(reconstruct(bands, dc.wavelet, dc.dwt_level, dc.dwt_mode, nt), np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--inputs", default="syn,clean,noisy,corpus")
    a = ap.parse_args()
    import h5py
    import torch
    import yaml
    from helix.tpc.noise import digitize, generate_noise
    from helix.core.coeff_io import read_coeff_event
    from sp.data import Event, registry, PLANES
    from sp.noise_filter import gate_clean
    from sp.pipeline import fine_cells
    from sp.response import load_kernel, kernel_full
    import build as BD
    import resim as RS
    sys.path.insert(0, RS.JAXTPC)
    from tools.config import create_sim_config
    cfg = create_sim_config(yaml.safe_load(open(os.path.join(RS.JAXTPC, "config", "cubic_wireplane_config.yaml"))),
                            total_pad=1000, include_digitize=True, num_s=16)
    dev = torch.device("cuda")
    reg = registry()
    inputs = a.inputs.split(",")
    os.makedirs(a.out, exist_ok=True)
    lo, hi = (int(x) for x in a.events.split("-"))
    PADW, PADT = 24, 400
    for ev in range(lo, hi + 1):
        out = os.path.join(a.out, f"ev{ev:03d}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        E = Event(ev)
        nshape = {g: (reg[g]["n_wires"], reg[g]["n_ticks"]) for g in range(6)}
        Tfull, _, _ = BD.rebuild(ev, E, cfg, nshape)
        shard = E.shard.replace(BD.PS_CORPUS, BD.CFS_CORPUS)
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], E.event))
            bl = fh["config"]["band_lengths"][:]
        ce = read_coeff_event(shard, pos)
        rng = np.random.default_rng(1000 + ev)
        res = {}
        for g in range(6):
            nw, nt = nshape[g]
            n_inc = generate_noise((nw, nt), rng=rng, wire_lengths_m=np.asarray(reg[g]["wire_lengths"]), incoherent=True, coherent=False)
            n_coh = generate_noise((nw, nt), rng=rng, incoherent=False, coherent=True, group_size=64)
            pl = PLANES[g % 3]
            C = E.clean(g, reg).astype(np.float64)
            Tp = Tfull[g]
            R, wo, tzb = load_kernel(pl, "exact")
            NW, NT = nw + 2 * PADW, nt + 2 * PADT
            Kf = torch.as_tensor(np.fft.rfft2(kernel_full(R, wo, tzb, NW, NT)), device=dev)
            pad = lambda X: np.pad(X, ((PADW, PADW), (PADT, PADT)))
            crop = lambda X: X[PADW:PADW + nw, PADT:PADT + nt]
            fwd = lambda x: torch.fft.irfft2(torch.fft.rfft2(x) * Kf, s=(NW, NT))
            adj = lambda y: torch.fft.irfft2(torch.fft.rfft2(y) * Kf.conj(), s=(NW, NT))
            supp = torch.as_tensor(pad(Tp >= SUPP_MIN), device=dev)
            S = {}
            if "syn" in inputs:
                S["syn"] = crop(fwd(torch.as_tensor(pad(Tp), device=dev)).cpu().numpy())
            if "clean" in inputs:
                S["clean"] = C
            if "noisy" in inputs:
                S["noisy"] = gate_clean(digitize(C.astype(np.float32) + n_inc + n_coh, reg[g]["pedestal"])).astype(np.float64)
            if "corpus" in inputs:
                S["corpus"] = corpus_image(ce, bl, g, nw, nt)
            res[f"truth_g{g}"] = fine_cells(Tp).astype(np.float32)
            res[f"supp_frac_g{g}"] = np.float32(Tp[Tp >= SUPP_MIN].sum() / max(Tp.sum(), 1e-9))
            for k, Sk in S.items():
                y = torch.as_tensor(pad(Sk), device=dev)
                # oracle-support CG on A^T A x = A^T y, x restricted to supp
                b = adj(y) * supp
                x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = (r * r).sum()
                for _ in range(a.iters):
                    Ap = adj(fwd(p * supp)) * supp
                    al = rs / (p * Ap).sum().clamp_min(1e-30)
                    x = x + al * p; r = r - al * Ap
                    rn = (r * r).sum()
                    if rn < 1e-12 * (b * b).sum():
                        break
                    p = r + (rn / rs) * p; rs = rn
                res[f"{k}_oracle_g{g}"] = fine_cells(crop(x.cpu().numpy())).astype(np.float32)
                Y = torch.fft.rfft2(y); mx = (Kf.abs() ** 2).max()
                for lam in LAMS:
                    xt = torch.fft.irfft2(Y * Kf.conj() / (Kf.abs() ** 2 + lam * mx), s=(NW, NT))
                    res[f"{k}_tik{lam:g}_g{g}"] = fine_cells(crop(xt.cpu().numpy())).astype(np.float32)
        np.savez(out + ".tmp.npz", **res)
        os.replace(out + ".tmp.npz", out)
        print(f"ev{ev:03d}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
