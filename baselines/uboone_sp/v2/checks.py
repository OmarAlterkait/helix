#!/usr/bin/env python3
"""Step-level checks of the Wire-Cell port and of the inputs it relies on, on
TUNING events only (ev000-ev259 are never scored).

    scripts/helix_run.sh python baselines/uboone_sp/v2/checks.py --events 0-3 --out baselines/uboone_sp/v2/checks.json

  kernel     energy of the fitted kernel in its outermost wires / ticks (is the
             21 x 211 support big enough?)
  drift      ||clean - store(hits (*) R)|| / ||clean|| in thirds of the readout
             (does the response depend on drift time? the kernel is translation invariant)
  align      noise-free Wire-Cell decon (Gaus_wide, no ROI) of the stored clean
             sensor vs the hits truth: lag of the peak pixel cross-correlation
             (wires, ticks) and the best fine-cell lag (cells), per plane
  point      a 10k e- point charge (delta, and with 2-tick longitudinal diffusion)
             through the fitted kernel and the port: time smearing on the
             central wire, per-wire charge fractions, total recovery (paper I
             sec. 5.2: 2.7 / 2.3 ticks incl. ~2 ticks of 1 m diffusion; >60% on the closest wire)
  noise      noise-only input -> gate -> decon: per-wire ROI RMS (Wiener_tight for
             Y, tight-LF for U/V) in e-/tick, and the std of a 2x16 Gaus_wide cell sum
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)


def xcorr_peak(A, B, mw=3, mt=24):
    """Lag (dw, dt) maximising sum A[w, t] * B[w + dw, t + dt]."""
    FA, FB = np.fft.rfft2(A), np.fft.rfft2(B)
    cc = np.fft.irfft2(FA.conj() * FB, s=A.shape)
    W = np.arange(-mw, mw + 1); T = np.arange(-mt, mt + 1)
    sub = cc[np.ix_(W % A.shape[0], T % A.shape[1])]
    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    # sub-tick refinement (parabola)
    dt = float(T[j])
    if 0 < j < len(T) - 1:
        y0, y1, y2 = sub[i, j - 1], sub[i, j], sub[i, j + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            dt += 0.5 * (y0 - y2) / den
    return int(W[i]), dt


def per_event(ev):
    from helix.tpc.noise import digitize, generate_noise
    from sp.data import Event, registry, PLANES
    from sp.noise_filter import gate_clean
    from sp.pipeline import Kernel, fine_cells
    from sp.response import load_kernel, forward, store_like_sensor
    from sp import wirecell_sp as WC
    reg = registry()
    E = Event(ev)
    rng = np.random.default_rng(777 + ev)
    out = {}
    for g in (0, 1, 2):
        pl = PLANES[g]
        C = E.clean(g, reg); T = E.truth(g, C.shape)
        R, wo, tzb = load_kernel(pl, "fit")
        K = Kernel(R, wo, tzb)
        PADT = 400                                      # zero-padded (linear) forward model: no cyclic wrap
        S = store_like_sensor(forward(np.pad(T, ((0, 0), (PADT, PADT))), R, wo, tzb)[:, PADT:-PADT])
        nt = C.shape[1]
        thirds = [slice(0, nt // 3), slice(nt // 3, 2 * nt // 3), slice(2 * nt // 3, nt)]
        drift = [float(np.linalg.norm((C - S)[:, s]) / max(np.linalg.norm(C[:, s]), 1e-9)) for s in thirds]
        d = WC.decon_outputs(C, pl, K)
        G = d["gauss"]
        lag_pix = xcorr_peak(T.astype(np.float64), G)
        Tc, Gc = fine_cells(T), np.maximum(fine_cells(G), 0)
        best, bl = -2, None
        for dw in (-1, 0, 1):
            for dt in (-1, 0, 1):
                Gs = np.roll(np.roll(Gc, -dw, 0), -dt, 1)
                r = np.corrcoef(Tc.ravel(), Gs.ravel())[0, 1]
                if r > best:
                    best, bl = r, (dw, dt)
        r0 = float(np.corrcoef(Tc.ravel(), Gc.ravel())[0, 1])
        # noise only
        L = np.asarray(reg[g]["wire_lengths"])
        n_inc = generate_noise(C.shape, rng=rng, wire_lengths_m=L, incoherent=True, coherent=False)
        n_coh = generate_noise(C.shape, rng=rng, incoherent=False, coherent=True, group_size=64)
        Mn = gate_clean(digitize(n_inc + n_coh, reg[g]["pedestal"]))
        dn = WC.decon_outputs(Mn, pl, K)
        roi_sig = dn["wiener_tight"] if pl == "Y" else dn["tight_lf"]
        rms = np.array([WC.cal_rms(r) for r in roi_sig])
        cs = _cell_sum(dn["gauss"])
        out[pl] = dict(drift_resid_thirds=drift, align_pixel_lag_w_t=lag_pix, align_cell_best_lag=bl,
                       align_cell_r_best=float(best), align_cell_r_zero=r0,
                       noise_roi_rms_e_per_tick_median=float(np.median(rms)),
                       noise_roi_rms_e_per_tick_p16_p84=[float(np.percentile(rms, 16)), float(np.percentile(rms, 84))],
                       noise_threshold_e_per_tick=float(np.median(rms)) * (5.0 if pl == "Y" else 3.0),
                       noise_gauss_cell_std=float(np.std(cs[:, 5:-5])),
                       noise_raw_adc_rms=float(np.std(Mn)))
    return ev, out


def _cell_sum(X, FW=2, FT=16):
    nw, nt = X.shape
    Y = np.pad(X, ((0, (-nw) % FW), (0, (-nt) % FT)))
    return Y.reshape(Y.shape[0] // FW, FW, Y.shape[1] // FT, FT).sum((1, 3))


def kernel_and_point():
    from sp.pipeline import Kernel
    from sp.response import load_kernel, forward
    from sp import wirecell_sp as WC
    res = {}
    for pl in "UVY":
        R, wo, tzb = load_kernel(pl, "fit")
        e = R ** 2
        tot = e.sum()
        r = dict(edge_wire_frac=float((e[:, 0].sum() + e[:, -1].sum()) / tot),
                 outer2_wire_frac=float((e[:, :2].sum() + e[:, -2:].sum()) / tot),
                 first10_tick_frac=float(e[:10].sum() / tot), last10_tick_frac=float(e[-10:].sum() / tot))
        nw, nt = (1969 if pl != "Y" else 1443), 4321
        K = Kernel(R, wo, tzb)
        for lab, sig_t in (("delta", 0.0), ("diff2tick", 2.0)):
            H = np.zeros((nw, nt))
            w0, t0 = nw // 2, 2000
            if sig_t > 0:
                tt = np.arange(nt) - t0
                prof = np.exp(-0.5 * (tt / sig_t) ** 2); prof /= prof.sum()
                H[w0] = 1e4 * prof
            else:
                H[w0, t0] = 1e4
            M = forward(H, R, wo, tzb)
            G = WC.decon_outputs(M, pl, K)["gauss"]
            row = G[w0, t0 - 60:t0 + 60]
            p = np.maximum(row, 0)
            tt = np.arange(-60, 60)
            mu = (p * tt).sum() / p.sum()
            sd = float(np.sqrt((p * (tt - mu) ** 2).sum() / p.sum()))
            wsum = G[w0 - 4:w0 + 5, t0 - 60:t0 + 60].sum(1)
            r[f"point_{lab}"] = dict(time_sigma_ticks=sd, time_mean_offset_ticks=float(mu),
                                     wire_frac=[round(float(x / 1e4), 3) for x in wsum],
                                     total_recovery=float(G[:, t0 - 200:t0 + 200].sum() / 1e4))
        res[pl] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="0-3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.events.split("-"))
    from multiprocessing import Pool
    with Pool(hi - lo + 2) as pool:
        kp = pool.apply_async(kernel_and_point)
        evs = dict(pool.map(per_event, range(lo, hi + 1)))
        kp = kp.get()
    agg = {}
    for pl in "UVY":
        rows = [evs[e][pl] for e in evs]
        agg[pl] = {k: (np.mean([r[k] for r in rows], 0).tolist() if not isinstance(rows[0][k], tuple) else [r[k] for r in rows])
                   for k in rows[0]}
    json.dump(dict(kernel_point=kp, per_event=evs, mean=agg), open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(kernel_point=kp, mean=agg), indent=1, default=float))


if __name__ == "__main__":
    main()
