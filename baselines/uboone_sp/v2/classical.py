#!/usr/bin/env python3
"""Tunable classical (Wire-Cell / MicroBooNE) detector: every candidate window
statistic for a grid of ROI settings, on tuning AND test events, one job per
(event, plane gid).

    scripts/helix_run.sh python baselines/uboone_sp/v2/classical.py --events 0-95,260-387 \
        --workers 110 --out /pscratch/sd/o/oalter/helix_work/uboone_sp/v2/cls

Input = R2/R3's (digitize(clean + incoherent + coherent) -> our gate), the same
noise stream as run_ladder.py / roi_sweep.py (seed 1000 + ev). ``--input corpus``
instead thresholds the gated coefficients like the corpus (kappa 1, per-band
universal threshold, D1 dropped) -- the model's information.

Decon = sp.wirecell_sp.decon_outputs (Wire-Cell filters, zero-padded FFT).

Statistics, each a fine-cell (2 wires x 16 ticks) map; a window scores the max
over its cells, exactly as eval_denoise does with the model's charge:
  zw      Wiener_tight / per-wire RMS (cal_rms), max over the cell's pixels
          (the statistic Wire-Cell's collection ROI finder thresholds)
  zlf     tight-LF Wiener / per-wire RMS, cell max (induction ROI statistic; Y = zw)
  zg      Gaus_wide summed over the cell / robust per-fine-wire sigma of cell sums
  zg3     zg on the 3x3-cell box sum (6 wires x 48 ticks)
  zglf, zg3lf  the same on Gaus_wide x the tight LF filter (induction; Y aliases)
  g       Gaus_wide cell sum, clipped at 0 (no ROI: R2nrp-like charge)
  For every ROI setting s{scale}_f{fake}:
  q_<set>   ROI-masked Gaus_wide charge (Wire-Cell apply_roi), cell sum clipped at 0
  zr_<set>  (zlf for induction / zw for collection) inside ROI cells, 0 elsewhere
  ROI grid: thresholds (th_factor_ind 3, th_factor_col 5, loose l_factor 3.5) x
  scale; fake-signal cut (bin > 1000 e- or mean > 500 e-) x fake (0 = off),
  applied on U too (Wire-Cell master never removes on U: master-U == fake 0).
  MicroBooNE default = s1_f1 on V/Y, s1_f0 on U.
  qor       Gaus_wide with oracle ROIs (truth > 25 e-, pad 10), reference only.
Stored: every statistic at the floor windows' cells (w_*) and the near-activity
windows' cells (n_*) of this plane; with ``--full`` also full-plane cell maps
of the statistics named by ``--full`` (test-event aspects).
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"

SCALES_IND = (0.67, 0.83, 1.0, 1.25, 1.5)
SCALES_COL = (0.4, 0.5, 0.67, 0.83, 1.0, 1.25, 1.5)
FAKES = (0.0, 0.25, 0.5, 1.0)


def cell_reduce(X, op, FW=2, FT=16):
    nw, nt = X.shape
    pw, pt = (-nw) % FW, (-nt) % FT
    fill = -np.inf if op == "max" else 0.0
    Y = np.pad(X.astype(np.float64), ((0, pw), (0, pt)), constant_values=fill)
    Y = Y.reshape(Y.shape[0] // FW, FW, Y.shape[1] // FT, FT)
    return Y.max((1, 3)) if op == "max" else Y.sum((1, 3))


def row_sigma(C):
    """Robust per-row sigma (Wire-Cell cal_rms) of a cell map."""
    from sp.wirecell_sp import cal_rms
    s = np.array([cal_rms(r) for r in C])
    return np.where(s > 0, s, np.median(s[s > 0]) if (s > 0).any() else 1.0)


def box3(C):
    from scipy.ndimage import uniform_filter
    return uniform_filter(C, size=3, mode="constant") * 9.0


def corpus_like(img):
    """Gate -> corpus threshold (kappa 1, universal, per band) -> drop D1 -> image."""
    from helix.core import backend
    backend.set_backend("numpy")
    from helix.core.wavelet import ThresholdSpec, threshold_bands, wavedec
    from helix.core.wavelet_ops_numpy import reconstruct
    from helix.tpc.coherent_gate import coherent_gate
    from helix.tpc.config import DetectorConfig
    cfg = DetectorConfig()
    nt = img.shape[1]
    x = np.pad(np.asarray(img, np.float32), ((0, 0), (0, (-nt) % (1 << cfg.dwt_level))))
    co, lev = wavedec(x, wavelet=cfg.wavelet, level=cfg.dwt_level, mode=cfg.dwt_mode)
    g = [np.asarray(b) for b in coherent_gate(co, group_size=cfg.group_size, kgate=cfg.gate_kgate, ksig=cfg.gate_ksig,
                                              npass=cfg.gate_npass, tau=cfg.gate_tau, gate_approx=True)]
    spec = ThresholdSpec(method="universal", func=cfg.threshold_mode, scale=1.0, per_band_sigma=True,
                         threshold_approx=cfg.threshold_include_approx)
    t = [np.asarray(b) for b in threshold_bands(g, spec)[0]]
    t[-1] = np.zeros_like(t[-1])
    return np.asarray(reconstruct(t, cfg.wavelet, lev, cfg.dwt_mode, nt), np.float32)


def chain_pre(d, pl, P):
    """sp.wirecell_sp.roi_chain up to (not including) the fake-signal cleanup."""
    from sp import wirecell_sp as WC
    P = dict(WC.DEFAULTS, **P)
    nw = d["wiener_tight"].shape[0]
    if pl == "Y":
        tight, rms = WC.find_tight(d["wiener_tight"], d["wiener_tight"], P["th_factor_col"], P["pad"])
        tight = WC.connect_info(tight, P["asy"])
        Rf = WC.Refiner(pl, nw, rms, P)
        Rf.load(tight, None, d["wiener_tight"])
        return Rf, "tight"
    tight, rms = WC.find_tight(d["tight_lf"], d["tighter_lf"], P["th_factor_ind"], P["pad"])
    tight = WC.connect_info(tight, P["asy"])
    loose = WC.extend_loose(WC.find_loose(d["loose_lf"], P), tight)
    Rf = WC.Refiner(pl, nw, rms, P)
    Rf.load(tight, loose, d["wiener_tight"])
    Rf.cleanup(); Rf.generate_merge()
    for _ in range(P["r_break_roi_loop"]):
        Rf.break_all(); Rf.check(); Rf.cleanup()
    Rf.shrink_all(); Rf.check(); Rf.cleanup()
    return Rf, "loose"


def finish(Rf, attr, fake, gauss, pl):
    from sp import wirecell_sp as WC
    R2 = copy.deepcopy(Rf)
    lists = getattr(R2, attr)
    if fake > 0:
        R2.P = dict(R2.P, fake_low=500.0 * fake, fake_high=1000.0 * fake, fix_u_cleanup=True)
        R2.fake_cleanup(lists)
    R2.extend(lists)
    return WC.apply_roi(gauss, lists, pl)


def setting_names(pl):
    sc = SCALES_COL if pl == "Y" else SCALES_IND
    return [f"s{s:g}_f{f:g}" for s in sc for f in FAKES]


def process(job):
    ev, g, out_dir, inp, full, kern, only = job
    out = os.path.join(out_dir, f"ev{ev:03d}_g{g}.npz")
    if os.path.exists(out):
        return ev, g, "exists"
    t00 = time.time()
    from helix.tpc.noise import digitize, generate_noise
    from sp.data import Event, registry, PLANES
    from sp.noise_filter import gate_clean
    from sp.pipeline import Kernel, lookup, fine_cells, form_rois_true
    from sp.response import load_kernel
    from sp import wirecell_sp as WC

    reg = registry()
    E = Event(ev)
    rng = np.random.default_rng(1000 + ev)
    for gg in range(g + 1):                          # noise stream aligned with run_ladder.py / roi_sweep.py
        nw_, nt_ = reg[gg]["n_wires"], reg[gg]["n_ticks"]
        n_inc = generate_noise((nw_, nt_), rng=rng, wire_lengths_m=np.asarray(reg[gg]["wire_lengths"]),
                               incoherent=True, coherent=False)
        n_coh = generate_noise((nw_, nt_), rng=rng, incoherent=False, coherent=True, group_size=64)
    pl = PLANES[g % 3]
    C = E.clean(g, reg)
    K = Kernel(*load_kernel(pl, kern))
    if inp in ("gate", "corpus"):
        raw = digitize(C + n_inc + n_coh, reg[g]["pedestal"])
        M = gate_clean(raw) if inp == "gate" else corpus_like(raw)
    elif inp == "inc":                                   # R1: coherent noise removed perfectly
        M = digitize(C + n_inc, reg[g]["pedestal"])
    elif inp == "clean":                                 # R0: the stored clean sensor
        M = C
    else:                                                # R0syn: hits (*) this kernel, float, linear (zero-padded)
        from sp.response import forward
        Tz = np.pad(E.truth(g, C.shape).astype(np.float64), ((0, 0), (400, 400)))
        M = forward(Tz, K.R, K.wo, K.tzb)[:, 400:-400].astype(np.float32)
    d = WC.decon_outputs(M, pl, K)
    zn = np.load(os.path.join(NEAR, f"ev{ev:03d}.npz"), allow_pickle=True)
    wkey, nkey = E.z["wkey"], zn["wkey"]
    maps = {}

    rms_w = np.array([WC.cal_rms(r) for r in d["wiener_tight"]])
    maps["zw"] = cell_reduce(d["wiener_tight"] / np.maximum(rms_w, 1e-6)[:, None], "max")
    if pl != "Y":
        rms_l = np.array([WC.cal_rms(r) for r in d["tight_lf"]])
        maps["zlf"] = cell_reduce(d["tight_lf"] / np.maximum(rms_l, 1e-6)[:, None], "max")
    else:
        maps["zlf"] = maps["zw"]
    Gs = cell_reduce(d["gauss"], "sum")
    maps["g"] = np.maximum(Gs, 0)
    maps["zg"] = Gs / row_sigma(Gs)[:, None]
    B3 = box3(Gs)
    maps["zg3"] = B3 / row_sigma(B3)[:, None]
    if pl != "Y":
        nt = d["gauss"].shape[1]
        ft = np.abs(np.fft.rfftfreq(nt, d=0.5))
        glf = np.fft.irfft(np.fft.rfft(d["gauss"], axis=1) * WC.lf(ft, WC.LF_TIGHT)[None, :], n=nt, axis=1)
        Gl = cell_reduce(glf, "sum")
        maps["zglf"] = Gl / row_sigma(Gl)[:, None]
        B3l = box3(Gl)
        maps["zg3lf"] = B3l / row_sigma(B3l)[:, None]
    else:
        maps["zglf"], maps["zg3lf"] = maps["zg"], maps["zg3"]
    zsel = maps["zlf"] if pl != "Y" else maps["zw"]

    timing = {}
    by_scale = {}
    for name in [n for n in setting_names(pl) if not only or n in only]:
        s, f = (float(x[1:]) for x in name.split("_"))
        t1 = time.time()
        if s not in by_scale:
            P = dict(th_factor_ind=3.0 * s, th_factor_col=5.0 * s, l_factor=3.5 * s)
            by_scale[s] = chain_pre(d, pl, P)
        Rf, attr = by_scale[s]
        q, mask = finish(Rf, attr, f, d["gauss"], pl)
        maps[f"q_{name}"] = fine_cells(q)
        mc = cell_reduce(mask.astype(np.float64), "max") > 0
        maps[f"zr_{name}"] = np.where(mc, zsel, 0.0)
        timing[name] = time.time() - t1
    T = E.truth(g, C.shape)
    rl = [[] for _ in range(C.shape[0])]
    for w, lst in form_rois_true(T, pad_ticks=10).items():
        for s_, e_ in lst:
            r = WC.ROI.__new__(WC.ROI)
            r.wire, r.start, r.end, r.ext_start, r.ext_end = w, s_, e_ - 1, s_, e_ - 1
            rl[w].append(r)
    maps["qor"] = fine_cells(WC.apply_roi(d["gauss"], rl, pl)[0])

    res = {}
    for k, X in maps.items():
        res[f"w_{k}"] = lookup({g: X}, wkey).astype(np.float32)
        res[f"n_{k}"] = lookup({g: X}, nkey).astype(np.float32)
    if full:
        keep = [k for k in maps if k in full or full == ["all"]]
        for k in keep:
            res[f"full_{k}"] = maps[k].astype(np.float32)
        res["full_truth"] = fine_cells(T).astype(np.float32)
    tmp = out + ".tmp.npz"
    np.savez(tmp, timing=json.dumps(timing), plane=pl, **res)
    os.replace(tmp, out)
    return ev, g, f"{time.time() - t00:.0f}s"


def parse_events(s):
    evs = []
    for part in s.split(","):
        lo, hi = (int(x) for x in part.split("-"))
        evs += list(range(lo, hi + 1))
    return evs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="0-95,260-387")
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input", default="gate", choices=("gate", "corpus", "inc", "clean", "syn"),
                    help="gate = R2/R3 input; corpus = the model's information; inc = R1; clean = R0; syn = R0syn")
    ap.add_argument("--kernel", default="fit", choices=("fit", "example", "exact"))
    ap.add_argument("--planes", default="0,1,2,3,4,5", help="plane gids to run")
    ap.add_argument("--settings", default="", help="comma-separated ROI settings to run (default: the whole grid)")
    ap.add_argument("--full", default="", help="comma-separated statistics to store as full-plane cell maps ('all')")
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.makedirs(a.out, exist_ok=True)
    full = [x for x in a.full.split(",") if x]
    only = set(x for x in a.settings.split(",") if x)
    jobs = [(ev, g, a.out, a.input, full, a.kernel, only) for ev in parse_events(a.events)
            for g in (int(x) for x in a.planes.split(","))]
    jobs.sort(key=lambda j: (j[1] % 3 == 2, j[0]))                 # induction first (slowest)
    from multiprocessing import Pool
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for ev, g, msg in pool.imap_unordered(process, jobs):
            print(f"ev{ev:03d} g{g}: {msg}", flush=True)


if __name__ == "__main__":
    main()
