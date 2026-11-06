#!/usr/bin/env python3
"""More no-ROI significance statistics inside the Wire-Cell filter family (the
induction planes are where the tuned chain is weakest), same input / noise /
decon as classical.py; window-cell lookups only, merged into classical.py's files.

    scripts/helix_run.sh python baselines/uboone_sp/v2/classical_extra.py --events 0-95,260-387 \
        --workers 100 --out /pscratch/sd/o/oalter/helix_work/uboone_sp/v2/cls_extra

  ztlf        tighter-LF (tau 0.1 MHz) Wiener / per-wire RMS, cell max
  zor         max(zlf, ztlf): the OR Wire-Cell's tight-ROI finder applies
  zlf04/06/10 Wiener x LF(tau = 0.04 / 0.06 / 0.10 MHz) -- a scan of the low-frequency cut
  zlfa2, zora2  zlf / zor after averaging the cell's two wires pixel by pixel
                (normalised by the RMS of the average)
  round 2 (--round 2; after round 1 showed V gains from wire averaging + a higher LF cut):
  ztlfa2, zlf06a2, zor06a2 (= max of the two), zora4 (zor on a 4-wire mean: the cell's
  two wires and one neighbour each side), zor06a4
On Y (unipolar, no LF filter in Wire-Cell) every name aliases the Wiener z (zw)
and its 2-wire average (zwa2).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def process(job):
    ev, g, out_dir, inp, kern, rnd = job
    out = os.path.join(out_dir, f"ev{ev:03d}_g{g}.npz")         # merged INTO classical.py's file
    if ("w_zor" if rnd == 1 else "w_zor06a4") in np.load(out).files:
        return ev, g, "exists"
    t0 = time.time()
    from helix.tpc.noise import digitize, generate_noise
    from sp.data import Event, registry, PLANES
    from sp.noise_filter import gate_clean
    from sp.pipeline import Kernel, lookup
    from sp.response import load_kernel
    from sp import wirecell_sp as WC
    from classical import cell_reduce, NEAR, corpus_like

    reg = registry()
    E = Event(ev)
    rng = np.random.default_rng(1000 + ev)
    for gg in range(g + 1):
        nw_, nt_ = reg[gg]["n_wires"], reg[gg]["n_ticks"]
        n_inc = generate_noise((nw_, nt_), rng=rng, wire_lengths_m=np.asarray(reg[gg]["wire_lengths"]),
                               incoherent=True, coherent=False)
        n_coh = generate_noise((nw_, nt_), rng=rng, incoherent=False, coherent=True, group_size=64)
    pl = PLANES[g % 3]
    C = E.clean(g, reg)
    raw = digitize(C + n_inc + n_coh, reg[g]["pedestal"])
    M = gate_clean(raw) if inp == "gate" else corpus_like(raw)
    K = Kernel(*load_kernel(pl, kern))
    d = WC.decon_outputs(M, pl, K)

    def z(x):
        r = np.array([WC.cal_rms(row) for row in x])
        return x / np.maximum(r, 1e-6)[:, None]

    def avg2(x):
        y = x.copy()
        n = (x.shape[0] // 2) * 2
        y[:n] = np.repeat(0.5 * (x[0:n:2] + x[1:n:2]), 2, 0)
        return y

    def avg4(x):
        n = (x.shape[0] // 2) * 2
        p = np.pad(x, ((1, 1), (0, 0)), mode="edge")
        m = 0.25 * (p[0:n:2] + p[1:n + 1:2] + p[2:n + 2:2] + p[3:n + 3:2])
        y = x.copy(); y[:n] = np.repeat(m, 2, 0)
        return y

    maps = {}
    if rnd == 2:
        if pl == "Y":
            zwa = cell_reduce(z(avg2(d["wiener_tight"])), "max"); zw4 = cell_reduce(z(avg4(d["wiener_tight"])), "max")
            maps = dict(ztlfa2=zwa, zlf06a2=zwa, zor06a2=zwa, zora4=zw4, zor06a4=zw4)
        else:
            nt = d["wiener_tight"].shape[1]
            ft = np.abs(np.fft.rfftfreq(nt, d=0.5))
            l06 = np.fft.irfft(np.fft.rfft(d["wiener_tight"], axis=1) * WC.lf(ft, 0.06)[None, :], n=nt, axis=1)
            ta, l6a = z(avg2(d["tighter_lf"])), z(avg2(l06))
            maps["ztlfa2"] = cell_reduce(ta, "max")
            maps["zlf06a2"] = cell_reduce(l6a, "max")
            maps["zor06a2"] = cell_reduce(np.maximum(ta, l6a), "max")
            maps["zora4"] = cell_reduce(np.maximum(z(avg4(d["tight_lf"])), z(avg4(d["tighter_lf"]))), "max")
            maps["zor06a4"] = cell_reduce(np.maximum(z(avg4(l06)), z(avg4(d["tighter_lf"]))), "max")
    elif pl == "Y":
        zw = z(d["wiener_tight"])
        zwa = z(avg2(d["wiener_tight"]))
        for k in ("ztlf", "zor", "zlf04", "zlf06", "zlf10"):
            maps[k] = cell_reduce(zw, "max")
        for k in ("zlfa2", "zora2"):
            maps[k] = cell_reduce(zwa, "max")
    else:
        zl, zt = z(d["tight_lf"]), z(d["tighter_lf"])
        maps["ztlf"] = cell_reduce(zt, "max")
        maps["zor"] = cell_reduce(np.maximum(zl, zt), "max")
        nt = d["wiener_tight"].shape[1]
        ft = np.abs(np.fft.rfftfreq(nt, d=0.5))
        Wf = np.fft.rfft(d["wiener_tight"], axis=1)
        for tau, k in ((0.04, "zlf04"), (0.06, "zlf06"), (0.10, "zlf10")):
            maps[k] = cell_reduce(z(np.fft.irfft(Wf * WC.lf(ft, tau)[None, :], n=nt, axis=1)), "max")
        za, zta = z(avg2(d["tight_lf"])), z(avg2(d["tighter_lf"]))
        maps["zlfa2"] = cell_reduce(za, "max")
        maps["zora2"] = cell_reduce(np.maximum(za, zta), "max")
    zn = np.load(os.path.join(NEAR, f"ev{ev:03d}.npz"), allow_pickle=True)
    res = dict(np.load(out))
    for k, X in maps.items():
        res[f"w_{k}"] = lookup({g: X}, E.z["wkey"]).astype(np.float32)
        res[f"n_{k}"] = lookup({g: X}, zn["wkey"]).astype(np.float32)
    np.savez(out + ".tmp.npz", **res)
    os.replace(out + ".tmp.npz", out)
    return ev, g, f"{time.time() - t0:.0f}s"


def main():
    from classical import parse_events
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="0-95,260-387")
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--out", required=True, help="a classical.py output dir (same --input / --kernel); keys are merged in")
    ap.add_argument("--input", default="gate", choices=("gate", "corpus"))
    ap.add_argument("--kernel", default="fit", choices=("fit", "example", "exact"))
    ap.add_argument("--round", type=int, default=1, choices=(1, 2))
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.makedirs(a.out, exist_ok=True)
    jobs = [(ev, g, a.out, a.input, a.kernel, a.round) for ev in parse_events(a.events) for g in range(6)]
    from multiprocessing import Pool
    with Pool(a.workers, maxtasksperchild=8) as pool:
        for ev, g, msg in pool.imap_unordered(process, jobs):
            print(f"ev{ev:03d} g{g}: {msg}", flush=True)


if __name__ == "__main__":
    main()
