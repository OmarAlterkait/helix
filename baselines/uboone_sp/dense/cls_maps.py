#!/usr/bin/env python3
"""Full-plane fine-cell maps of the classical (Wire-Cell / MicroBooNE) chain for
the dense-activity comparison, one npz per (event, plane gid).

    scripts/helix_run.sh python baselines/uboone_sp/dense/cls_maps.py --events 260-387 \
        --kernel fit --reuse /pscratch/sd/o/oalter/helix_work/uboone_sp/v2/cls_full \
        --out /pscratch/sd/o/oalter/helix_work/uboone_sp/dense/cls_fit

Everything is computed with the v2 code, imported unchanged (v2/classical.py:
noise stream seed 1000 + ev, digitize, gate, sp.wirecell_sp.decon_outputs,
chain_pre / finish for the ROI chain; v2/classical_extra.py's statistics are
re-implemented here line for line because they are local to its process()).

Maps written (all (ceil(nw/2), ceil(nt/16)) fine cells, readout only):
  default   MicroBooNE default ROI Gaus_wide charge (U q_s1_f0, V/Y q_s1_f1)
  tuned_q   frozen tuned ROI charge (picks from v2/results/benchmark.json)
  tuned_any frozen tuned best detection statistic (U zlf06, V zor06a2, Y = tuned_q)
  oracle    Gaus_wide inside truth ROIs (truth > 0, pad 10 ticks): reference, not a detector
  truth     fine_cells(hits)  (sp.data.Event.truth)
``--reuse DIR`` (only valid with --kernel fit): take default / tuned_q / oracle /
truth from v2's cls_full maps (same code, same kernel, same noise) and compute
only tuned_any; any other kernel recomputes everything, so the whole comparison
can be repeated with the exact kernel (`--kernel exact`) once
response/exact_<P>.npz exists. ``--picks`` overrides the frozen picks (json
{"tuned_q": {"0": ..}, "tuned_any": {..}, "default": {..}}), e.g. a re-tune on
the exact kernel.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
V2 = os.path.join(BASE, "v2")
sys.path.insert(0, V2)
sys.path.insert(0, BASE)
BENCH = os.path.join(V2, "results", "benchmark.json")


def load_picks(path):
    B = json.load(open(path))
    P = B["selection"]["pick"] if "selection" in B else B
    return {f: {int(k): v for k, v in P[f].items()} for f in ("default", "tuned_q", "tuned_any")}


def zmap(x):
    from sp import wirecell_sp as WC
    r = np.array([WC.cal_rms(row) for row in x])
    return x / np.maximum(r, 1e-6)[:, None]


def avg2(x):
    y = x.copy()
    n = (x.shape[0] // 2) * 2
    y[:n] = np.repeat(0.5 * (x[0:n:2] + x[1:n:2]), 2, 0)
    return y


def lf_tau(x, tau):
    from sp import wirecell_sp as WC
    nt = x.shape[1]
    ft = np.abs(np.fft.rfftfreq(nt, d=0.5))
    return np.fft.irfft(np.fft.rfft(x, axis=1) * WC.lf(ft, tau)[None, :], n=nt, axis=1)


def detect_stat(name, d):
    """classical_extra.py's statistics (round 1 zlf04/06/10, round 2 zor06a2 / zlf06a2 / ztlfa2), induction."""
    from classical import cell_reduce
    if name in ("zlf04", "zlf06", "zlf10"):
        return cell_reduce(zmap(lf_tau(d["wiener_tight"], float("0." + name[3:]))), "max")
    if name == "zlf":
        return cell_reduce(zmap(d["tight_lf"]), "max")
    if name in ("zor06a2", "zlf06a2", "ztlfa2"):
        ta, l6a = zmap(avg2(d["tighter_lf"])), zmap(avg2(lf_tau(d["wiener_tight"], 0.06)))
        X = {"zor06a2": np.maximum(ta, l6a), "zlf06a2": l6a, "ztlfa2": ta}[name]
        return cell_reduce(X, "max")
    raise ValueError(name)


def process(job):
    ev, g, out_dir, kern, reuse, picks = job
    out = os.path.join(out_dir, f"ev{ev:03d}_g{g}.npz")
    if os.path.exists(out):
        return ev, g, "exists"
    t0 = time.time()
    from helix.tpc.noise import digitize, generate_noise
    from sp.data import Event, registry, PLANES
    from sp.noise_filter import gate_clean
    from sp.pipeline import Kernel, fine_cells, form_rois_true
    from sp.response import load_kernel
    from sp import wirecell_sp as WC
    from classical import chain_pre, finish

    reg = registry()
    E = Event(ev)
    rng = np.random.default_rng(1000 + ev)
    for gg in range(g + 1):                      # identical noise stream to v2/classical.py
        nw_, nt_ = reg[gg]["n_wires"], reg[gg]["n_ticks"]
        n_inc = generate_noise((nw_, nt_), rng=rng, wire_lengths_m=np.asarray(reg[gg]["wire_lengths"]),
                               incoherent=True, coherent=False)
        n_coh = generate_noise((nw_, nt_), rng=rng, incoherent=False, coherent=True, group_size=64)
    p = g % 3
    pl = PLANES[p]
    C = E.clean(g, reg)
    raw = digitize(C + n_inc + n_coh, reg[g]["pedestal"])
    M = gate_clean(raw)
    K = Kernel(*load_kernel(pl, kern))
    d = WC.decon_outputs(M, pl, K)
    res = {}
    if reuse:
        F = np.load(os.path.join(reuse, f"ev{ev:03d}_g{g}.npz"))
        res["default"] = F[f"full_{picks['default'][p]}"]
        res["tuned_q"] = F[f"full_{picks['tuned_q'][p]}"]
        res["oracle"] = F["full_qor"]
        res["truth"] = F["full_truth"]
    else:
        by_scale = {}
        for fam in ("default", "tuned_q"):
            name = picks[fam][p]
            s, f = (float(x[1:]) for x in name[2:].split("_"))
            if s not in by_scale:
                by_scale[s] = chain_pre(d, pl, dict(th_factor_ind=3.0 * s, th_factor_col=5.0 * s, l_factor=3.5 * s))
            Rf, attr = by_scale[s]
            res[fam] = fine_cells(finish(Rf, attr, f, d["gauss"], pl)[0])
        T = E.truth(g, C.shape)
        rl = [[] for _ in range(C.shape[0])]
        for w, lst in form_rois_true(T, pad_ticks=10).items():
            for s_, e_ in lst:
                r = WC.ROI.__new__(WC.ROI)
                r.wire, r.start, r.end, r.ext_start, r.ext_end = w, s_, e_ - 1, s_, e_ - 1
                rl[w].append(r)
        res["oracle"] = fine_cells(WC.apply_roi(d["gauss"], rl, pl)[0])
        res["truth"] = fine_cells(T)
    ta = picks["tuned_any"][p]
    res["tuned_any"] = res["tuned_q"] if ta == picks["tuned_q"][p] else detect_stat(ta, d)
    res = {k: np.asarray(v, np.float32) for k, v in res.items()}
    np.savez(out + ".tmp.npz", plane=pl, kernel=kern, picks=json.dumps({f: picks[f][p] for f in picks}), **res)
    os.replace(out + ".tmp.npz", out)
    return ev, g, f"{time.time() - t0:.0f}s"


def main():
    sys.path.insert(0, V2)
    from classical import parse_events
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--planes", default="0,1,2,3,4,5")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--kernel", default="fit", choices=("fit", "example", "exact", "exact0"))
    ap.add_argument("--reuse", default="", help="v2 cls_full dir (fit kernel only)")
    ap.add_argument("--picks", default=BENCH)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.reuse and a.kernel != "fit":
        sys.exit("--reuse holds fitted-kernel maps; drop it for another kernel")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.makedirs(a.out, exist_ok=True)
    picks = load_picks(a.picks)
    jobs = [(ev, g, a.out, a.kernel, a.reuse, picks) for ev in parse_events(a.events)
            for g in (int(x) for x in a.planes.split(","))]
    jobs.sort(key=lambda j: (j[1] % 3 == 2, j[0]))
    from multiprocessing import Pool
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for ev, g, msg in pool.imap_unordered(process, jobs):
            print(f"ev{ev:03d} g{g}: {msg}", flush=True)


if __name__ == "__main__":
    main()
