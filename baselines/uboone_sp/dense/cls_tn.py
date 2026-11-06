#!/usr/bin/env python3
"""The classical tuned detection statistic (tuned_any) on the SAME fresh torch-GPU
noise draw the model read in model_dump_cn.py --noise tn (planes saved as int16 ADC).

    scripts/helix_run.sh python baselines/uboone_sp/dense/cls_tn.py --events 260-387 \
        --planes-dir /global/cfs/.../causes2/tn_planes --out /global/cfs/.../causes2/cls_tn

Same code path as cls_maps.py (gate_clean -> sp.wirecell_sp.decon_outputs, fitted kernel,
frozen picks); writes per (event, plane) npz with key tuned_any.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "v2")); sys.path.insert(0, BASE); sys.path.insert(0, HERE)


def process(job):
    ev, g, pdir, out_dir, kern, picks = job
    out = os.path.join(out_dir, f"ev{ev:03d}_g{g}.npz")
    if os.path.exists(out):
        return ev, g, "exists"
    t0 = time.time()
    from sp.data import PLANES
    from sp.noise_filter import gate_clean
    from sp.pipeline import Kernel, fine_cells
    from sp.response import load_kernel
    from sp import wirecell_sp as WC
    from classical import chain_pre, finish
    from cls_maps import detect_stat
    p = g % 3
    pl = PLANES[p]
    M = gate_clean(np.load(os.path.join(pdir, f"ev{ev:03d}.npz"))[f"g{g}"].astype(np.float32))
    K = Kernel(*load_kernel(pl, kern))
    d = WC.decon_outputs(M, pl, K)
    name = picks["tuned_any"][p]
    if name.startswith("q_"):
        s, f = (float(x[1:]) for x in name[2:].split("_"))
        Rf, attr = chain_pre(d, pl, dict(th_factor_ind=3.0 * s, th_factor_col=5.0 * s, l_factor=3.5 * s))
        X = fine_cells(finish(Rf, attr, f, d["gauss"], pl)[0])
    else:
        X = detect_stat(name, d)
    np.savez(out + ".tmp.npz", tuned_any=np.asarray(X, np.float32), pick=name)
    os.replace(out + ".tmp.npz", out)
    return ev, g, f"{time.time() - t0:.0f}s"


def main():
    from classical import parse_events
    from cls_maps import load_picks, BENCH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--planes-dir", required=True)
    ap.add_argument("--workers", type=int, default=90)
    ap.add_argument("--kernel", default="fit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.makedirs(a.out, exist_ok=True)
    picks = load_picks(BENCH)
    jobs = [(ev, g, a.planes_dir, a.out, a.kernel, picks) for ev in parse_events(a.events) for g in range(6)
            if os.path.exists(os.path.join(a.planes_dir, f"ev{ev:03d}.npz"))]
    from multiprocessing import Pool
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for ev, g, msg in pool.imap_unordered(process, jobs):
            print(f"ev{ev:03d} g{g}: {msg}", flush=True)


if __name__ == "__main__":
    main()
