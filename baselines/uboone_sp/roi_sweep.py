#!/usr/bin/env python3
"""Threshold sweep of two MicroBooNE ROI chains, plus the per-deposit ROI decomposition.

    scripts/helix_run.sh python baselines/uboone_sp/roi_sweep.py --events 260-387 --workers 100 \
        --out /pscratch/sd/o/oalter/helix_work/uboone_sp/roi_sweep

One job per (event, plane gid). Input: R2/R3's input (digitize(clean + incoherent +
coherent) -> our gate) with the SAME noise realisation as run_ladder.py (seed
1000 + ev; earlier planes' noise is drawn and discarded so the stream lines up).

Chains and settings (``SETTINGS``):
  ex_*   the example's chain (sp.pipeline / reference roi.py), its filters, with
         its roi_th_factor scaled (3.5 induction / 5 collection x s)
  wc_*   the Wire-Cell port (sp.wirecell_sp) with the MicroBooNE configuration,
         thresholds scaled jointly (tight induction 3, collection 5, loose 3.5
         x s) or independently, and variants: no fake-signal removal, the U-plane
         fake-signal removal applied (Wire-Cell master skips it on U)
  *_oracle  the same chain's filters and charge rule with ROIs from truth
         (the example's form_rois_true: truth > 25 e-, padded 10 ticks)

Stored per job (npz): for every setting, the fine-cell charge at the floor
windows' cells (w_<s>) and the near-activity windows' cells (n_<s>) of this
plane (other planes' cells are 0 -- the scorer sums the six plane files); per
isolated-deposit window on this plane: truth charge in its 32 x 128 box, truth
charge inside the found ROIs, whether any truth pixel lies in a found ROI, and
the reconstructed charge in the box for found and oracle ROIs; per plane: ROI
pixel count and ROI pixels in truth-empty fine cells, positive ROI charge and
positive ROI charge in truth-empty cells.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"

SCALES = (0.5, 0.67, 0.83, 1.0, 1.25, 1.5, 2.0, 3.0)
SETTINGS = [("ex_oracle", None), ("wc_oracle", None)]
SETTINGS += [(f"ex_s{s:g}", dict(scale=s)) for s in SCALES]
SETTINGS += [(f"wc_s{s:g}", dict(th_factor_ind=3.0 * s, th_factor_col=5.0 * s, l_factor=3.5 * s)) for s in SCALES]
SETTINGS += [("wc_tight0.67", dict(th_factor_ind=2.0, th_factor_col=5.0 * 0.67)),
             ("wc_loose0.67", dict(l_factor=3.5 * 0.67)),
             ("wc_tight1.5", dict(th_factor_ind=4.5, th_factor_col=7.5)),
             ("wc_nofake", dict(fake_removal=False)),
             ("wc_fixU", dict(fix_u_cleanup=True)),
             ("wc_fake0.5", dict(fake_low=250.0, fake_high=500.0)),
             ("wc_s0.67_nofake", dict(th_factor_ind=2.0, th_factor_col=5.0 * 0.67, l_factor=3.5 * 0.67, fake_removal=False))]
NAMES = [n for n, _ in SETTINGS]


def box(w0, t0, nw, nt):
    return slice(min(w0, nw), min(w0 + 32, nw)), slice(min(t0, nt), min(t0 + 128, nt))


def process(job):
    ev, g, out_dir, skip = job
    out = os.path.join(out_dir, f"ev{ev:03d}_g{g}.npz")
    if os.path.exists(out):
        return ev, g, "exists"
    t0_ = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from helix.tpc.noise import digitize, generate_noise
    from sp.data import Event, registry, PLANES
    from sp.noise_filter import gate_clean
    from sp.pipeline import (Kernel, deconvolve_2d_dual, apply_lf_filter_to_wiener, fine_cells, lookup, PLANE_PARAMS,
                             form_rois_true)
    from sp.response import load_kernel
    from sp import wirecell_sp as WC
    from uboone_pipeline import roi as RO

    reg = registry()
    E = Event(ev)
    rng = np.random.default_rng(1000 + ev)
    for gg in range(g + 1):                          # keep the noise stream aligned with run_ladder.py
        nw_, nt_ = reg[gg]["n_wires"], reg[gg]["n_ticks"]
        n_inc = generate_noise((nw_, nt_), rng=rng, wire_lengths_m=np.asarray(reg[gg]["wire_lengths"]),
                               incoherent=True, coherent=False)
        n_coh = generate_noise((nw_, nt_), rng=rng, incoherent=False, coherent=True, group_size=64)
    pl = PLANES[g % 3]
    C = E.clean(g, reg)
    T = E.truth(g, C.shape)
    nw, nt = C.shape
    M = gate_clean(digitize(C + n_inc + n_coh, reg[g]["pedestal"]))
    K = Kernel(*load_kernel(pl, "fit"))
    zn = np.load(os.path.join(NEAR, f"ev{ev:03d}.npz"), allow_pickle=True)
    wkey, nkey = E.z["wkey"], zn["wkey"]
    isos = [m for m in E.meta if m["kind"] == "iso" and m["g"] == g]
    Tc = fine_cells(T)
    Tc_full = np.repeat(np.repeat(Tc == 0, 2, 0), 16, 1)[:nw, :nt]   # pixel lies in a truth-empty fine cell
    oracle_rois = form_rois_true(T, pad_ticks=10)

    # shared deconvolutions
    p_ex = dict(PLANE_PARAMS[pl])
    cw, cg = deconvolve_2d_dual(M, pl, K, noiseless=False)
    if pl != "Y":
        cl = apply_lf_filter_to_wiener(cw, pl, which="loose")
        ct = apply_lf_filter_to_wiener(cw, pl, which="tight")
    d = WC.decon_outputs(M, pl, K)

    res = {}
    timing = {}
    for name, par in SETTINGS:
        if name in skip:
            continue
        t1 = time.time()
        if name == "ex_oracle":
            q, mask = RO.apply_rois(cg, oracle_rois, pl != "Y")
        elif name == "wc_oracle":
            rl = [[] for _ in range(nw)]
            for w, lst in oracle_rois.items():
                for s, e in lst:
                    r = WC.ROI.__new__(WC.ROI)
                    r.wire, r.start, r.end, r.ext_start, r.ext_end = w, s, e - 1, s, e - 1
                    rl[w].append(r)
            q, mask = WC.apply_roi(d["gauss"], rl, pl)
        elif name.startswith("ex_"):
            p = dict(p_ex); p["roi_th_factor"] = p_ex["roi_th_factor"] * par["scale"]
            if pl == "Y":
                rois = RO.form_rois_collection(cw, p, noiseless=False)
            else:
                rois = RO.form_rois_induction(cw, cl, ct, p, noiseless=False)
            q, mask = RO.apply_rois(cg, rois, pl != "Y")
        else:
            rois, _, _ = WC.roi_chain(d, pl, par)
            q, mask = WC.apply_roi(d["gauss"], rois, pl)
        X = fine_cells(q)
        res[f"w_{name}"] = lookup({g: X}, wkey).astype(np.float32)
        res[f"n_{name}"] = lookup({g: X}, nkey).astype(np.float32)
        dec = []
        for m in isos:
            sl = box(m["w0"], m["t0"], nw, nt)
            Tb, Mb, Qb = T[sl], mask[sl], q[sl]
            dec.append([Tb.sum(), Tb[Mb].sum(), float((Mb & (Tb > 0)).any()), Qb.sum(), np.maximum(Qb, 0).sum()])
        res[f"iso_{name}"] = np.asarray(dec, np.float64).reshape(-1, 5)
        qp = np.maximum(q, 0)
        res[f"pur_{name}"] = np.array([mask.sum(), (mask & Tc_full).sum(), qp.sum(), qp[Tc_full].sum(),
                                       T[mask].sum(), T.sum()], np.float64)
        timing[name] = time.time() - t1
    res["iso_meta"] = np.array([[m["w0"], m["t0"], float(np.sum(m["E"]))] for m in isos], np.float64).reshape(-1, 3)
    tmp = out + ".tmp.npz"
    np.savez(tmp, timing=json.dumps(timing), **res)
    os.replace(tmp, out)
    return ev, g, f"{time.time() - t0_:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="", help="comma-separated settings not to run")
    a = ap.parse_args()
    skip = set(x for x in a.skip.split(",") if x)
    lo, hi = (int(x) for x in a.events.split("-"))
    os.makedirs(a.out, exist_ok=True)
    jobs = [(ev, g, a.out, skip) for ev in range(lo, hi + 1) for g in (0, 1, 3, 4, 2, 5)]   # induction first (slowest)
    jobs.sort(key=lambda j: (j[1] % 3 == 2, j[0]))
    from multiprocessing import Pool
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for ev, g, msg in pool.imap_unordered(process, jobs):
            print(f"ev{ev:03d} g{g}: {msg}", flush=True)


if __name__ == "__main__":
    main()
