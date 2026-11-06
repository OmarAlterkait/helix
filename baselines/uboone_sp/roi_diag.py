#!/usr/bin/env python3
"""Where the example's MicroBooNE ROI chain creates fake ROIs and misses faint deposits.

    scripts/helix_run.sh python baselines/uboone_sp/roi_diag.py --events 260-291 --workers 15 \
        --out /pscratch/sd/o/oalter/helix_work/uboone_sp/roi_diag

Per event, per plane, on R3's input (digitize(clean + incoherent + coherent) ->
our gate, the SAME noise realisation as run_ladder.py, seed 1000 + ev):

  stage accounting  the example's induction chain step by step (raw tight ROIs,
                    raw loose ROIs, tight/loose reconciliation, connectivity,
                    clean, break, shrink) and collection (threshold + pad): for
                    every far (bg), near (bgn) and isolated-deposit (iso) window,
                    whether any ROI pixel of that stage lies inside it.
  noise RMS         the same deconvolution + filters applied to the NOISE ALONE
                    (digitize(incoherent + coherent) -> gate, same realisation):
                    its true per-wire std and robust RMS in each filtered domain
                    (Wiener, tight-LF, loose-LF, Gaussian), against the RMS the
                    example actually thresholds with (robust RMS of the
                    signal+noise waveform, per wire), and the per-sample rate at
                    which noise alone crosses the example's threshold.
  deposit peaks     for each isolated deposit, the peak of its noise-free signal
                    in the thresholded domain (the clean sensor through the same
                    deconvolution and filters, noisy-mode DC) against the
                    threshold on its wire.
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
STAGES_IND = ("tight_raw", "loose_raw", "reconciled", "connect", "clean", "break", "shrink")


def window_flags(mask, wins):
    """For each window (w0, t0), whether any mask pixel lies in its 32 x 128 box."""
    if not wins:
        return []
    c = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)    # 2D prefix sum
    out = []
    nw, nt = mask.shape
    for w0, t0 in wins:
        a, b, s, e = min(w0, nw), min(w0 + 32, nw), min(t0, nt), min(t0 + 128, nt)
        out.append(bool(c[b, e] - c[a, e] - c[b, s] + c[a, s] > 0))
    return out


def rois_to_mask(rois, shape):
    m = np.zeros(shape, bool)
    for w, lst in rois.items():
        for s, e in lst:
            m[w, s:e] = True
    return m


def process_event(job):
    ev, out_dir = job
    out = os.path.join(out_dir, f"ev{ev:03d}.json")
    if os.path.exists(out):
        return ev, "exists"
    t0_ = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from helix.tpc.noise import digitize, generate_noise
    from sp.data import Event, registry, PLANES
    from sp.noise_filter import gate_clean
    from sp.pipeline import Kernel, deconvolve_2d_dual, apply_lf_filter_to_wiener, PLANE_PARAMS
    from sp.response import load_kernel
    from uboone_pipeline import roi as RO

    reg = registry()
    E = Event(ev)
    near = json.loads(str(np.load(os.path.join(NEAR, f"ev{ev:03d}.npz"), allow_pickle=True)["meta"]))
    rng = np.random.default_rng(1000 + ev)
    rec = dict(ev=ev, planes={})
    for g in range(6):
        pl = PLANES[g % 3]
        p = PLANE_PARAMS[pl]
        K = Kernel(*load_kernel(pl, "fit"))
        C = E.clean(g, reg)
        T = E.truth(g, C.shape)
        nw, nt = C.shape
        L = np.asarray(reg[g]["wire_lengths"]); ped = reg[g]["pedestal"]
        # the SAME draws, in the same order, as sp.data.noisy_planes
        n_inc = generate_noise((nw, nt), rng=rng, wire_lengths_m=L, incoherent=True, coherent=False)
        n_coh = generate_noise((nw, nt), rng=rng, incoherent=False, coherent=True, group_size=64)
        M = gate_clean(digitize(C + n_inc + n_coh, ped))
        N = gate_clean(digitize(n_inc + n_coh, ped))                      # noise alone, same realisation

        def domains(X):
            cw, cg = deconvolve_2d_dual(X, pl, K, noiseless=False)
            d = dict(wiener=cw, gauss=cg)
            if pl != "Y":
                d["tight"] = apply_lf_filter_to_wiener(cw, pl, which="tight")
                d["loose"] = apply_lf_filter_to_wiener(cw, pl, which="loose")
            return d
        DM, DN, DC = domains(M), domains(N), domains(C)
        quiet = T.sum(1) == 0                                              # wires with no charge at all
        # --- noise RMS: true (noise-only std), robust on noise-only, and what the example uses (signal+noise)
        rms = {}
        thr_dom = ("tight", "loose") if pl != "Y" else ("wiener",)
        for k in DM:
            used = np.array([RO.compute_rms(DM[k][w]) for w in range(nw)])
            true_std = DN[k].std(1)
            robust_noise = np.array([RO.compute_rms(DN[k][w]) for w in range(nw)])
            r = dict(true_std_median=float(np.median(true_std)), robust_noise_median=float(np.median(robust_noise)),
                     used_median_all=float(np.median(used)), used_median_quiet=float(np.median(used[quiet])) if quiet.any() else None,
                     used_median_busy=float(np.median(used[~quiet])) if (~quiet).any() else None,
                     used_over_true_p10_p50_p90=[float(x) for x in np.percentile(used / np.maximum(true_std, 1e-9), [10, 50, 90])])
            if k in thr_dom:
                thr = p["roi_th_factor"] * used                            # the example's per-wire threshold
                cross = (np.abs(DN[k]) > thr[:, None]).mean()              # per-sample crossing rate, noise alone
                r.update(th_factor=p["roi_th_factor"], threshold_median_e=float(np.median(thr)),
                         threshold_over_true_std=float(np.median(thr / np.maximum(true_std, 1e-9))),
                         noise_crossing_rate_per_sample=float(cross),
                         noise_crossing_rate_gauss_expect=float(2 * 0.5 * __import__("math").erfc(p["roi_th_factor"] / 2 ** 0.5)))
            rms[k] = r
        # --- stage-by-stage ROIs on the noisy input
        stages = {}
        pad = p["roi_pad"]
        if pl != "Y":
            Ct, Cl = DM["tight"], DM["loose"]
            tight_raw, loose_raw = {}, {}
            for w in range(nw):
                t_ = RO.pad_rois(RO.find_rois_1d(Ct[w], RO._get_threshold(Ct[w], p, False)), pad, nt)
                l_ = RO.pad_rois(RO.find_rois_1d(Cl[w], RO._get_threshold(Cl[w], p, False)), pad, nt)
                if t_:
                    tight_raw[w] = t_
                if l_:
                    loose_raw[w] = l_
            tight, loose = RO.form_initial_rois_induction(DM["wiener"], Cl, Ct, p, noiseless=False)
            stages["tight_raw"], stages["loose_raw"], stages["reconciled"] = tight_raw, loose_raw, {w: list(v) for w, v in loose.items()}
            loose = RO.connectivity_propagation(tight, loose, nw, nt); stages["connect"] = {w: list(v) for w, v in loose.items()}
            loose = RO.clean_rois(tight, loose); stages["clean"] = {w: list(v) for w, v in loose.items()}
            loose = RO.break_rois(loose, DM["wiener"], p); stages["break"] = {w: list(v) for w, v in loose.items()}
            loose = RO.shrink_rois(loose, tight, DM["wiener"], p); stages["shrink"] = loose
        else:
            stages["collection"] = RO.form_rois_collection(DM["wiener"], p, noiseless=False)
        wins = {k: [(m["w0"], m["t0"]) for m in ms if m["g"] == g]
                for k, ms in (("bg", [m for m in E.meta if m["kind"] == "bg"]), ("bgn", near),
                              ("iso", [m for m in E.meta if m["kind"] == "iso"]))}
        flags = {}
        for s, rois in stages.items():
            mk = rois_to_mask(rois, (nw, nt))
            flags[s] = {k: window_flags(mk, v) for k, v in wins.items()}
            flags[s]["pixel_frac"] = float(mk.mean())
            flags[s]["n_rois"] = int(sum(len(v) for v in rois.values()))
            flags[s]["n_rois_quiet_wires"] = int(sum(len(v) for w, v in rois.items() if quiet[w]))
        # --- deposit peaks vs threshold in the thresholded domain
        dom = "tight" if pl != "Y" else "wiener"
        used_dom = np.array([RO.compute_rms(DM[dom][w]) for w in range(nw)])
        isos = []
        for m in [m for m in E.meta if m["kind"] == "iso" and m["g"] == g]:
            sl = (slice(m["w0"], min(m["w0"] + 32, nw)), slice(m["t0"], min(m["t0"] + 128, nt)))
            sig = np.abs(DC[dom][sl])
            if not sig.size:
                continue
            iw, it = np.unravel_index(np.argmax(sig), sig.shape)
            w = m["w0"] + iw
            isos.append(dict(w0=m["w0"], t0=m["t0"], E=float(np.sum(m["E"])), q=float(T[sl].sum()),
                             peak=float(sig.max()), thr=float(p["roi_th_factor"] * used_dom[w]),
                             noise_std=float(DN[dom][w].std()),
                             peak_gauss=float(DC["gauss"][sl].max())))
        rec["planes"][str(g)] = dict(rms=rms, flags=flags, wins={k: v for k, v in wins.items()}, isos=isos)
    with open(out + ".tmp", "w") as fh:
        json.dump(rec, fh)
    os.replace(out + ".tmp", out)
    return ev, f"{time.time() - t0_:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-291")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.events.split("-"))
    os.makedirs(a.out, exist_ok=True)
    from multiprocessing import Pool
    with Pool(a.workers, maxtasksperchild=2) as pool:
        for ev, msg in pool.imap_unordered(process_event, [(e, a.out) for e in range(lo, hi + 1)]):
            print(f"ev{ev:03d}: {msg}", flush=True)


if __name__ == "__main__":
    main()
