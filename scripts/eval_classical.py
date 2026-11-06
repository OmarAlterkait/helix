#!/usr/bin/env python3
"""Score the tuned classical (Wire-Cell / MicroBooNE) chain with scripts/eval_denoise.py's
keys, on the SAME digitised waveforms the model reads (scripts/eval_draws.py), next to
the model rows.

    scripts/helix_run.sh python scripts/eval_classical.py --truth <truth_v2> --near <truth_v2_near> \
        --annot <noise_vs_hits.npz> --q0 2043.2578108909702 --draws 5 [--v2] [--complete-truth DIR] \
        --out results/cls.jsonl

The chain is the frozen tuned configuration of baselines/uboone_sp (v2/results/benchmark.json
picks, fitted kernel; imported, not copied):
  digitised plane -> sp.noise_filter.gate_clean -> sp.wirecell_sp.decon_outputs ->
    tuned_q    ROI Gaus_wide charge (U q_s1.25_f0.25, V q_s1.5_f0, Y q_s0.67_f0), fine-cell sums
    tuned_any  best detection statistic (U zlf06, V zor06a2, Y = tuned_q)
Sources: ``d<k>`` = draw k of the cache (sha256-checked planes); ``v2`` (``--v2``) = the
classical benchmark's own noise draw (v2/classical.py, seed 1000+ev).
Rows: ``cls_tuned@<src>_q`` (charge: tuned_q on map rows and windows) and
``cls_tuned@<src>_any`` (detection: tuned_any; no charge keys), and with --draws the
aggregates ``cls_tuned@draws<K>_{q,any}`` (eval_denoise.aggregate). Map rows: ALL of the
truth file's rows (the model rows keep only token-covered rows; the classical chain has
no tokens). Classical maps are cached per source under --map-cache.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.join(ROOT, "baselines", "uboone_sp")
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
MAP_CACHE = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/eval_draws/cls_maps"


def _paths():
    for p in (BASE, os.path.join(BASE, "v2"), os.path.join(BASE, "dense")):
        if p not in sys.path:
            sys.path.insert(0, p)


def maps_for_plane(raw, g, kern, picks):
    """tuned_q / tuned_any fine-cell maps of one digitised plane (dense/cls_maps.py logic)."""
    _paths()
    from sp.data import PLANES
    from sp.noise_filter import gate_clean
    from sp.pipeline import Kernel, fine_cells
    from sp.response import load_kernel
    from sp import wirecell_sp as WC
    from classical import chain_pre, finish
    from cls_maps import detect_stat
    p = g % 3
    pl = PLANES[p]
    d = WC.decon_outputs(gate_clean(np.asarray(raw, np.float32)), pl, Kernel(*load_kernel(pl, kern)))
    name = picks["tuned_q"][p]
    s, f = (float(x[1:]) for x in name[2:].split("_"))
    Rf, attr = chain_pre(d, pl, dict(th_factor_ind=3.0 * s, th_factor_col=5.0 * s, l_factor=3.5 * s))
    tq = fine_cells(finish(Rf, attr, f, d["gauss"], pl)[0])
    ta = tq if picks["tuned_any"][p] == name else detect_stat(picks["tuned_any"][p], d)
    return np.asarray(tq, np.float32), np.asarray(ta, np.float32)


def job(args):
    ev, g, src, truth, cache, map_cache, kern, picks = args
    out = os.path.join(map_cache, src, f"ev{ev:03d}_g{g}.npz")
    if os.path.exists(out):
        return ev, g, "exists"
    t0 = time.time()
    _paths()
    os.environ["OMP_NUM_THREADS"] = "1"
    if src == "v2":                                      # v2/classical.py's noise stream, seed 1000 + ev
        from helix.tpc.noise import digitize, generate_noise
        from sp.data import Event, registry
        reg = registry(); E = Event(ev, truth_dir=truth)
        rng = np.random.default_rng(1000 + ev)
        for gg in range(g + 1):
            nw_, nt_ = reg[gg]["n_wires"], reg[gg]["n_ticks"]
            n_inc = generate_noise((nw_, nt_), rng=rng, wire_lengths_m=np.asarray(reg[gg]["wire_lengths"]),
                                   incoherent=True, coherent=False)
            n_coh = generate_noise((nw_, nt_), rng=rng, incoherent=False, coherent=True, group_size=64)
        raw = digitize(E.clean(g, reg) + n_inc + n_coh, reg[g]["pedestal"])
    else:
        import eval_draws as ED
        k = int(src[1:])
        raw = ED.planes(os.path.join(cache, f"ev{ev:03d}_d{k}.npz"))[g]
    tq, ta = maps_for_plane(raw, g, kern, picks)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out + ".tmp.npz", tuned_q=tq, tuned_any=ta)
    os.replace(out + ".tmp.npz", out)
    return ev, g, f"{time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth", required=True)
    ap.add_argument("--near", default=None)
    ap.add_argument("--annot", default=None)
    ap.add_argument("--q0", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-events", type=int, default=260)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--v2", action="store_true", help="also the classical benchmark's own noise draw (seed 1000+ev)")
    ap.add_argument("--draw-cache", default="/global/cfs/cdirs/m5238/users/oalter/helix_runs/eval_draws/coeffs")
    ap.add_argument("--map-cache", default=MAP_CACHE)
    ap.add_argument("--kernel", default="fit")
    ap.add_argument("--complete-truth", default=None)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--maps-only", action="store_true")
    a = ap.parse_args()
    import eval_denoise as EV
    _paths()
    from cls_maps import load_picks, BENCH
    from helix.data.denoise import DenoiseEvents
    from helix.probe import resolution as R
    picks = load_picks(BENCH)
    files, ev_list = EV.scored_files(a)
    evs = [int(os.path.basename(f)[2:5]) for _, f in ev_list]
    sources = (["v2"] if a.v2 else []) + [f"d{k}" for k in range(a.draws)]
    jobs = [(ev, g, s, a.truth, a.draw_cache, a.map_cache, a.kernel, picks) for s in sources for ev in evs for g in range(6)]
    jobs.sort(key=lambda j: (j[1] % 3 == 2,))                 # induction first (slowest)
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.workers, maxtasksperchild=8) as pool:
        for i, (ev, g, msg) in enumerate(pool.imap_unordered(job, jobs)):
            if i % 200 == 0:
                print(f"maps {i}/{len(jobs)}: ev{ev:03d} g{g} {msg}", flush=True)
    if a.maps_only:
        return
    cfg = DenoiseEvents("", [], "", items=[]).cfg
    Q0 = float(a.q0)
    ct = EV.CompleteTruth(a.complete_truth) if a.complete_truth else None
    a.tag = "cls_tuned"
    per = {}
    for src in sources:
        maps = {}

        def tokens_of(z, f, src=src):
            ev = int(os.path.basename(f)[2:5])
            maps["cur"] = [np.load(os.path.join(a.map_cache, src, f"ev{ev:03d}_g{g}.npz")) for g in range(6)]
            return None, None

        def predict_fn(B, bl, keys):
            g, fw, ft = R.unkey(np.asarray(keys, np.int64))
            q = np.zeros(len(keys), np.float32); s = np.zeros(len(keys), np.float32)
            for gg in range(6):
                Mq, Ma = maps["cur"][gg]["tuned_q"], maps["cur"][gg]["tuned_any"]
                m = (g == gg) & (fw >= 0) & (fw < Mq.shape[0]) & (ft >= 0) & (ft < Mq.shape[1])
                q[m] = Mq[fw[m], ft[m]]; s[m] = Ma[fw[m], ft[m]]
            # rows: log1p(q / Q0) as the model's charge head; "presence" slot carries the detection statistic
            return np.log1p(q / Q0).astype(np.float32), np.ones(len(keys), bool), s
        C = EV.collect(a, ev_list, Q0, tokens_of, predict_fn, ct)
        ctx = EV.context(a, C, Q0, cfg)
        hdr = dict(checkpoint="classical:tuned(v2 picks, kernel=%s)" % a.kernel, step=None, train_events=96,
                   init="baselines/uboone_sp v2 (tuned on ev000-095)", P=C["P"], extra=dict(source=src))
        wq_ = np.concatenate(C["WP"])
        t = f"{a.tag}@{src}"
        _, pt_q, b_q = EV.score(a, ctx, t + "_q", np.concatenate(C["P"]), np.expm1(np.maximum(wq_, 0)) * Q0, hdr, charge=True)
        _, pt_a, b_a = EV.score(a, ctx, t + "_any", np.concatenate(C["PR"]), np.concatenate(C["WPR"]), hdr, charge=False)
        if src.startswith("d"):
            per.setdefault("_q", []).append((pt_q, b_q)); per.setdefault("_any", []).append((pt_a, b_a))
    for suf, pd in per.items():
        EV.aggregate(a, f"{a.tag}@draws{len(pd)}{suf}", pd, dict(checkpoint="classical:tuned", step=None, train_events=96,
                                                                 init="baselines/uboone_sp v2", Q0=Q0, n_windows=len(ctx["META"])))


if __name__ == "__main__":
    main()
