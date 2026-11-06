#!/usr/bin/env python3
"""Run the classical (MicroBooNE / Wire-Cell) chain on the floor evaluation's test
events as a ladder of rungs, and store per-fine-cell predictions for scoring.

    scripts/helix_run.sh python baselines/uboone_sp/run_ladder.py --events 260-387 --workers 12 \
        --out /pscratch/sd/o/oalter/helix_work/uboone_sp/preds [--dump 300]

Rungs (each removes one piece of perfection from the one above):
  R0    clean sensor image (perfect denoise) + oracle ROIs from truth, our sim's
        fitted response, noiseless DC handling          -- deconvolution alone
  R0ex  as R0 with the EXAMPLE's response kernel        -- cost of a wrong response
  R1    digitize(clean + incoherent noise)  (coherent noise removed perfectly)
        + oracle ROIs, noisy DC handling                -- cost of intrinsic noise
  R2    digitize(clean + incoherent + coherent) -> our coherent gate (corpus's
        removal, inverse-DWT'd) + oracle ROIs          -- cost of coherent removal
  R2mp  as R2 with image-space multipass removal (helix.tpc.coherent)
  R3    R2's input + ROIs found by the MicroBooNE algorithm  = the full chain
  R0syn hits (*) fitted response as a float image (no rounding, no zero
        suppression) + oracle ROIs                      -- the deconvolution's own limit
  R0zs  R0syn rounded to integer ADC with |v| < 2 dropped, as the sensor writer
        does                                            -- cost of the storage alone
  diagnostics (not rungs):
  R1nr / R2nr   the Gaussian-filtered deconvolution with NO ROI mask at all
  R0nr / R0synnr  the same for noise-free inputs: deconvolution artefacts alone
  R0nrp / R1nrp / R2nrp  no ROI, with the waveform zero-padded by 512 ticks at both
        ends before the FFT -- removes the cyclic wrap at the readout edges
  R3inc  R1's input (no coherent noise) + algorithmic ROIs

Charge per fine cell (2 wires x 16 ticks) = sum of the ROI-masked,
baseline-subtracted Gaussian-filter output (the example's C_charge, its "naive"
charge extraction), clipped at 0. Noise: one fresh forward-model realisation per
event, seed 1000 + ev (as scripts/snr_stages.py), shared by every rung.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RUNGS = ("R0", "R0ex", "R1", "R2", "R2mp", "R3", "R1nr", "R2nr", "R3inc")
Q0 = 2043.2578108909702            # the floor evaluation's charge scale (median map-row charge, ev000-ev259)
PAD = 512                          # zero padding (ticks) for the *nrp edge-effect diagnostics
FP_FRACS = (0.05, 0.25, 1.0)       # 'cells above f x Q0 where the truth is 0' thresholds


def per_wire(C, mask, T):
    return (C * mask).sum(1).astype(np.float32), T.sum(1).astype(np.float32)


def process_event(job):
    ev, out_dir, dump, rungs, near_dir = job[:5]
    kname = job[5] if len(job) > 5 else "fit"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from sp.data import Event, registry, noisy_planes, PLANES
    from sp.noise_filter import gate_clean, multipass_clean
    from sp.pipeline import Kernel, run_sp, fine_cells, lookup, filter_truth, deconvolve_2d_dual
    from sp.response import load_kernel, forward, store_like_sensor
    from uboone_pipeline.metrics import pixel_metrics

    out = os.path.join(out_dir, f"ev{ev:03d}.npz")
    if os.path.exists(out):
        return ev, "exists"
    t0 = time.time()
    reg = registry()
    K = {p: Kernel(*load_kernel(p, kname)) for p in PLANES}
    KX = {p: Kernel(*load_kernel(p, "example")) for p in PLANES}
    E = Event(ev)
    rng = np.random.default_rng(1000 + ev)
    maps = {r: {} for r in rungs}
    planes = {}
    wires = {}
    need_noise = any(r not in ("R0", "R0ex", "R0syn", "R0zs", "R0nr", "R0synnr", "R0zsnr") for r in rungs)
    for g in range(6):
        pl = PLANES[g % 3]
        C = E.clean(g, reg)
        T = E.truth(g, C.shape)
        L = np.asarray(reg[g]["wire_lengths"]); ped = reg[g]["pedestal"]
        info = dict(hits=float(T.sum()), clean_adc=float(C.sum()))
        if need_noise:                      # always drawn, so every rung sees the same realisation
            inc, coh = noisy_planes(C, L, ped, rng)
            coh_gate = gate_clean(coh)
            coh_mp = multipass_clean(coh) if "R2mp" in rungs else None
            info.update(noise_rms_inc=float(np.std(inc - C)), noise_rms_coh=float(np.std(coh - C)),
                        resid_rms_gate=float(np.std(coh_gate - inc)),
                        resid_rms_mp=float(np.std(coh_mp - inc)) if coh_mp is not None else float("nan"))
        if any(r.startswith(("R0syn", "R0zs")) for r in rungs):
            syn = forward(T, *load_kernel(pl, kname)).astype(np.float32)
        # rung -> (input, kernel, oracle ROI?, noiseless DC?)
        spec = dict(R0=(lambda: C, K, True, True), R0ex=(lambda: C, KX, True, True),
                    R0syn=(lambda: syn, K, True, True), R0zs=(lambda: store_like_sensor(syn), K, True, True),
                    R1=(lambda: inc, K, True, False), R3inc=(lambda: inc, K, False, False),
                    R2=(lambda: coh_gate, K, True, False), R2mp=(lambda: coh_mp, K, True, False),
                    R3=(lambda: coh_gate, K, False, False))
        res = {}
        for r in rungs:
            if r in spec:
                src, kk, oracle, nl = spec[r]
                res[r] = run_sp(src(), pl, kk[pl], truth=T, use_true_roi=oracle, noiseless=nl)
        Tf = filter_truth(T, pl)
        for r in rungs:
            if r.endswith("nrp"):           # no ROI, waveform zero-padded 512 ticks each side before the FFT
                base = r[:-3]
                Mb = spec[base][0]()
                _, cgp = deconvolve_2d_dual(np.pad(Mb, ((0, 0), (PAD, PAD))), pl, K[pl], noiseless=spec[base][3])
                Cq = cgp[:, PAD:PAD + Mb.shape[1]]; mask = np.ones(Cq.shape, bool)
            elif r.endswith("nr"):
                base = r[:-2]
                src = res[base] if base in res else run_sp(spec[base][0](), pl, spec[base][1][pl], truth=T,
                                                            use_true_roi=True, noiseless=spec[base][3])
                Cq = src["C_gauss"]; mask = np.ones(Cq.shape, bool)
            else:
                Cq, mask = res[r]["C_charge"], res[r]["roi_mask"]
            X = fine_cells(Cq)
            maps[r][g] = X
            Tc = fine_cells(T)
            pm = pixel_metrics(Cq, Tf, mask) if not r.endswith(("nr", "nrp")) else {}
            info[r] = dict(sum_charge=float(Cq[mask].sum()), sum_cells=float(X.sum()),
                           roi_frac=float(mask.mean()), n_roi_wires=int(mask.any(1).sum()),
                           F0_f=pm.get("F0", np.nan), corr_f=pm.get("corr", np.nan), alpha_f=pm.get("alpha", np.nan),
                           fp_cells=[int(np.sum((Tc == 0) & (X > f * Q0))) for f in FP_FRACS], n_cells=int(X.size))
            if not r.endswith(("nr", "nrp")):
                wires[f"{r}_{g}"], wires[f"T_{g}"] = per_wire(Cq, mask, T)
        if "R3" in res:     # fraction of truth charge inside the algorithmic ROI
            info["R3_truth_in_roi"] = float(T[res["R3"]["roi_mask"]].sum() / max(T.sum(), 1))
        if "R0" in res:
            info["R0_truth_in_roi"] = float(T[res["R0"]["roi_mask"]].sum() / max(T.sum(), 1))
        planes[g] = info
        if ev in dump and all(r in res for r in ("R0", "R1", "R2", "R3")):
            np.savez_compressed(os.path.join(out_dir, f"full_ev{ev:03d}_g{g}.npz"),
                                truth=T, clean=C, inc=inc, coh=coh, coh_gate=coh_gate,
                                **{f"C_{r}": res[r]["C_charge"] for r in ("R0", "R1", "R2", "R3")},
                                R3_gauss=res["R3"]["C_gauss"], R3_wiener=res["R3"]["C_wiener"],
                                R3_mask=res["R3"]["roi_mask"], R0_mask=res["R0"]["roi_mask"])
    z = E.z
    arrs = {}
    for r in rungs:
        arrs[f"m_{r}"] = lookup(maps[r], z["mrows"]).astype(np.float32)
        arrs[f"w_{r}"] = lookup(maps[r], z["wkey"]).astype(np.float32)
        if near_dir:
            zn = np.load(os.path.join(near_dir, f"ev{ev:03d}.npz"), allow_pickle=True)
            arrs[f"n_{r}"] = lookup(maps[r], zn["wkey"]).astype(np.float32)
    if ev in dump:
        np.savez_compressed(os.path.join(out_dir, f"cells_ev{ev:03d}.npz"),
                            **{f"{r}_{g}": maps[r][g].astype(np.float32) for r in rungs for g in range(6)})
    tmp = out + ".tmp.npz"
    np.savez(tmp, planes=json.dumps(planes), **arrs, **wires)
    os.replace(tmp, out)
    return ev, f"{time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default="", help="comma-separated events to dump full planes for figures")
    ap.add_argument("--rungs", default=",".join(RUNGS), help="subset of rungs to run (default: all)")
    ap.add_argument("--near", default=None, help="truth_v2_near sidecar dir: also store each rung at the "
                                                  "near-activity noise windows' cells (n_<rung>)")
    ap.add_argument("--kernel", default="fit", choices=("fit", "exact", "exact0"),
                    help="response used by every rung that says 'fitted' (R0syn's forward model too)")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.events.split("-")) if "-" in a.events else (int(a.events), int(a.events))
    dump = {int(x) for x in a.dump.split(",") if x}
    os.makedirs(a.out, exist_ok=True)
    rungs = tuple(a.rungs.split(","))
    jobs = [(ev, a.out, dump, rungs, a.near, a.kernel) for ev in range(lo, hi + 1)]
    if a.workers <= 1:
        for j in jobs:
            print(*process_event(j), flush=True)
        return
    from multiprocessing import Pool
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for ev, msg in pool.imap_unordered(process_event, jobs):
            print(f"ev{ev:03d}: {msg}", flush=True)


if __name__ == "__main__":
    main()
