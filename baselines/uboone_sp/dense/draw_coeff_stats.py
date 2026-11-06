#!/usr/bin/env python3
"""Coefficient statistics of the STORED corpus vs fresh noise draws through today's DSP.

    scripts/helix_run.sh python baselines/uboone_sp/dense/draw_coeff_stats.py --events 260-387 \
        --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/draws/coeff_stats

Per event, plane gid, band (A4, D4, D3, D2 -- what the tokenizer reads), for three versions
of the same event:
  stored   the corpus shard (2080 Ti, torch backend, helix c646886)
  tn       fresh torch-GPU draw, recorded noise_seed (causes2/tn_planes), today's DSP (numpy)
  cn       the classical chain's numpy draw (seed 1000+ev), today's DSP
Regions from the truth (fine cells, dilated): 'sig' = coefficient footprint on a charged
cell; 'near' = within +-8 cells of charge but not on it; 'far' = > 16 cells from charge.
Recorded: sigma_threshold; kept counts per region; |v|/sigma quantiles of kept far
(noise) coefficients and tail fractions; adjacent-wire co-occurrence of kept far
coefficients (coherent residual); sum of |v| on signal-footprint coefficients.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(BASE)))
CFS = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
CLS = (f"{CFS}/dense2/cls_fix", "/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/cls_fit")
DEC = {0: 16, 1: 16, 2: 8, 3: 4}           # ticks per coefficient, bands A4, D4, D3, D2


def stats(ce, T, ev):
    from scipy.ndimage import maximum_filter
    b, g, w, tau, v = (np.asarray(x) for x in (ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value))
    sig = np.asarray(ce.sigma_threshold)
    out = {}
    for gg in range(6):
        occ = T[gg] > 0
        near = maximum_filter(occ.astype(np.uint8), size=17) > 0
        far = ~(maximum_filter(occ.astype(np.uint8), size=33) > 0)
        for bb in range(4):
            m = (g == gg) & (b == bb)
            fw = np.clip(w[m] // 2, 0, occ.shape[0] - 1)
            ft = np.clip((tau[m] * DEC[bb] + DEC[bb] // 2) // 16, 0, occ.shape[1] - 1)
            on, nr, fr = occ[fw, ft], near[fw, ft] & ~occ[fw, ft], far[fw, ft]
            a = np.abs(v[m]) / max(sig[gg, bb], 1e-9)
            af = a[fr]
            # adjacent-wire co-occurrence among far coefficients
            key = (w[m][fr].astype(np.int64) << 20) | tau[m][fr].astype(np.int64)
            ks = set(key.tolist())
            nb = sum(1 for k in key.tolist() if ((k + (1 << 20)) in ks) or ((k - (1 << 20)) in ks))
            out[f"{gg}|{bb}"] = dict(sigma=float(sig[gg, bb]), n=int(m.sum()), n_sig=int(on.sum()), n_near=int(nr.sum()), n_far=int(fr.sum()),
                                     far_cells=int(far.sum()), near_cells=int((near & ~occ).sum()),
                                     far_q=[float(x) for x in np.quantile(af, (0.5, 0.9, 0.99, 0.999))] if len(af) else None,
                                     far_gt5=int((af > 5).sum()), far_gt8=int((af > 8).sum()), far_adj=int(nb),
                                     sig_sumabs=float(a[on].sum()), near_sumabs=float(a[nr].sum()))
    return out


def process(job):
    ev, out_dir = job
    out = os.path.join(out_dir, f"ev{ev:03d}.json")
    if os.path.exists(out):
        return ev, "exists"
    from helix.core import backend
    from helix.core.coeff_io import read_coeff_event
    from helix.tpc.noise import digitize, generate_noise
    from helix.tpc.pipeline import process_plane, event_coeff_event
    import corpus_noise as CN
    from sp.data import Event, registry
    shard, pos, event, seed, sensor = CN.event_files(ev)
    reg, cfg, spec = CN.setup(sensor)
    backend.set_backend("numpy")
    T = {g: np.load(next(os.path.join(d, f"ev{ev:03d}_g{g}.npz") for d in CLS if os.path.exists(os.path.join(d, f"ev{ev:03d}_g{g}.npz"))))["truth"] for g in range(6)}
    res = {"stored": stats(read_coeff_event(shard, pos), T, ev)}
    P = np.load(f"{CFS}/causes2/tn_planes/ev{ev:03d}.npz")
    tn = event_coeff_event({g: process_plane(P[f"g{g}"].astype(np.float32), cfg, removal="gate", with_images=False) for g in range(6)},
                           cfg, run="", source_file="", event=event, removal="gate")
    res["tn"] = stats(tn, T, ev)
    reg2 = registry(); E = Event(ev); rng = np.random.default_rng(1000 + ev); r = {}
    for g in range(6):
        nw_, nt_ = reg2[g]["n_wires"], reg2[g]["n_ticks"]
        ni = generate_noise((nw_, nt_), rng=rng, wire_lengths_m=np.asarray(reg2[g]["wire_lengths"]), incoherent=True, coherent=False)
        nc = generate_noise((nw_, nt_), rng=rng, incoherent=False, coherent=True, group_size=64)
        r[g] = process_plane(np.asarray(digitize(E.clean(g, reg2) + ni + nc, reg2[g]["pedestal"]), np.float32), cfg, removal="gate", with_images=False)
    res["cn"] = stats(event_coeff_event(r, cfg, run="", source_file="", event=event, removal="gate"), T, ev)
    json.dump(res, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    return ev, "ok"


def main():
    sys.path.insert(0, os.path.join(BASE, "v2"))
    from classical import parse_events
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.makedirs(a.out, exist_ok=True)
    from multiprocessing import Pool
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for ev, msg in pool.imap_unordered(process, [(e, a.out) for e in parse_events(a.events)]):
            print(f"ev{ev:03d}: {msg}", flush=True)


if __name__ == "__main__":
    main()
