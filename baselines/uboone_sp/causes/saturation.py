#!/usr/bin/env python3
"""ADC saturation in the test events: stored clean sensor (post-digitize, pedestal-subtracted)
samples at the 12-bit limits (code 0 or 4095: value -ped or 4095-ped), the clean range per
plane, and how close a noisy waveform (clean + the v2 noise stream, seed 1000+ev) gets.
    scripts/helix_run.sh python baselines/uboone_sp/causes/saturation.py --out <json>"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "v2"))


def one(ev):
    from sp.data import Event, registry
    from helix.tpc.noise import generate_noise
    reg = registry(); E = Event(ev); rng = np.random.default_rng(1000 + ev); out = {}
    for g in range(6):
        nw, nt = reg[g]["n_wires"], reg[g]["n_ticks"]
        n_inc = generate_noise((nw, nt), rng=rng, wire_lengths_m=np.asarray(reg[g]["wire_lengths"]), incoherent=True, coherent=False)
        n_coh = generate_noise((nw, nt), rng=rng, incoherent=False, coherent=True, group_size=64)
        C = E.clean(g, reg); ped = reg[g]["pedestal"]
        hi, lo = 4095 - ped, -ped
        N = C + n_inc + n_coh
        out[g] = dict(cmax=float(C.max()), cmin=float(C.min()), n_clip_clean=int(((C >= hi) | (C <= lo)).sum()),
                      nmax=float(N.max()), nmin=float(N.min()), n_clip_noisy=int(((N >= hi - 0.5) | (N <= lo + 0.5)).sum()),
                      headroom_hi=float(hi - C.max()), headroom_lo=float(C.min() - lo))
    return ev, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    from multiprocessing import Pool
    with Pool(32) as p:
        R = dict(p.map(one, range(260, 388)))
    summ = {}
    for P, gs in (("U", (0, 3)), ("V", (1, 4)), ("Y", (2, 5))):
        rows = [R[e][g] for e in R for g in gs]
        summ[P] = dict(max_clean=max(r["cmax"] for r in rows), min_clean=min(r["cmin"] for r in rows),
                       n_clip_clean=sum(r["n_clip_clean"] for r in rows), n_clip_noisy=sum(r["n_clip_noisy"] for r in rows),
                       min_headroom_hi=min(r["headroom_hi"] for r in rows), min_headroom_lo=min(r["headroom_lo"] for r in rows),
                       max_noisy=max(r["nmax"] for r in rows), min_noisy=min(r["nmin"] for r in rows))
    json.dump(dict(summary=summ, per_event={str(k): v for k, v in R.items()}), open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


main()
