#!/usr/bin/env python3
"""First-principles B: what the production's per-sample 25 e- cut does to the hits-truth label, per cell.

    scripts/helix_run.sh python baselines/uboone_sp/causes/firstp_cut.py \
        --windows /global/cfs/cdirs/m5238/users/oalter/helix_runs/firstp/windows --workers 32

The hits truth keeps a (deposit group x wire x tick) sample only if it exceeds 25 e-, and a
fine cell is 2 wires x 16 ticks = 32 such samples per group: a cell whose charge is spread over
many samples, or shared by several groups, can carry hundreds of electrons and still be
labelled 0 or far below its charge. Two measurements:

  1. the 200 train events of firstp_sampling (hits vs complete truth files): for cells with
     complete charge > 102 e- and > 500 e-, the fraction whose hits charge is 0, or < 50 % of
     the complete charge, by cell charge and by density (complete charge within +-8 cells,
     the cell excluded), per plane;
  2. the 128 test events of firstp_windows (re-simulated with group identity): the same
     fractions by the number of deposit groups sharing the cell and by its largest single
     per-group sample -- the mechanism.

Writes causes/results/firstp_cut.json (read by firstp_detection_sum.py).
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from firstp_sampling import COMPLETE, DLAB, HITS, Grid, dbin  # noqa: E402

QB = np.array([102, 250, 500, 1e3, 2e3, 4e3, 8e3, 1e12])
QBL = ["102-250", "250-500", "500-1k", "1-2k", "2-4k", "4-8k", ">8k"]
NGB = np.array([1, 2, 3, 5, 9, 1e9]); NGL = ["1", "2", "3-4", "5-8", ">8"]
SMB = np.array([0, 25, 50, 100, 250, 1e12]); SML = ["<=25", "25-50", "50-100", "100-250", ">250"]


def one(job):
    run, tag, pos = job
    tr = {}
    for nm, root in (("hits", HITS), ("complete", COMPLETE)):
        z = np.load(os.path.join(root, run, f"truth_{tag}.npz"))
        a, b = int(z["offsets"][pos]), int(z["offsets"][pos + 1])
        tr[nm] = (z["keys"][a:b].astype(np.int64), z["q"][a:b].astype(np.float64))
    from helix.probe.resolution import unkey
    kc, qc = tr["complete"]; kh, qh_ = tr["hits"]
    j = np.clip(np.searchsorted(kh, kc), 0, max(len(kh) - 1, 0))
    qh = np.where(kh[j] == kc, qh_[j], 0.0) if len(kh) else np.zeros(len(kc))
    g, w, t = unkey(kc)
    gids = sorted(set(g.tolist()))
    G = Grid(kc, qc, gids, [0] * len(gids), int(t.min()), int(t.max()) + 1)
    dens = np.zeros(len(kc))
    for gg in gids:
        m = g == gg
        dens[m] = G.density(gg, w[m], t[m])
    sel = qc > 102
    hits_only = int(np.sum(~np.isin(kh, kc)))                   # sanity: hits cells absent from complete
    return dict(q=qc[sel], qh=qh[sel], d=dbin(dens[sel]), p=(g[sel] % 3).astype(np.int8),
                tot=np.array([qc.sum(), qh.sum(), qc[qh == 0].sum(), hits_only]))


def fr(q, qh, m):
    n = int(m.sum())
    if not n:
        return dict(n=0)
    return dict(n=n, frac_hits0=float(np.mean(qh[m] == 0)), frac_hits_lt50=float(np.mean(qh[m] < 0.5 * q[m])),
                median_ratio=float(np.median(qh[m] / q[m])), charge_lost=float(1 - qh[m].sum() / q[m].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "firstp_cut.json"))
    a = ap.parse_args()
    S = json.load(open(os.path.join(HERE, "results", "firstp_sampling.json")))
    jobs = []
    for e in S["meta"]["events"]:
        run, tag, pos = e.rsplit("_", 2)
        jobs.append((run, tag, int(pos)))
    with get_context("spawn").Pool(a.workers) as pool:
        R = pool.map(one, jobs)
    q, qh, d, p = (np.concatenate([r[k] for r in R]) for k in ("q", "qh", "d", "p"))
    tot = sum(r["tot"] for r in R)
    out = dict(meta=dict(n_events=len(R), note="cells with complete charge > 102 e-; density = complete charge within "
                         "+-8 fine cells, cell excluded; frac_hits0: hits label exactly 0; frac_hits_lt50: hits < 50% "
                         "of complete; charge_lost: 1 - sum hits / sum complete over those cells"),
               totals=dict(complete_charge=float(tot[0]), hits_charge=float(tot[1]),
                           complete_charge_in_cells_with_hits0=float(tot[2]), hits_cells_absent_from_complete=int(tot[3])))
    T = {}
    for thr in (102, 500):
        T[f">{thr}"] = {"all": fr(q, qh, q > thr)}
        T[f">{thr}"]["by_plane"] = {pl: fr(q, qh, (q > thr) & (p == i)) for i, pl in enumerate("UVY")}
        T[f">{thr}"]["by_density"] = {dl: fr(q, qh, (q > thr) & (d == i)) for i, dl in enumerate(DLAB)}
    T["by_charge"] = {ql: fr(q, qh, (q > QB[i]) & (q <= QB[i + 1])) for i, ql in enumerate(QBL)}
    T["density_x_charge"] = {dl: {ql: fr(q, qh, (d == di) & (q > QB[i]) & (q <= QB[i + 1])) for i, ql in enumerate(QBL)}
                             for di, dl in enumerate(DLAB)}
    out["train_events"] = T
    # ---- mechanism on the test events
    files = sorted(glob.glob(os.path.join(a.windows, "ev*.npz")))
    C = {k: [] for k in ("q", "qthr", "ng", "smax", "np")}
    for f in files:
        z = np.load(f)
        for k in C:
            C[k].append(z[f"cell_{k}"])
    C = {k: np.concatenate(v) for k, v in C.items()}
    M = {}
    for thr in (102, 500):
        base = C["q"] > thr
        M[f">{thr}"] = dict(all=fr(C["q"], C["qthr"], base),
                            by_groups={nl: fr(C["q"], C["qthr"], base & (C["ng"] >= NGB[i]) & (C["ng"] < NGB[i + 1]))
                                       for i, nl in enumerate(NGL)},
                            by_max_sample={sl: fr(C["q"], C["qthr"], base & (C["smax"] > SMB[i]) & (C["smax"] <= SMB[i + 1]))
                                           for i, sl in enumerate(SML)},
                            by_particles={nl: fr(C["q"], C["qthr"], base & (C["np"] >= NGB[i]) & (C["np"] < NGB[i + 1]))
                                          for i, nl in enumerate(NGL)})
    out["test_events_mechanism"] = dict(n_events=len(files), **M)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out["train_events"][k]["all"] for k in (">102", ">500")}, indent=1))
    print(json.dumps(out["train_events"][">102"]["by_density"], indent=1))
    print(json.dumps(out["test_events_mechanism"][">102"], indent=1))


if __name__ == "__main__":
    main()
