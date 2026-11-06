#!/usr/bin/env python3
"""Derive our simulation's 2D response from (hits, clean sensor) pairs, and test
how well clean ~= hits (*) response holds for it and for the example's kernel.

    scripts/helix_run.sh python baselines/uboone_sp/fit_response.py [--fit-events 16] [--test-events 4]

Fit: exact least squares for a compact kernel (21 wires x 211 ticks) over the
fit events (truth files ev000.., the probe split's TRAINING part -- never the
scored test events), both volumes pooled per plane type. Zero suppression in the
stored sensor (|v| < 2 ADC absent) biases the tails; ``--impute N`` re-fits N
times with the suppressed samples replaced by the current prediction clipped to
(-2, 2) -- the only values they can have held.

Test: on held-out test events, relative residual ||clean - store(H (*) R)|| /
||clean||, where store() rounds to integer ADC and drops |v| < 2 exactly as the
sensor writer does. Writes response/fit_{U,V,Y}.npz in the example's layout and
response/residuals.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit-events", type=int, default=16)
    ap.add_argument("--test-events", type=int, default=4)
    ap.add_argument("--impute", type=int, default=2)
    a = ap.parse_args()

    from sp.data import Event, registry, PLANES
    from sp.response import LSAccumulator, forward, load_kernel, store_like_sensor, FITTED

    reg = registry()
    os.makedirs(FITTED, exist_ok=True)
    fit_pairs = {p: [] for p in range(3)}
    for ev in range(a.fit_events):
        E = Event(ev)
        for g in range(6):
            C = E.clean(g, reg)
            H = E.truth(g, C.shape)
            if H.sum() > 0:
                fit_pairs[g % 3].append((H, C))
    print(f"fit planes: {[len(v) for v in fit_pairs.values()]}", flush=True)

    kernels = {}
    for p in range(3):
        Ss = [C.astype(np.float64) for _, C in fit_pairs[p]]
        for it in range(a.impute + 1):
            acc = LSAccumulator()
            for (H, _), S in zip(fit_pairs[p], Ss):
                acc.add(H, S)
            R, wo, tzb = acc.solve()
            if it < a.impute:            # impute the suppressed samples, refit
                Ss = []
                for H, C in fit_pairs[p]:
                    pred = forward(H, R, wo, tzb)
                    S = C.astype(np.float64).copy()
                    z = C == 0
                    S[z] = np.clip(pred[z], -1.999, 1.999)
                    Ss.append(S)
            r = np.mean([np.linalg.norm(C - store_like_sensor(forward(H, R, wo, tzb))) / np.linalg.norm(C)
                         for H, C in fit_pairs[p]])
            print(f"  {PLANES[p]} iteration {it}: fit-set residual {r:.4f}  sum(R) {R.sum():+.5f}", flush=True)
        kernels[p] = (R, wo, tzb)
        np.savez(os.path.join(FITTED, f"fit_{PLANES[p]}.npz"), R_avg=R.astype(np.float32), wire_offsets=wo,
                 time_zero_bin=np.int32(tzb), time_bin_size=np.float32(0.5), plane=PLANES[p],
                 fit_events=np.arange(a.fit_events), impute=a.impute)

    # held-out test: residual of each kernel, per plane gid
    res = {}
    for ev in range(260, 260 + a.test_events):
        E = Event(ev)
        for g in range(6):
            p = g % 3
            C = E.clean(g, reg).astype(np.float64)
            H = E.truth(g, C.shape)
            if not C.any():
                continue
            row = dict(ev=ev, g=g, clean_sum=float(C.sum()), hits_sum=float(H.sum()))
            for src in ("example", "fit"):
                R, wo, tzb = load_kernel(PLANES[p], src) if src == "example" else kernels[p]
                pred = forward(H, R, wo, tzb)
                st = store_like_sensor(pred)
                row[f"resid_{src}"] = float(np.linalg.norm(C - st) / np.linalg.norm(C))
                row[f"resid_float_{src}"] = float(np.linalg.norm(C - pred) / np.linalg.norm(C))
                row[f"corr_{src}"] = float(np.corrcoef(C.ravel(), st.ravel())[0, 1])
                row[f"zs_agree_{src}"] = float(np.mean((st != 0) == (C != 0)))
                row[f"sum_ratio_{src}"] = float(st.sum() / C.sum())
            res.setdefault(PLANES[p], []).append(row)
            print(f"  ev{ev} g{g}: resid example {row['resid_example']:.4f} fit {row['resid_fit']:.4f} | "
                  f"float-pred resid example {row['resid_float_example']:.4f} fit {row['resid_float_fit']:.4f} | "
                  f"sum ratio example {row['sum_ratio_example']:.4f} fit {row['sum_ratio_fit']:.4f}", flush=True)

    # kernel agreement
    cmp = {}
    for p in range(3):
        Re, woe, tze = load_kernel(PLANES[p], "example")
        Rf, wof, tzf = kernels[p]
        E_ = np.zeros_like(Rf)
        for j, w in enumerate(woe):
            jj = int(np.nonzero(wof == w)[0][0])
            for k in range(Re.shape[0]):
                kk = k - tze + tzf
                if 0 <= kk < Rf.shape[0]:
                    E_[kk, jj] = Re[k, j]
        cmp[PLANES[p]] = dict(rel_diff=float(np.linalg.norm(Rf - E_) / np.linalg.norm(E_)),
                              sum_example=float(Re.sum()), sum_fit=float(Rf.sum()),
                              outside_example_support=float(np.linalg.norm(Rf[E_ == 0]) / np.linalg.norm(Rf)))
        print(f"  {PLANES[p]} kernel: ||fit - example|| / ||example|| = {cmp[PLANES[p]]['rel_diff']:.4f}; "
              f"sum fit {Rf.sum():+.5f} example {Re.sum():+.5f}; fit norm outside example support "
              f"{cmp[PLANES[p]]['outside_example_support']:.4f}")
    summ = {p: {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k.startswith(("resid", "corr", "zs", "sum_ratio"))}
            for p, rows in res.items()}
    print(json.dumps(summ, indent=1))
    with open(os.path.join(FITTED, "residuals.json"), "w") as fh:
        json.dump(dict(summary=summ, kernels=cmp, rows=res, fit_events=a.fit_events, impute=a.impute), fh, indent=1)


if __name__ == "__main__":
    main()
