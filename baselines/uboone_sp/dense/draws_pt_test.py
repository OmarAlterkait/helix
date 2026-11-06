#!/usr/bin/env python3
"""Stored corpus vs fresh draws on MORE independent stored realisations: the 258 probe-train
events (ev000-259 without ev041/ev160; never used to train the denoisers) plus the 128 test
events. Detection only (near-activity 1%-FPR per plane, eval windows), presence scores.

    scripts/helix_run.sh python baselines/uboone_sp/dense/draws_pt_test.py \
        --out baselines/uboone_sp/dense/results/draws_pt_test.json

Per event set, per model: efficiency on the stored corpus and on each fresh draw, the
stored - mean(draws) difference with an event bootstrap that RE-SETS the thresholds in
every resample (so threshold noise is included), and its z.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CFS = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"
MODELS = ("M6d", "M6dc", "M4n")
B = 200


def srcdir(evset, src, model):
    if evset == "pt":
        return f"{CFS}/draws/pt_{'stored' if src == 'stored' else src}"
    if src == "stored":
        return f"{CFS}/{'dense2' if model == 'M4n' else 'dense3'}/model"
    return f"{CFS}/draws/model_{src}"


def per_event(args):
    ev, evset, sources = args
    from helix.probe.resolution import unkey
    out = {"meta": [], "win": {}}
    wins = []
    for f in (TRUTH, NEAR):
        z = np.load(os.path.join(f, f"ev{ev:03d}.npz"), allow_pickle=True)
        mm = json.loads(str(z["meta"]))
        if not mm:
            continue
        starts = np.searchsorted(z["wid"], np.arange(len(mm))); ends = np.append(starts[1:], len(z["wid"]))
        wins.append((mm, z["wkey"], starts, ends))
        out["meta"] += [dict(kind=m["kind"], p=m["g"] % 3, E=float(m["E"][0]) if m["kind"] == "iso" else -1) for m in mm]
    for src in sources:
        for model in MODELS:
            Mz = np.load(os.path.join(srcdir(evset, src, model), f"ev{ev:03d}.npz"))
            to = int(Mz["toff"]); P = {g: Mz[f"p_{model}_g{g}"] for g in range(6)}
            sc = []
            for mm, keys, starts, ends in wins:
                g_, fw, ft = unkey(keys)
                v = np.zeros(len(keys))
                for gg in range(6):
                    m = (g_ == gg) & (fw < P[gg].shape[0]) & (ft + to >= 0) & (ft + to < P[gg].shape[1])
                    v[m] = P[gg][fw[m], ft[m] + to]
                s = np.maximum.reduceat(v, starts) if len(v) else np.zeros(len(mm))
                s[ends == starts] = 0
                sc.append(s)
            out["win"][f"{src}|{model}"] = np.concatenate(sc)
    return ev, out


def effs(S, kind, p, E, w=None):
    """near-pp efficiency at 0.1-0.2 MeV per plane + pooled, thresholds set on (weighted) bgn windows."""
    w = np.ones(len(S)) if w is None else w
    res = np.zeros(4)
    eb = (kind == "iso") & (E > 0.1) & (E <= 0.2)
    num = den = 0.0
    for pl in range(3):
        bg = (kind == "bgn") & (p == pl)
        x, ww = S[bg], w[bg]
        o = np.argsort(x); cw = np.cumsum(ww[o]) / ww.sum()
        t = x[o][min(np.searchsorted(cw, 0.99), len(o) - 1)]
        sel = eb & (p == pl)
        n_ = np.sum(w[sel] * (S[sel] > t)); d_ = np.sum(w[sel])
        res[pl] = n_ / d_; num += n_; den += d_
    res[3] = num / den
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    sets = {"pt": ([e for e in range(0, 260) if e not in (41, 160)], ["stored", "d0", "d1"]),
            "test": (list(range(260, 388)), ["stored"] + [f"d{k}" for k in range(10)])}
    res = {}
    rng = np.random.default_rng(3)
    for name, (evs, sources) in sets.items():
        with Pool(a.workers) as pool:
            R = dict(pool.map(per_event, [(e, name, sources) for e in evs]))
        meta = [m for e in evs for m in R[e]["meta"]]
        evi = np.array([i for i, e in enumerate(evs) for _ in R[e]["meta"]])
        kind = np.array([m["kind"] for m in meta]); p = np.array([m["p"] for m in meta]); E = np.array([m["E"] for m in meta])
        Wb = [np.bincount(rng.integers(0, len(evs), len(evs)), minlength=len(evs))[evi].astype(float) for _ in range(B)]
        for model in MODELS:
            S = {s: np.concatenate([R[e]["win"][f"{s}|{model}"] for e in evs]) for s in sources}
            c = {s: effs(S[s], kind, p, E) for s in sources}
            dif = c["stored"] - np.mean([c[s] for s in sources[1:]], 0)
            bd = np.array([effs(S["stored"], kind, p, E, w) - np.mean([effs(S[s], kind, p, E, w) for s in sources[1:]], 0) for w in Wb])
            sd = bd.std(0)
            res[f"{name}|{model}"] = {pl: dict(stored=float(c["stored"][i]), draws=[float(c[s][i]) for s in sources[1:]],
                                               diff=float(dif[i]), boot_sd=float(sd[i]), z=float(dif[i] / sd[i]) if sd[i] > 0 else None)
                                      for i, pl in enumerate("UVYa")}
            print(name, model, {pl: (round(v["stored"], 4), round(np.mean(v["draws"]), 4), round(v["diff"], 4), round(v["boot_sd"], 4), round(v["z"], 2))
                                for pl, v in res[f"{name}|{model}"].items()}, flush=True)
    # combined (inverse-variance) stored - draws over the two independent event sets
    for model in MODELS:
        comb = {}
        for pl in "UVYa":
            a_, b_ = res[f"pt|{model}"][pl], res[f"test|{model}"][pl]
            w1, w2 = 1 / a_["boot_sd"] ** 2, 1 / b_["boot_sd"] ** 2
            d = (w1 * a_["diff"] + w2 * b_["diff"]) / (w1 + w2); s = (w1 + w2) ** -0.5
            comb[pl] = dict(diff=float(d), sd=float(s), z=float(d / s))
        res[f"combined|{model}"] = comb
        print("combined", model, {k: (round(v["diff"], 4), round(v["sd"], 4), round(v["z"], 2)) for k, v in comb.items()})
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
