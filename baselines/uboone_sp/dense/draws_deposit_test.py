#!/usr/bin/env python3
"""Deposit-level exchangeability test: is the STORED corpus realisation just one more
draw? For each isolated deposit, the detection probability over the fresh draws p_i
(each draw at its own near-pp threshold) vs the stored outcome s_i;
  Z = sum(s_i - p_i) / sqrt(sum p_i (1 - p_i))       (deposit noise variance)
and the same at a COMMON threshold (each plane's mean fresh-draw threshold applied to
every source), which separates a deposit-score shift from a threshold shift. Also the
spread of the noise-window threshold across draws.

    scripts/helix_run.sh python baselines/uboone_sp/dense/draws_deposit_test.py \
        --draws 0-9 --out baselines/uboone_sp/dense/results/draws_deposit_test.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import draw_metrics as DM  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draws", default="0-9")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.draws.split("-"))
    sources = ["stored"] + [f"d{k}" for k in range(lo, hi + 1)]
    with Pool(a.workers) as pool:
        RES = dict(pool.map(DM.per_event, [(e, sources) for e in DM.EVS]))
    meta = [m for e in DM.EVS for m in RES[e]["meta"]]
    evi = np.array([e for e in DM.EVS for _ in RES[e]["meta"]])
    kind = np.array([m["kind"] for m in meta]); p = np.array([m["p"] for m in meta]); E = np.array([m["E"] for m in meta])
    A = np.load(DM.ANNOT)
    ann = {k: (float(s) if pk > 0 else 0.0, bool(fo <= 0.25 * q_)) for k, s, pk, fo, q_ in
           zip(zip(A["ev"].tolist(), A["g"].tolist(), A["w0"].tolist(), A["t0"].tolist()), A["snr"], A["peak"], A["foreign"], A["q"])}
    snr = np.array([ann.get((evi[i], m["g"], m["w0"], m["t0"]), (np.nan, False))[0] if m["kind"] == "iso" else np.nan for i, m in enumerate(meta)])
    res = {}
    for model in DM.MODELS:
        for sk in ("p", "g"):
            S = {s: np.concatenate([RES[e]["win"][f"{s}|{model}"][sk] for e in DM.EVS]) for s in sources}
            thr = {s: {pl: float(np.quantile(S[s][(kind == "bgn") & (p == pl)], 0.99)) for pl in range(3)} for s in sources}
            tcom = {pl: float(np.mean([thr[s][pl] for s in sources[1:]])) for pl in range(3)}
            for mode in ("own", "common"):
                det = {s: S[s] > np.array([(thr[s] if mode == "own" else tcom)[x] for x in p]) for s in sources}
                for setn, sel0 in (("0.1-0.2", (kind == "iso") & (E > 0.1) & (E <= 0.2)), ("snr7-10", (kind == "iso") & (snr >= 7) & (snr < 10)),
                                   ("snr3-7", (kind == "iso") & (snr >= 3) & (snr < 7))):
                    for pl in (0, 1, 2, 3):
                        sel = sel0 & ((p == pl) if pl < 3 else True)
                        pi = np.mean([det[s][sel] for s in sources[1:]], 0)
                        si = det["stored"][sel].astype(float)
                        var = np.sum(pi * (1 - pi))
                        res[f"{model}|{sk}|{mode}|{setn}|{'UVYa'[pl]}"] = dict(
                            n=int(sel.sum()), stored=float(si.mean()), draws_mean=float(pi.mean()),
                            diff=float(si.mean() - pi.mean()), Z=float((si - pi).sum() / np.sqrt(var)) if var > 0 else None,
                            frac_deposits_always=float(np.mean(pi == 1)), frac_never=float(np.mean(pi == 0)))
            res[f"{model}|{sk}|thresholds"] = {s: thr[s] for s in sources}
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        if "thresholds" in k or "|p|" not in k:
            continue
        if k.endswith(("|Y", "|a")):
            print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
