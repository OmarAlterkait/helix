#!/usr/bin/env python3
"""T15: does the per-cell error depend on WHERE the cell sits inside its token (the tokenisation's phase)?

A token packs pw wires x pt samples of one band; a 2-wire x 16-tick output cell sits at one of several phases
inside it. Deconvolution is translation-equivariant; patch tokens are not, so a network must learn a separate map
per phase, and a cell at a token edge needs its context from the NEXT token through attention. If tokenisation
limits precision, the error should vary with phase (worse at edges). Per plane and phase, hw68 / median of
log(q / truth) for covered cells with complete truth in [16 k, 64 k) and >= 64 k, from the products calibration dumps
(raw charge maps, ev000-063 minus ev041) of each model given.

    scripts/helix_run.sh python baselines/uboone_sp/causes/t15_phase.py --model M7dc=model_cal_m7dc \
        --model M10lin=model_cal_m10lin --model M11=model_cal_m11 --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
P = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/products"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=dump dir under products/")
    ap.add_argument("--events", default="0-63")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import denoise_products as DP
    from helix.data.denoise import DenoiseEvents
    from helix.probe.resolution import unkey
    cfg = DenoiseEvents("", [], "", items=[]).cfg
    pw, pt = cfg.pw, cfg.pt
    dec0 = 1 << int(np.asarray(cfg.lev)[0]); d0 = float(cfg.delta[0])
    print(f"pw {pw} pt {pt}; band 0 decimation {dec0}, delta {d0}, toff {list(cfg.toff)}", flush=True)
    shapes = DP._shapes()
    lo, hi = (int(x) for x in a.events.split("-"))
    acc = {}
    for spec in a.model:
        name, d = spec.split("=")
        for ev in range(lo, hi + 1):
            f = os.path.join(P, d, f"ev{ev:03d}.npz")
            if not os.path.exists(f):
                continue
            Mz = np.load(f)
            shard, event, _ = DP._event_inputs(ev, None)
            ck, cq = DP._corpus_truth(DP.COMPLETE, shard, event)
            T = DP._maps(ck, cq, shapes)
            for g, sh in shapes.items():
                q, _, cov = DP._dump_planes(Mz, name, g, sh)
                W, Tt = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing="ij")
                m = cov & (T[g] >= 16e3)
                lr = np.log(np.maximum(q[m], 1.0) / T[g][m])
                tq = T[g][m]
                wph = (W[m] * 2) % pw                                         # wire phase in the token
                tau = (Tt[m] * 16 + 8 + np.asarray(cfg.toff)[g % 3]) / dec0 - d0
                tph = np.floor(tau).astype(int) % pt                          # band-0 sample phase in the token
                for kind, ph in (("t", tph), ("w", wph)):
                    for lo_q, tag in ((16e3, "16k"), (64e3, "64k")):
                        sel = (tq >= lo_q) & (tq < (64e3 if lo_q == 16e3 else np.inf))
                        for v in np.unique(ph[sel]):
                            acc.setdefault((name, "UVY"[g % 3], kind, tag, int(v)), []).append(lr[sel & (ph == v)])
            print(f"{name} ev{ev:03d}", flush=True)
    res = {}
    for (name, pl, kind, tag, v), xs in sorted(acc.items()):
        x = np.concatenate(xs)
        if len(x) < 50:
            continue
        a16, a50, a84 = np.percentile(x, [16, 50, 84])
        res.setdefault(name, {}).setdefault(f"{pl}_{kind}_{tag}", {})[v] = dict(hw68=float((a84 - a16) / 2), med=float(a50), n=int(len(x)))
    json.dump(res, open(a.out, "w"), indent=1)
    for name, r in res.items():
        for k, per in r.items():
            print(name, k, " ".join(f"{v}:{s['hw68']:.4f}" for v, s in per.items()))


if __name__ == "__main__":
    main()
