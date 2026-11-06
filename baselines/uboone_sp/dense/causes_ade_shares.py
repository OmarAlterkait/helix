#!/usr/bin/env python3
"""Share accounting for hypotheses A and D from causes2/windows.npz (written by causes_ade.py).

    scripts/helix_run.sh python baselines/uboone_sp/dense/causes_ade_shares.py \
        --out baselines/uboone_sp/dense/results/causes_ade_shares.json

A: the classical-minus-model gap on IDENTICAL waveforms (classical draw: cls vs M6dcn,
   evidence z_k1 = matched filter on that draw's corpus-like input; fresh torch draw:
   clstn vs M6dtn, no per-deposit evidence) decomposed over corpus-input evidence bins:
     info   = gap carried by deposits whose corpus-input evidence is < 2 gate-noise sigma
              (nothing for any detector reading the corpus input to find; KL_k1 also given)
     model  = gap carried by deposits WITH corpus-input evidence (>= 2)
   and a counterfactual for the 'model' part: the model's own evidence->efficiency curve
   measured on the OTHER planes applied to this plane's deposits (Y vs U+V).
D: the Y-minus-plane gap re-weighted to equal deposit SNR (analytic, from noise_vs_hits)
   and to equal corpus-input evidence.
Event bootstrap (200) for every share.
"""
import argparse
import json

import numpy as np

CFS = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
EDGES = np.array([-1e9, 1, 2, 3, 4, 5, 6, 8, 10, 15, 1e9])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Z = np.load(f"{CFS}/causes2/windows.npz")
    kind, p, E, snr, ev = Z["kind"], Z["p"].astype(int), Z["E"], Z["snr"], Z["ev"].astype(int)
    zk, zc, zg = Z["z_k1"], Z["z_corp"], Z["z_gate"]
    iso = kind == "iso"; has = iso & np.isfinite(zk)
    clean = iso & (Z["foreign"] <= 0.25 * Z["q"])
    D = {k[2:]: Z[k] for k in Z.files if k.startswith("D_")}
    evs = np.unique(ev)
    rng = np.random.default_rng(1)
    W = np.stack([np.bincount(rng.integers(0, len(evs), len(evs)), minlength=len(evs)) for _ in range(200)]).astype(float)
    eidx = np.searchsorted(evs, ev)

    def boot(f, sel):
        """f(weights per window) -> scalar; central + 16/84 over event bootstrap."""
        c = f(np.ones(len(ev)) * sel)
        b = [f(W[i][eidx] * sel) for i in range(len(W))]
        return [float(c), float(np.percentile(b, 16)), float(np.percentile(b, 84))]

    res = {"A": {}, "D": {}}
    sets = {"0.1-0.2": iso & (E > 0.1) & (E <= 0.2), "snr3-10": iso & (snr >= 3) & (snr < 10),
            "snr3-10_clean": clean & (snr >= 3) & (snr < 10), "0.1-0.2_clean": clean & (E > 0.1) & (E <= 0.2)}
    for setn, ss in sets.items():
        for pl in (0, 1, 2, 3):
            sel = (ss & has & ((p == pl) if pl < 3 else True)).astype(float)
            if sel.sum() < 30:
                continue
            cls, mod = D["cls"].astype(float), D["M6dcn"].astype(float)
            low = zk < 2
            gap = lambda w: np.sum(w * (cls - mod)) / np.sum(w)
            info = lambda w: np.sum(w * (cls - mod) * low) / np.sum(w)
            evid = lambda w: np.sum(w * (cls - mod) * ~low) / np.sum(w)
            info_kl = lambda w: np.sum(w * (cls - mod) * ~D["KL_k1"]) / np.sum(w)
            r = dict(n=int(sel.sum()), gap=boot(gap, sel), info_part_zlt2=boot(info, sel), evidence_part_zge2=boot(evid, sel),
                     info_part_KLfail=boot(info_kl, sel), frac_deposits_zlt2=float(np.sum(sel * low) / sel.sum()),
                     cls_eff_zlt2=float(np.sum(sel * low * cls) / max(np.sum(sel * low), 1)),
                     model_eff_zlt2=float(np.sum(sel * low * mod) / max(np.sum(sel * low), 1)),
                     cls_eff_zge2=float(np.sum(sel * ~low * cls) / max(np.sum(sel * ~low), 1)),
                     model_eff_zge2=float(np.sum(sel * ~low * mod) / max(np.sum(sel * ~low), 1)))
            if pl == 2:            # counterfactual: Y deposits through the U+V evidence->efficiency curve
                uv = (ss & has & (p < 2)).astype(bool)
                def cf(w):
                    yy = w > 0
                    cur = np.array([mod[uv & (zk >= lo) & (zk < hi)].mean() if (uv & (zk >= lo) & (zk < hi)).sum() >= 10 else np.nan
                                    for lo, hi in zip(EDGES[:-1], EDGES[1:])])
                    ib = np.clip(np.digitize(zk, EDGES) - 1, 0, len(EDGES) - 2)
                    pred = np.where(np.isfinite(cur[ib]), cur[ib], mod)
                    return np.sum(w * pred) / np.sum(w) - np.sum(w * mod) / np.sum(w)
                r["model_gain_if_UV_curve"] = boot(cf, sel)
            res["A"][f"{setn}|{'UVYa'[pl]}"] = r
    # D: plane gaps at matched SNR / matched corpus evidence (bootstrap)
    bins_snr = np.array([-1e9, 1e-9, 3, 5, 7, 10, 15, 1e9])
    for m in ("cls", "M6d", "M6dcn", "M6dtn", "clstn", "KL_gate", "KL_k1", "KL_corp"):
        if m not in D:
            continue
        x = D[m].astype(float)
        for setn in ("0.1-0.2", "0.1-0.2_clean"):
            for pl in (0, 1):
                for zn, zz, bb in (("snr", np.where(np.isfinite(snr), snr, 0), bins_snr), ("z_corp", zc, EDGES), ("z_k1", zk, EDGES)):
                    sp = sets[setn] & (p == pl) & has
                    sy = sets[setn] & (p == 2) & has
                    ib = np.clip(np.digitize(zz, bb) - 1, 0, len(bb) - 2)

                    def raw(w):
                        return np.sum(w * sy * x) / np.sum(w * sy) - np.sum(w * sp * x) / np.sum(w * sp)

                    def resid(w):
                        num = den = 0.0
                        for b in range(len(bb) - 1):
                            npb = np.sum(w * sp * (ib == b))
                            if npb == 0:
                                continue
                            ny = np.sum(w * sy * (ib == b))
                            ey = np.sum(w * sy * (ib == b) * x) / ny if ny >= 5 else np.sum(w * sp * (ib == b) * x) / npb
                            num += npb * (ey - np.sum(w * sp * (ib == b) * x) / npb); den += npb
                        return num / den
                    rr, re_ = boot(raw, np.ones(len(x))), boot(resid, np.ones(len(x)))
                    res["D"][f"{m}|{setn}|{zn}|{'UV'[pl]}"] = dict(raw_gap=rr, residual_gap_at_matched=re_,
                                                                  share_explained=float(1 - re_[0] / rr[0]) if abs(rr[0]) > 1e-9 else None)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res["A"].items():
        print("A", k, {kk: (vv if not isinstance(vv, list) else [round(x, 3) for x in vv]) for kk, vv in v.items()})
    for k, v in res["D"].items():
        if "M6d|" in k or "cls|" in k:
            print("D", k, {kk: (vv if not isinstance(vv, list) else [round(x, 3) for x in vv]) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
