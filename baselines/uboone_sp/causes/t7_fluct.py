#!/usr/bin/env python3
"""T7: does M7dc transmit real cell-to-cell charge fluctuations at the slope its global gain response (~0.67)
suggests? Predictions fixed before running (m7_tests_plan.md, T7):
  confirmed  model slope ~0.6-0.75, inversion slope ~1, model error strongly anti-correlated with the true
             fluctuation (the untransmitted part (1 - b) * delta_true carries much of the error variance);
  killed     model slope ~1: the gain sublinearity is not loss of per-cell fluctuations.

    scripts/helix_run.sh python baselines/uboone_sp/causes/t7_fluct.py --out <json>

Cells: the 48 floor events, truth T (training-label complete truth, as t2t3_model.py) >= 16 k e-, single-particle
(majority share >= 0.9, from m7diag cell records' attribution), tracks with >= 10 such cells in the plane.
delta_x = log x - median over the track's cells of log x, for x = truth, model (fp32 and bf16 raw charge), corpus
Tikhonov (lambda 1e-4 and 1e-3, floor.py). slope b = OLS of delta_model on delta_true (truth is exact, so no
regression dilution). Also: corr(e, delta_true) for e = log model - log T, and the share of var(e) removed by
subtracting -(1 - b) delta_true.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/global/u1/o/oalter/nu/helix")
from helix.probe.resolution import unkey  # noqa: E402

R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
M = f"{R}/m7tests"
Q0 = 2043.2578108909702


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-share", type=float, default=0.9)
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--tmin", type=float, default=16e3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = json.load(open(f"{R}/m7diag/cells/columns.json"))["rec_cols"]
    ci = {c: i for i, c in enumerate(cols)}
    per = {p: {k: [] for k in ("T", "fp32", "bf16", "tik4", "tik3", "trk")} for p in range(3)}
    for f in sorted(glob.glob(f"{M}/t2_fp32_test/ev*.npz")):
        ev = int(os.path.basename(f)[2:5])
        Z = np.load(f); Zb = np.load(f"{M}/t2t3_test/ev{ev:03d}.npz")
        assert (Z["keys"] == Zb["keys"]).all()
        g, fw, ft = unkey(Z["keys"])
        T = Z["Tq"]; ok = Z["covered"] & (T >= a.tmin)
        rec = np.load(f"{R}/m7diag/cells/ev{ev:03d}.npz")["rec"]
        F = np.load(f"{R}/bright/floor/ev{ev:03d}.npz")
        # join attribution on (g, fw, ft)
        rk = (rec[:, ci["g"]].astype(np.int64) * 10_000 + rec[:, ci["fw"]].astype(np.int64)) * 1_000 + rec[:, ci["ft"]].astype(np.int64)
        ck = (g.astype(np.int64) * 10_000 + fw) * 1_000 + ft
        o = np.argsort(rk); rks = rk[o]
        pos = np.clip(np.searchsorted(rks, ck), 0, len(rks) - 1)
        hit = rks[pos] == ck
        r = o[pos]
        share = np.where(hit, rec[r, ci["maj_share"]], 0)
        trk = np.where(hit, rec[r, ci["maj_trk"]], -1).astype(np.int64)
        sel = ok & hit & (share >= a.min_share) & (trk >= 0)
        for gg in range(6):
            m = sel & (g == gg)
            if not m.any():
                continue
            p = gg % 3
            per[p]["T"].append(T[m])
            per[p]["fp32"].append(Q0 * np.expm1(np.maximum(Z["y_M7dc_fp32"][m], 0)))
            per[p]["bf16"].append(Q0 * np.expm1(np.maximum(Zb["y_M7dc_base"][m], 0)))
            per[p]["tik4"].append(F[f"corpus_tik0.0001_g{gg}"][fw[m], ft[m]].astype(np.float64))
            per[p]["tik3"].append(F[f"corpus_tik0.001_g{gg}"][fw[m], ft[m]].astype(np.float64))
            per[p]["trk"].append(ev * 10_000_000 + gg * 1_000_000 + trk[m])        # unique per (event, plane, track)
    out = {}
    for p in range(3):
        D = {k: np.concatenate(v) for k, v in per[p].items()}
        u, inv, cnt = np.unique(D["trk"], return_inverse=True, return_counts=True)
        keep = cnt[inv] >= a.min_cells
        D = {k: v[keep] for k, v in D.items()}
        u, inv = np.unique(D["trk"], return_inverse=True)

        def dev(x):
            lx = np.log(np.maximum(x, 1.0))
            med = np.array([np.median(lx[inv == i]) for i in range(len(u))])
            return lx - med[inv], lx

        dT, lT = dev(D["T"])
        res = dict(n_cells=int(len(dT)), n_tracks=int(len(u)), sd_delta_true=float(dT.std()))
        for k in ("fp32", "bf16", "tik4", "tik3"):
            dX, lX = dev(D[k])
            b = float((dT * dX).sum() / (dT * dT).sum())
            e = lX - lT
            e_c = e - e.mean()
            corr = float(np.corrcoef(e, dT)[0, 1])
            removed = 1 - np.var(e_c + (1 - b) * dT) / np.var(e_c)
            q16, q84 = np.percentile(e, [16, 84])
            q16r, q84r = np.percentile(e + (1 - b) * dT, [16, 84])
            # error variance by scale: between events, between tracks within events, within tracks
            evid = D["trk"] // 10_000_000
            tm = np.array([e[inv == i].mean() for i in range(len(u))])
            ue, einv = np.unique(evid, return_inverse=True)
            em = np.array([e[einv == i].mean() for i in range(len(ue))])
            v_tot = e.var(); v_ev = em[einv].var(); v_trk = (tm[inv] - em[einv]).var(); v_in = (e - tm[inv]).var()
            res[k] = dict(slope=b, corr_err_delta=corr, var_share_removed=float(removed),
                          var_share_event=float(v_ev / v_tot), var_share_track=float(v_trk / v_tot), var_share_within=float(v_in / v_tot),
                          err_hw68=float((q84 - q16) / 2), err_hw68_if_full_transmission=float((q84r - q16r) / 2))
        out["UVY"[p]] = res
        print("UVY"[p], json.dumps(res))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
