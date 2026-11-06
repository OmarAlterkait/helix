#!/usr/bin/env python3
"""Per-cell / tile / particle records for the M7 diagnostics (m7_diagnostics.md): complete truth
and attribution exactly as bright_build.py (its rebuild_attr / sat_mask, imported unchanged), plus
the posterior products of scripts/denoise_products.py for M6d / M6dc / M7d / M7dc, and two
deploy-time input measures per cell.

    JAX_PLATFORMS=cpu scripts/helix_run.sh python baselines/uboone_sp/causes/m7diag_build.py --events 260-387 \
        --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/m7diag/cells
    (calibration events:  --events 0-63 --skip 41 --cal --out .../m7diag/cal_cells)

Differences from bright_build.py:
  * cell records for EVERY cell with Tf > 0, Th > 0, or any model's calo > 1 e- or gated charge > 0 (not
    only >= 1 k e-; the rest carry no truth and < 1 e- of calo),
    so faint predicted bins and the truly-uncharged cells under them are present;
  * per model M: raw charge q (Q0 expm1(max(y,0))), presence p, p_cal, the conditional charge estimate
    qJ = Q0 (exp(y + sigma^2/2) - 1), calo, clean, and pdens (box sum of the model's own raw charge, the
    products' stratum variable) -- from products params_<M>.json fitted on calibration events;
  * ncoef: kept corpus coefficients (bands < n_bands) projecting into the fine cell (helix.data.denoise
    rows' projection), ncoef5: its 5 x 5-cell box sum. Both are deploy-time quantities;
  * tiles and particles carry raw / gated (p > 0.5) / calo / clean for every model.
Sources: test M6* = dense3/model (model_dump2, bf16), M7* = products/model_test_m7 (model_dump2, bf16),
optional fp32 M7 maps (bright_dump.py --precision fp32) as M7d32 / M7dc32 (raw q, p, products with the
M7 params); calibration M6* = draws/pt_stored, M7d = products/model_cal_m7d, M7dc = products/model_cal_m7dc.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BASE))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "v2")); sys.path.insert(0, HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
C3 = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
PS_CORPUS, CFS_CORPUS = "/pscratch/sd/o/oalter/coeff_tpc_r1", "/global/cfs/cdirs/m5238/users/oalter/coeff_tpc_r1"
MODELS = ("M6d", "M6dc", "M7d", "M7dc")


def sources(cal):
    P = os.path.join(C3, "products")
    if cal:
        return [("M6d", os.path.join(C3, "draws", "pt_stored"), "M6d", "M6d"),
                ("M6dc", os.path.join(C3, "draws", "pt_stored"), "M6dc", "M6dc"),
                ("M7d", os.path.join(P, "model_cal_m7d"), "M7d", "M7d"),
                ("M7dc", os.path.join(P, "model_cal_m7dc"), "M7dc", "M7dc")]
    return [("M6d", os.path.join(C3, "dense3", "model"), "M6d", "M6d"),
            ("M6dc", os.path.join(C3, "dense3", "model"), "M6dc", "M6dc"),
            ("M7d", os.path.join(P, "model_test_m7"), "M7d", "M7d"),
            ("M7dc", os.path.join(P, "model_test_m7"), "M7dc", "M7dc"),
            ("M7d32", os.path.join(C3, "m7diag", "fp32"), "M7d", "M7d"),
            ("M7dc32", os.path.join(C3, "m7diag", "fp32"), "M7dc", "M7dc")]


def coef_counts(ev, shapes):
    """Kept coefficients per fine cell (the projection of helix.data.denoise.DenoiseEvents.rows)."""
    import h5py
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents
    from helix.model.tokenize import tick_of_tau
    from helix.probe.resolution import FT, FW
    z = np.load(os.path.join(TRUTH, f"ev{ev:03d}.npz"), allow_pickle=True)
    shard, event = str(z["shard"]).replace(PS_CORPUS, CFS_CORPUS), int(z["event"])
    with h5py.File(shard, "r") as fh:
        pos = int(np.searchsorted(fh["ident"]["event"][:], event))
    cfg = DenoiseEvents("", [], "", items=[]).cfg
    ce = read_coeff_event(shard, pos)
    k = ce.band < cfg.n_bands
    ct = tick_of_tau(ce.tau[k], ce.plane_gid[k], ce.band[k], cfg)
    g, fw, ft = ce.plane_gid[k], ce.wire[k] // FW, np.floor(ct / FT).astype(np.int64)
    out = {}
    for gg, (nw, nt) in shapes.items():
        m = (g == gg) & (fw >= 0) & (fw < nw) & (ft >= 0) & (ft < nt)
        out[gg] = np.bincount(fw[m] * nt + ft[m], minlength=nw * nt).reshape(nw, nt).astype(np.float64)
    return out


def per_event(job):
    ev, out_dir, cal = job
    out = os.path.join(out_dir, f"ev{ev:03d}.npz")
    if os.path.exists(out):
        return ev, "exists"
    t00 = time.time()
    import yaml
    import resim as RS
    sys.path.insert(0, RS.JAXTPC)
    from tools.config import create_sim_config
    from sp.data import Event, registry
    from sp.pipeline import fine_cells
    from bright_build import rebuild_attr, sat_mask
    import denoise_products as DP
    cfg = create_sim_config(yaml.safe_load(open(os.path.join(RS.JAXTPC, "config", "cubic_wireplane_config.yaml"))),
                            total_pad=1000, include_digitize=True, num_s=16)
    reg = registry()
    E = Event(ev)
    nshape = {g: (reg[g]["n_wires"], reg[g]["n_ticks"]) for g in range(6)}
    A, tracks = rebuild_attr(E, cfg, nshape)
    tid = {(t["v"], t["trk"]): i for i, t in enumerate(tracks)}
    srcs = [s for s in sources(cal) if os.path.exists(os.path.join(s[1], f"ev{ev:03d}.npz"))]
    assert [s[0] for s in srcs][:4] == list(MODELS), srcs
    Z = {s[0]: np.load(os.path.join(s[1], f"ev{ev:03d}.npz")) for s in srcs}
    PRM = {s[3]: json.load(open(os.path.join(C3, "products", f"params_{s[3]}.json")))[s[3]] for s in srcs}
    labs = [s[0] for s in srcs]
    cshape = {g: (-(-nshape[g][0] // 2), -(-nshape[g][1] // 16)) for g in range(6)}
    NC = coef_counts(ev, cshape)
    recs, tiles, parts = [], [], []
    for g in range(6):
        p = g % 3
        C = E.clean(g, reg)
        Th = fine_cells(E.truth(g, C.shape).astype(np.float64))
        shape = Th.shape
        if g not in A:
            continue
        a = A[g]
        Tf = a["Tf"]
        assert Tf.shape == shape, (Tf.shape, shape)
        per = {}
        for lab, d, mt, pt in srcs:
            z = Z[lab]
            to = int(z["toff"])
            crop = lambda X: np.asarray(X[:shape[0], to:to + shape[1]], np.float64)
            q, pp, cov = crop(z[f"q_{mt}_g{g}"]), crop(z[f"p_{mt}_g{g}"]), crop(z[f"cov_g{g}"]) > 0
            prm = PRM[pt]
            y = np.log1p(np.maximum(q, 0) / prm["q0"])
            pd = DP.box_excl(q)
            P_ = DP.products(y, pp, p, pd, prm, cov=cov)
            s2 = np.asarray(prm["sigma2"])[DP.ybin(y), DP.dbin(pd)]
            qJ = prm["q0"] * np.expm1(y + s2 / 2) * (y > 0)
            per[lab] = dict(q=q, p=pp, pcal=P_["p_cal"], qJ=qJ, calo=P_["calo"], clean=P_["clean"], pd=pd,
                            gated=np.where(pp > 0.5, q, 0.0), cov=cov)
        cov = per["M6d"]["cov"]
        nc = NC[g][:shape[0], :shape[1]]
        P5 = np.pad(nc, 2); S = P5.cumsum(0).cumsum(1); S = np.pad(S, ((1, 0), (1, 0)))
        nc5 = S[5:, 5:] - S[:-5, 5:] - S[5:, :-5] + S[:-5, :-5]
        sat = sat_mask(C, reg[g]["pedestal"], p, shape)
        ncell = shape[0] * shape[1]
        tcc, tct, tcq = a["tc_cell"], a["tc_trk"], a["tc_q"]
        Tf_flat = Tf.ravel()
        o2 = np.lexsort((tcq, tcc)); last = np.r_[tcc[o2][1:] != tcc[o2][:-1], True]
        maj_cell, maj_trk, maj_q = tcc[o2][last], tct[o2][last], tcq[o2][last]
        Mtrk = np.full(ncell, -1, np.int64); Mtrk[maj_cell] = [tid.get((a["v"], int(t)), -1) for t in maj_trk]
        Msh = np.zeros(ncell); Msh[maj_cell] = maj_q / np.maximum(Tf_flat[maj_cell], 1e-9)
        n10 = np.bincount(tcc, weights=(tcq >= 0.1 * Tf_flat[tcc]), minlength=ncell)
        fw, ft = np.divmod(np.arange(ncell), shape[1])
        anyq = np.zeros(ncell, bool)
        for lab in labs:
            anyq |= (per[lab]["calo"].ravel() > 1.0) | (per[lab]["gated"].ravel() > 0)
        sel = (Tf_flat > 0) | (Th.ravel() > 0) | anyq
        cols = [np.full(sel.sum(), ev), np.full(sel.sum(), g), fw[sel], ft[sel], Th.ravel()[sel], Tf_flat[sel],
                a["drift"].ravel()[sel], a["dedx"].ravel()[sel], Msh[sel], n10[sel], Mtrk[sel], sat.ravel()[sel],
                cov.ravel()[sel], nc.ravel()[sel], nc5.ravel()[sel]]
        for lab in labs:
            cols += [per[lab][k].ravel()[sel] for k in ("q", "p", "pcal", "qJ", "calo", "clean", "pd")]
        recs.append(np.stack(cols, 1).astype(np.float32))
        tmaps = [(lab, k) for lab in labs for k in ("q", "gated", "calo", "clean")]
        for (ta, tb, lv) in ((2, 2, 0), (4, 4, 1), (16, 8, 2)):
            nw_, nt_ = (shape[0] // ta) * ta, (shape[1] // tb) * tb
            tl = lambda X: X[:nw_, :nt_].reshape(nw_ // ta, ta, nt_ // tb, tb).sum((1, 3)).ravel()
            tf = tl(Tf); k_ = tf >= 4000
            tiles.append(np.stack([np.full(k_.sum(), lv), np.full(k_.sum(), g), np.full(k_.sum(), ev), tl(Th)[k_], tf[k_],
                                   tl(sat.astype(float))[k_] > 0] + [tl(per[l][k])[k_] for l, k in tmaps], 1))
        sh = tcq / np.maximum(Tf_flat[tcc], 1e-9)
        ut, iv = np.unique(tct, return_inverse=True)
        qp = np.bincount(iv, weights=tcq)
        dr = np.bincount(iv, weights=tcq * a["drift"].ravel()[tcc])
        ovl = np.bincount(iv, weights=tcq * (sh < 0.9))
        satq = np.bincount(iv, weights=tcq * sat.ravel()[tcc])
        preds = [np.bincount(iv, weights=sh * per[l][k].ravel()[tcc]) for l, k in tmaps]
        th_sh = np.bincount(iv, weights=sh * Th.ravel()[tcc])
        ang = cfg.volumes[a["v"]].angles_rad[p]
        u_p = np.array([0.0, np.sin(ang), np.cos(ang)]); u_w = np.array([0.0, np.cos(ang), -np.sin(ang)])
        for j, t in enumerate(ut):
            if qp[j] < 2043.0:
                continue
            i = tid.get((a["v"], int(t)), -1)
            if i < 0:
                continue
            T_ = tracks[i]; d = np.array(T_["dir"])
            cx, cp, cw = abs(d[0]), abs(d @ u_p), abs(d @ u_w)
            parts.append([ev, g, i, qp[j], T_["E"], np.degrees(np.arctan2(cx, cp)), np.degrees(np.arctan2(cw, np.hypot(cx, cp))),
                          T_["length_mm"], T_["dedx_med"], T_["dedx_p90"], T_["bragg_frac"], dr[j] / qp[j], ovl[j] / qp[j],
                          satq[j] / qp[j], th_sh[j]] + [x[j] for x in preds])
    np.savez(out + ".tmp.npz", rec=np.concatenate(recs), tiles=np.concatenate(tiles).astype(np.float64),
             parts=np.asarray(parts, np.float64), models=np.array(labs))
    os.replace(out + ".tmp.npz", out)
    return ev, f"{time.time() - t00:.0f}s cells {sum(len(r) for r in recs)} parts {len(parts)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--skip", default="")
    ap.add_argument("--cal", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    from classical import parse_events
    os.makedirs(a.out, exist_ok=True)
    skip = {int(x) for x in a.skip.split(",") if x}
    evs = [e for e in parse_events(a.events) if e not in skip]
    json.dump(dict(rec_cols=["ev", "g", "fw", "ft", "Th", "Tf", "drift_cm", "dedx", "maj_share", "n10", "maj_trk", "sat", "cov",
                             "ncoef", "ncoef5"],
                   rec_model_cols=["q", "p", "pcal", "qJ", "calo", "clean", "pd"],
                   tile_cols=["level", "g", "ev", "Th", "Tf", "sat"], tile_model_cols=["q", "gated", "calo", "clean"],
                   part_cols=["ev", "g", "trk_row", "q", "E", "th_drift", "th_wire", "length_mm", "dedx_med", "dedx_p90",
                              "bragg_frac", "drift_cm", "overlap", "sat_frac", "Th_share"],
                   part_model_cols=["q", "gated", "calo", "clean"]), open(os.path.join(a.out, "columns.json"), "w"))
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.workers, maxtasksperchild=2) as pool:
        for ev, msg in pool.imap_unordered(per_event, [(e, a.out, a.cal) for e in evs]):
            print(f"ev{ev:03d}: {msg}", flush=True)


if __name__ == "__main__":
    main()
