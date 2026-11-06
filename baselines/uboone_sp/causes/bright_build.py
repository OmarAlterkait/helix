#!/usr/bin/env python3
"""Per-cell, per-tile and per-particle records for the bright-charge error budget.

    JAX_PLATFORMS=cpu scripts/helix_run.sh python baselines/uboone_sp/causes/bright_build.py --events 260-387 \
        --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/bright/cells
    (calibration events:  --events 0-63 --skip 41 --cal --out .../bright/cal_cells)

Complete truth as causes/build.py (JAXTPC 1ea2282 hits code on every deposit, no thresholds),
extended with attribution: per (track, cell) charge, the charge-weighted drift distance and
dE/dx of each cell, and per track (from its step deposits, per volume): energy, charge,
principal direction (charge-weighted PCA), length, dE/dx distribution.

Cell records: every fine cell with complete or hits truth >= 1000 e- (2 x 16 cells), with
  ev, g, fw, ft, Th, Tf, drift_cm, dedx, maj_share, n10, maj_trk (row in the particle table), sat,
  cov, and the charge maps: M6d / M6dc (bf16 = dense3, head32, fp32 from bright_dump.py; raw
  charge + presence), M4n (dense2, bf16) and classical tuned (dense cls_fit / cls_fix).
sat: the stored clean sensor has a sample at the 12-bit limit (code 0 or 4095) on one of the
  cell's 2 wires within +-1 cell in time, after the sensor -> drift-time offset (tokenizer toff).
Tiles: 2 x 2, 4 x 4 and 16 x 8 cells, sums of Th, Tf and every charge map (tiles with Tf >= 4 k e-).
Particles: per (g, track) with complete charge >= 1 Q0: charge, E, angles to the drift axis and
  to the wire direction in this plane's frame, length, dE/dx median / 90th pct / Bragg fraction
  (charge with dE/dx > 4 MeV/cm), mean drift, overlap (share of its charge in cells where it holds
  < 90%), saturated share, and sum over cells of share x prediction for every charge map.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "v2")); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(BASE)))
C3 = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
SRC = "/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02"
TOFF = (-17.4, 2.6, 5.5)


def rebuild_attr(E, cfg, nshape):
    import h5py
    import hdf5plugin  # noqa: F401
    import jax
    import jax.numpy as jnp
    import resim as RS
    from tools.wires import compute_wire_distances, prepare_deposit_with_diffusion
    from tools.geometry import calculate_max_diffusion_sigmas
    _, _, sig_t, sig_l = calculate_max_diffusion_sigmas(216.0, 0.16, 12.0e-6, 7.2e-6, 0.3, 0.5)
    Kw, Kt = max(1, int(np.ceil(3.0 * sig_t))), max(1, int(np.ceil(3.0 * sig_l)))
    K = (2 * Kw + 1) * (2 * Kt + 1)
    prep = jax.vmap(prepare_deposit_with_diffusion, in_axes=(0,) * 11 + (None,) * 9)
    fs = h5py.File(os.path.join(SRC, "step", E.run, E.source_file.replace("_sensor_", "_step_")), "r")
    fh = h5py.File(os.path.join(SRC, "hits", E.run, E.source_file.replace("_sensor_", "_hits_")), "r")
    out, tracks = {}, []
    for v in (0, 1):
        vg, hv = fs[f"event_{E.event:03d}"][f"volume_{v}"], fh[f"event_{E.event:03d}"][f"volume_{v}"]
        if "positions" not in vg:
            continue
        org = np.array([vg.attrs["pos_origin_x"], vg.attrs["pos_origin_y"], vg.attrs["pos_origin_z"]])
        pos = vg["positions"][:].astype(np.float64) * float(vg.attrs["pos_step_mm"]) + org
        q = vg["charge"][:].astype(np.float64)
        de = vg["de"][:].astype(np.float64); dx = np.maximum(vg["dx"][:].astype(np.float64), 1e-3)
        dedx = de / (dx / 10.0)                                    # MeV / cm
        d2g = hv["deposit_to_group"][:].astype(np.int64)
        g2t = hv["group_to_track"][:].astype(np.int64)
        t0d, t0g, okg = RS.recover_t0(cfg, hv, pos, q, v)
        qg = np.bincount(d2g, weights=q, minlength=len(t0g))
        trk_g = g2t[: len(t0g)]
        tu, ti = np.unique(trk_g, return_inverse=True)
        num = np.bincount(ti, weights=np.where(okg, qg * np.nan_to_num(t0g), 0.0), minlength=len(tu))
        den = np.bincount(ti, weights=np.where(okg, qg, 0.0), minlength=len(tu))
        t0trk = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
        t0 = np.where(okg, t0g, t0trk[ti])[d2g]
        trk = g2t[d2g]                                             # track id per deposit
        # ---- track table (this volume)
        ut, inv = np.unique(trk, return_inverse=True)
        for k, t in enumerate(ut):
            m = inv == k
            w_ = q[m]
            if w_.sum() <= 0 or not np.isfinite(t0[m]).any():
                continue
            P = pos[m]
            mu = (P * w_[:, None]).sum(0) / w_.sum()
            if m.sum() >= 2:
                Cv = ((P - mu).T * w_) @ (P - mu) / w_.sum()
                ev_, evec = np.linalg.eigh(Cv)
                d = evec[:, -1]
                proj = (P - mu) @ d
                length = float(proj.max() - proj.min())
            else:
                d, length = np.array([1.0, 0, 0]), 0.0
            ww = w_ / w_.sum()
            dd_ = np.sort(dedx[m]); cw = np.cumsum(ww[np.argsort(dedx[m])])
            tracks.append(dict(v=v, trk=int(t), E=float(de[m].sum()), q=float(w_.sum()), dir=d.tolist(), length_mm=length,
                               dedx_med=float(dd_[np.searchsorted(cw, 0.5)]), dedx_p90=float(dd_[min(np.searchsorted(cw, 0.9), len(dd_) - 1)]),
                               bragg_frac=float(w_[dedx[m] > 4.0].sum() / w_.sum())))
        vgeo = cfg.volumes[v]
        posl = pos.copy()
        posl[:, 0] = vgeo.drift_direction * (vgeo.x_anode_cm * 10.0 - pos[:, 0])
        posl[:, 1] -= vgeo.yz_center_cm[0] * 10.0
        posl[:, 2] -= vgeo.yz_center_cm[1] * 10.0
        pcm = posl / 10.0
        for p in range(3):
            gid = 3 * v + p
            nw, nt = nshape[gid]
            tk, dd = RS.plane_ticks(cfg, pos, v, p)
            tick = tk + t0
            ok = np.isfinite(tick) & (tick >= 0) & (tick < nt * 0.5) & (q > 0)
            ci, cd = compute_wire_distances(jnp.asarray(pcm[ok, 1:3]), vgeo.angles_rad[p], vgeo.wire_spacings_cm[p],
                                            vgeo.max_wire_indices[p], vgeo.index_offsets[p])
            dt_ = dd[ok] / 0.16
            att = np.exp(-dt_ / 10000.0)
            n = int(ok.sum())
            W, Tt, V = [], [], []
            for a in range(0, n, 20000):
                sl = slice(a, a + 20000)
                m_ = min(n, a + 20000) - a
                one = jnp.ones(m_, jnp.float32)
                wi, ti_, val = prep(jnp.asarray(q[ok][sl], jnp.float32), jnp.asarray(dt_[sl], jnp.float32),
                                    jnp.asarray(tick[ok][sl], jnp.float32), jnp.asarray(dd[ok][sl], jnp.float32),
                                    ci[sl], cd[sl], jnp.asarray(att[sl], jnp.float32), one * 0.5, one * 0.5, one,
                                    jnp.ones(m_, bool), Kw, Kt, vgeo.wire_spacings_cm[p], 0.5, 7.2e-6, 12.0e-6, 0.16, nw, nt)
                W.append(np.asarray(wi)); Tt.append(np.asarray(ti_)); V.append(np.asarray(val, np.float64))
            W, Tt, V = (np.concatenate(x).reshape(-1) for x in (W, Tt, V))
            D = np.repeat(np.flatnonzero(ok), K)
            inb = (W >= 0) & (W < nw) & (Tt >= 0) & (Tt < nt) & (V > 0)
            W, Tt, V, D = W[inb], Tt[inb], V[inb], D[inb]
            ncw, nct = -(-nw // 2), -(-nt // 16)
            cell = (W // 2) * nct + Tt // 16
            Tf = np.bincount(cell, weights=V, minlength=ncw * nct)
            Dr = np.bincount(cell, weights=V * dd[D], minlength=ncw * nct)
            Dx = np.bincount(cell, weights=V * dedx[D], minlength=ncw * nct)
            key = trk[D].astype(np.int64) * (ncw * nct) + cell
            uk, iv = np.unique(key, return_inverse=True)
            s = np.bincount(iv, weights=V)
            out[gid] = dict(Tf=Tf.reshape(ncw, nct), drift=(Dr / np.maximum(Tf, 1e-9)).reshape(ncw, nct),
                            dedx=(Dx / np.maximum(Tf, 1e-9)).reshape(ncw, nct),
                            tc_trk=(uk // (ncw * nct)).astype(np.int64), tc_cell=(uk % (ncw * nct)).astype(np.int64), tc_q=s,
                            v=v, p=p)
    return out, tracks


def sat_mask(C, ped, p, shape):
    hi, lo = 4095 - ped, -ped
    w, t = np.nonzero((C >= hi) | (C <= lo))
    M = np.zeros(shape, bool)
    if len(w):
        fw = w // 2
        ft = np.floor((t - TOFF[p]) / 16).astype(int)
        for d in (-1, 0, 1):
            f2 = np.clip(ft + d, 0, shape[1] - 1)
            M[fw, f2] = True
    return M


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
    cfg = create_sim_config(yaml.safe_load(open(os.path.join(RS.JAXTPC, "config", "cubic_wireplane_config.yaml"))),
                            total_pad=1000, include_digitize=True, num_s=16)
    reg = registry()
    E = Event(ev)
    nshape = {g: (reg[g]["n_wires"], reg[g]["n_ticks"]) for g in range(6)}
    A, tracks = rebuild_attr(E, cfg, nshape)
    tid = {(t["v"], t["trk"]): i for i, t in enumerate(tracks)}
    if cal:
        srcs = {"bf16": os.path.join(C3, "bright", "cal_model", "bf16", f"ev{ev:03d}.npz")}
    else:
        srcs = {"bf16": os.path.join(C3, "dense3", "model", f"ev{ev:03d}.npz"),
                "head32": os.path.join(C3, "bright", "model", "head32", f"ev{ev:03d}.npz"),
                "fp32": os.path.join(C3, "bright", "model", "fp32", f"ev{ev:03d}.npz"),
                "m4n": os.path.join(C3, "dense2", "model", f"ev{ev:03d}.npz")}
    Z = {k: np.load(f) for k, f in srcs.items() if os.path.exists(f)}
    ost61 = set(x.strip() for x in open(os.path.join(C3, "causes", "ost61.txt")) if x.strip())
    cls_dirs = ["/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/cls_fit", os.path.join(C3, "dense2", "cls_fix")]
    recs, tiles, parts = [], [], []
    names = []
    for g in range(6):
        p = g % 3
        C = E.clean(g, reg)
        Th = fine_cells(E.truth(g, C.shape).astype(np.float64))
        shape = Th.shape
        if g not in A:
            continue
        a = A[g]
        Tf = a["Tf"]
        maps = {}
        for k, z in Z.items():
            to = int(z["toff"])
            crop = lambda X: np.asarray(X[:shape[0], to:to + shape[1]], np.float64)
            tags = ("M4n",) if k == "m4n" else ("M6d", "M6dc")
            for t in tags:
                if f"q_{t}_g{g}" not in z.files:
                    continue
                qq, pp = crop(z[f"q_{t}_g{g}"]), crop(z[f"p_{t}_g{g}"])
                lab = f"{t}_{k}" if k != "m4n" else "M4n_bf16"
                maps[lab] = np.where(pp > 0.5, qq, 0.0)          # gated product
                if t == "M6d":
                    maps[lab + "_raw"] = qq
                if k == "bf16" and t == "M6d":
                    maps["cov"] = crop(z[f"cov_g{g}"]).astype(np.float64)
        if not cal:
            fc = next(x for x in (os.path.join(d, f"ev{ev:03d}_g{g}.npz") for d in cls_dirs) if os.path.exists(x) and x not in ost61)
            maps["cls_tuned"] = np.asarray(np.load(fc)["tuned_q"], np.float64)
        mnames = sorted(k for k in maps if k != "cov")
        if not names:
            names = mnames
        assert mnames == names, (mnames, names)
        cov = maps.get("cov", np.zeros(shape))
        sat = sat_mask(C, reg[g]["pedestal"], p, shape)
        # majority track per cell, n tracks >= 10%
        ncell = shape[0] * shape[1]
        o = np.argsort(a["tc_cell"]); tcc, tct, tcq = a["tc_cell"][o], a["tc_trk"][o], a["tc_q"][o]
        st = np.searchsorted(tcc, np.arange(ncell)); en = np.searchsorted(tcc, np.arange(ncell), side="right")
        Tf_flat = Tf.ravel()
        # vectorised majority: max per cell via sort by (cell, q)
        o2 = np.lexsort((tcq, tcc)); last = np.r_[tcc[o2][1:] != tcc[o2][:-1], True]
        maj_cell, maj_trk, maj_q = tcc[o2][last], tct[o2][last], tcq[o2][last]
        Mtrk = np.full(ncell, -1, np.int64); Mtrk[maj_cell] = [tid.get((a["v"], int(t)), -1) for t in maj_trk]
        Msh = np.zeros(ncell); Msh[maj_cell] = maj_q / np.maximum(Tf_flat[maj_cell], 1e-9)
        n10 = np.bincount(tcc, weights=(tcq >= 0.1 * Tf_flat[tcc]), minlength=ncell)
        fw, ft = np.divmod(np.arange(ncell), shape[1])
        sel = ((Tf_flat >= 1000) | (Th.ravel() >= 1000))
        cols = [np.full(sel.sum(), ev), np.full(sel.sum(), g), fw[sel], ft[sel], Th.ravel()[sel], Tf_flat[sel],
                a["drift"].ravel()[sel], a["dedx"].ravel()[sel], Msh[sel], n10[sel], Mtrk[sel], sat.ravel()[sel], cov.ravel()[sel]]
        cols += [maps[k].ravel()[sel] for k in names]
        recs.append(np.stack(cols, 1))
        for (ta, tb, lab) in ((2, 2, 0), (4, 4, 1), (16, 8, 2)):
            nw_, nt_ = (shape[0] // ta) * ta, (shape[1] // tb) * tb
            tl = lambda X: X[:nw_, :nt_].reshape(nw_ // ta, ta, nt_ // tb, tb).sum((1, 3)).ravel()
            tf = tl(Tf); k_ = tf >= 4000
            tiles.append(np.stack([np.full(k_.sum(), lab), np.full(k_.sum(), g), tl(Th)[k_], tf[k_], tl(sat.astype(float))[k_] > 0]
                                  + [tl(maps[k])[k_] for k in names], 1))
        # particles on this plane
        sh = tcq / np.maximum(Tf_flat[tcc], 1e-9)
        ut, iv = np.unique(tct, return_inverse=True)
        qp = np.bincount(iv, weights=tcq)
        dr = np.bincount(iv, weights=tcq * a["drift"].ravel()[tcc])
        ovl = np.bincount(iv, weights=tcq * (sh < 0.9))
        satq = np.bincount(iv, weights=tcq * sat.ravel()[tcc])
        preds = [np.bincount(iv, weights=sh * maps[k].ravel()[tcc]) for k in names]
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
            th_drift = np.degrees(np.arctan2(cx, cp))              # 0: across wires at one time; 90: along the drift
            th_wire = np.degrees(np.arctan2(cw, np.hypot(cx, cp)))  # 90: along the wire
            parts.append([ev, g, i, qp[j], T_["E"], th_drift, th_wire, T_["length_mm"], T_["dedx_med"], T_["dedx_p90"],
                          T_["bragg_frac"], dr[j] / qp[j], ovl[j] / qp[j], satq[j] / qp[j], th_sh[j]] + [x[j] for x in preds])
    np.savez(out + ".tmp.npz", rec=np.concatenate(recs).astype(np.float64), tiles=np.concatenate(tiles).astype(np.float64),
             parts=np.asarray(parts, np.float64), names=np.array(names))
    os.replace(out + ".tmp.npz", out)
    return ev, f"{time.time() - t00:.0f}s cells {sum(len(r) for r in recs)} parts {len(parts)}"


def main():
    sys.path.insert(0, os.path.join(BASE, "v2"))
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
    json.dump(dict(rec_cols=["ev", "g", "fw", "ft", "Th", "Tf", "drift_cm", "dedx", "maj_share", "n10", "maj_trk", "sat", "cov"],
                   tile_cols=["level", "g", "Th", "Tf", "sat"],
                   part_cols=["ev", "g", "trk_row", "q", "E", "th_drift", "th_wire", "length_mm", "dedx_med", "dedx_p90",
                              "bragg_frac", "drift_cm", "overlap", "sat_frac", "Th_share"]), open(os.path.join(a.out, "columns.json"), "w"))
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.workers, maxtasksperchild=2) as pool:
        for ev, msg in pool.imap_unordered(per_event, [(e, a.out, a.cal) for e in evs]):
            print(f"ev{ev:03d}: {msg}", flush=True)


if __name__ == "__main__":
    main()
