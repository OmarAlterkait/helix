#!/usr/bin/env python3
"""First-principles C, completeness check: could a particle with NO hits sit inside the readout window?

    scripts/helix_run.sh python baselines/uboone_sp/causes/firstp_hitless.py \
        --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/firstp/hitless.json --workers 32

The complete truth (build_complete_corpus_truth.py) places a deposit in time with its group's
hits; particles with no hits at all are left out as out-of-window (``q_unplaced``, 26-30 % of
the deposit charge). That is safe only if such a particle WOULD have produced a hit had it been
in the window. Its per-sample amplitudes do not depend on its unknown t0 (diffusion and
attenuation depend on the drift distance only), so we re-simulate every hit-less particle at an
arbitrary in-window t0 and take its largest per-group x wire x tick sample over the three
planes: > 25 e- means it would have left hits, so it is genuinely out of window; <= 25 e- means
it is AMBIGUOUS -- it could be in the window, invisible to both truths.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from firstp_windows import JAXTPC, SOURCE, TRUTH  # noqa: E402


def one(f):
    import h5py
    import hdf5plugin  # noqa: F401
    import jax.numpy as jnp
    from build_complete_corpus_truth import (CHUNK, DT_US, D_L, D_T, LIFETIME_US, PRE_WINDOW_US, V_CM_US,
                                             _init, _local, _t0)
    S = _init(JAXTPC)
    cfg, Kw, Kt, prep, cwd = S["cfg"], S["Kw"], S["Kt"], S["prep"], S["cwd"]
    nt = int(cfg.num_time_steps)
    z = np.load(f, allow_pickle=True)
    shard, event = str(z["shard"]), int(z["event"])
    run = os.path.basename(os.path.dirname(shard)); tag = os.path.basename(shard)[-7:-3]
    rows = []
    t00 = time.time()
    with h5py.File(os.path.join(SOURCE, "step", run, f"sim_wire_step_{tag}.h5"), "r") as fs, \
            h5py.File(os.path.join(SOURCE, "hits", run, f"sim_wire_hits_{tag}.h5"), "r") as fh:
        se, he = fs[f"event_{event:03d}"], fh[f"event_{event:03d}"]
        for vk in sorted(x for x in se if x.startswith("volume_")):
            v = int(vk.split("_")[1])
            vg = se[vk]
            if "positions" not in vg or vk not in he:
                continue
            org = np.array([vg.attrs["pos_origin_x"], vg.attrs["pos_origin_y"], vg.attrs["pos_origin_z"]])
            pos = vg["positions"][:].astype(np.float64) * float(vg.attrs["pos_step_mm"]) + org
            q = vg["charge"][:].astype(np.float64); de = vg["de"][:].astype(np.float64)
            pl = _local(cfg, v, pos)
            t0, d2g = _t0(cfg, he[vk], pl, q, v)
            g2t = he[vk]["group_to_track"][:].astype(np.int64)
            un = ~np.isfinite(t0) & (q > 0)
            if not un.any():
                continue
            trk = g2t[d2g]
            tu, ti = np.unique(trk[un], return_inverse=True)
            E = np.bincount(ti, weights=de[un]); Q = np.bincount(ti, weights=q[un])
            smax = np.zeros(len(tu))
            vgeo = cfg.volumes[v]
            idx = np.nonzero(un)[0]
            for p in range(3):
                nw = int(vgeo.num_wires[p])
                dd = pl[idx, 0] - vgeo.plane_distances_cm[p]
                tdr = dd / V_CM_US
                tick = tdr + PRE_WINDOW_US + 200.0                     # arbitrary in-window t0 = 200 us
                ok = (tick >= 0) & (tick < nt * DT_US)
                n = int(ok.sum())
                if not n:
                    continue
                ci, cd = cwd(jnp.asarray(pl[idx[ok], 1:3]), vgeo.angles_rad[p], vgeo.wire_spacings_cm[p],
                             vgeo.max_wire_indices[p], vgeo.index_offsets[p])
                ci, cd = np.asarray(ci), np.asarray(cd)
                qq, tt, dr, dt_ = q[idx[ok]], tick[ok], dd[ok], tdr[ok]
                att = np.exp(-dt_ / LIFETIME_US)
                k_ = (2 * Kw + 1) * (2 * Kt + 1)
                W, T, V, G = [], [], [], []
                for a in range(0, n, CHUNK):
                    m = min(n, a + CHUNK) - a
                    pad = lambda x, dt=np.float32: np.pad(np.asarray(x[a:a + m], dt), (0, CHUNK - m))
                    one_ = np.ones(CHUNK, np.float32)
                    wi, ti_, val = prep(pad(qq), pad(dt_), pad(tt), pad(dr), pad(ci, np.int32), pad(cd), pad(att),
                                        one_ * 0.5, one_ * 0.5, one_, np.arange(CHUNK) < m,
                                        Kw, Kt, float(vgeo.wire_spacings_cm[p]), DT_US, D_L, D_T, V_CM_US, nw, nt)
                    W.append(np.asarray(wi)[:m].reshape(-1)); T.append(np.asarray(ti_)[:m].reshape(-1))
                    V.append(np.asarray(val, np.float64)[:m].reshape(-1)); G.append(np.repeat(d2g[idx[ok]][a:a + m], k_))
                W, T, V, G = map(np.concatenate, (W, T, V, G))
                inb = (W >= 0) & (W < nw) & (T >= 0) & (T < nt) & (V > 0)
                key = (G[inb].astype(np.int64) * nw + W[inb]) * nt + T[inb]
                uk, iv = np.unique(key, return_inverse=True)
                sv = np.bincount(iv, weights=V[inb])
                tr_of = np.searchsorted(tu, g2t[uk // (nw * nt)])
                np.maximum.at(smax, tr_of, sv)
            for j in range(len(tu)):
                rows.append([v, int(tu[j]), float(E[j]), float(Q[j]), float(smax[j])])
    return os.path.basename(f), rows, time.time() - t00


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    files = [os.path.join(TRUTH, f"ev{i:03d}.npz") for i in range(260, 388)]
    with get_context("spawn").Pool(a.workers, maxtasksperchild=8) as pool:
        R = pool.map(one, files)
    allr = np.array([r for _, rows, _ in R for r in rows]).reshape(-1, 5)
    E, Q, sm = allr[:, 2], allr[:, 3], allr[:, 4]
    amb = sm <= 25.0
    n = len(R)
    out = dict(n_events=n, n_hitless_particles=int(len(allr)),
               hitless_charge_per_event=float(Q.sum() / n), hitless_energy_per_event_MeV=float(E.sum() / n),
               ambiguous=dict(n_per_event=float(amb.sum() / n), charge_per_event=float(Q[amb].sum() / n),
                              energy_per_event_MeV=float(E[amb].sum() / n),
                              E_quantiles_MeV=[float(np.percentile(E[amb], q)) for q in (50, 90, 99, 100)] if amb.any() else None,
                              n_with_E_ge_0p02=int((E[amb] >= 0.02).sum()), n_with_E_ge_0p1=int((E[amb] >= 0.1).sum()),
                              Q_quantiles_e=[float(np.percentile(Q[amb], q)) for q in (50, 90, 99, 99.9, 100)] if amb.any() else None,
                              n_with_Q_gt_500=int((Q[amb] > 500).sum()), n_with_Q_gt_2043=int((Q[amb] > 2043).sum()),
                              Q_of_E_ge_0p1_quantiles_e=[float(np.percentile(Q[amb & (E >= 0.1)], q)) for q in (50, 90, 99, 100)]
                              if (amb & (E >= 0.1)).any() else None),
               out_of_window=dict(n_per_event=float((~amb).sum() / n), charge_per_event=float(Q[~amb].sum() / n)),
               note="ambiguous = largest per-group x wire x tick sample <= 25 e- if placed in window: could be "
                    "in-window and in neither truth; t0 placement 200 us")
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
