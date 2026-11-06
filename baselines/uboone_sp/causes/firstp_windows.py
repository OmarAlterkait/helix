#!/usr/bin/env python3
"""First-principles B + C, stage 1: complete truth WITH particle identity on the floor evaluation's windows.

    scripts/helix_run.sh python baselines/uboone_sp/causes/firstp_windows.py \
        --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/firstp/windows --workers 32

For each scored test event (truth_v2 ev260-ev387) the event's deposits are re-simulated
with JAXTPC 1ea2282's own drift / attenuation / Gaussian-CDF diffusion exactly as
scripts/build_complete_corpus_truth.py does (same functions, imported from it), but every
diffusion sample keeps its deposit group and particle (track; id = volume * 1e7 + track, the
truth_v2 convention). Written per event (``evNNN.npz``):

  windows   every window of the floor evaluation: truth_v2 ``iso``/``pair``/``bg`` and the
            near sidecar's ``bgn`` (kind, g, w0, t0, energy bin of iso, hits charge inside);
  wt_*      per (window, particle) with complete charge inside: q_in, the particle's total
            complete charge on that plane, its deposited energy (step ``de``), its charge-
            weighted centroid on that plane (pixel units), and the max cell charge it puts there;
  wc_*      per window: complete charge total and max complete cell charge;
  cell_*    (part B) every fine cell with complete charge > 25 e-: complete q, the
            production-cut q (sum of per-group x wire x tick samples > 25 e-: the hits truth),
            the number of deposit groups with any sample in the cell, the largest single
            per-group sample, and the number of particles;
  check     sum over particles of the window-cell charge vs denoise_truth_complete (must agree).
"""
import argparse
import glob
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

TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"
SOURCE = "/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02"
COMPLETE = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
JAXTPC = "/global/cfs/cdirs/m5238/users/oalter/jaxtpc_1ea2282"
KINDS = ["iso", "pair", "bg", "bgn"]


def rebuild(step_ev, hits_ev, S, wcells):
    """Per plane gid: dict with window-cell per-(cell, particle) charge, particle plane totals and
    centroids, and per-cell part-B stats. ``wcells``: {gid: sorted fine-cell codes (fw * nct + ft)}."""
    import jax.numpy as jnp
    from build_complete_corpus_truth import CHUNK, DT_US, D_L, D_T, LIFETIME_US, PRE_WINDOW_US, V_CM_US, _local, _t0
    cfg, Kw, Kt, prep, cwd = S["cfg"], S["Kw"], S["Kt"], S["prep"], S["cwd"]
    nt = int(cfg.num_time_steps)
    out, E = {}, {}
    for vk in sorted(x for x in step_ev if x.startswith("volume_")):
        v = int(vk.split("_")[1])
        vg = step_ev[vk]
        if "positions" not in vg or vk not in hits_ev:
            continue
        org = np.array([vg.attrs["pos_origin_x"], vg.attrs["pos_origin_y"], vg.attrs["pos_origin_z"]])
        pos = vg["positions"][:].astype(np.float64) * float(vg.attrs["pos_step_mm"]) + org
        q = vg["charge"][:].astype(np.float64)
        de = vg["de"][:].astype(np.float64)
        pl = _local(cfg, v, pos)
        t0, d2g = _t0(cfg, hits_ev[vk], pl, q, v)
        g2t = hits_ev[vk]["group_to_track"][:].astype(np.int64)
        trk = v * 10_000_000 + g2t[d2g]                              # truth_v2 particle id
        tu, ti = np.unique(g2t[d2g], return_inverse=True)
        for t_, e_ in zip(tu.tolist(), np.bincount(ti, weights=de).tolist()):
            E[v * 10_000_000 + t_] = e_
        vgeo = cfg.volumes[v]
        for p in range(3):
            gid = 3 * v + p
            nw = int(vgeo.num_wires[p])
            dd = pl[:, 0] - vgeo.plane_distances_cm[p]
            tdr = dd / V_CM_US
            tick = tdr + t0 + PRE_WINDOW_US
            ok = np.isfinite(tick) & (tick >= 0) & (tick < nt * DT_US) & (q > 0)
            n = int(ok.sum())
            if n == 0:
                continue
            ci, cd = cwd(jnp.asarray(pl[ok, 1:3]), vgeo.angles_rad[p], vgeo.wire_spacings_cm[p],
                         vgeo.max_wire_indices[p], vgeo.index_offsets[p])
            ci, cd = np.asarray(ci), np.asarray(cd)
            qq, tt, dr, dt_ = q[ok], tick[ok], dd[ok], tdr[ok]
            att = np.exp(-dt_ / LIFETIME_US)
            k_ = (2 * Kw + 1) * (2 * Kt + 1)
            W, T, V, D = [], [], [], []
            idx_ok = np.nonzero(ok)[0]
            for a in range(0, n, CHUNK):
                m = min(n, a + CHUNK) - a
                pad = lambda x, dt=np.float32: np.pad(np.asarray(x[a:a + m], dt), (0, CHUNK - m))
                one = np.ones(CHUNK, np.float32)
                wi, ti_, val = prep(pad(qq), pad(dt_), pad(tt), pad(dr), pad(ci, np.int32), pad(cd), pad(att),
                                    one * 0.5, one * 0.5, one, np.arange(CHUNK) < m,
                                    Kw, Kt, float(vgeo.wire_spacings_cm[p]), DT_US, D_L, D_T, V_CM_US, nw, nt)
                W.append(np.asarray(wi)[:m].reshape(-1)); T.append(np.asarray(ti_)[:m].reshape(-1))
                V.append(np.asarray(val, np.float64)[:m].reshape(-1)); D.append(np.repeat(idx_ok[a:a + m], k_))
            W, T, V, D = map(np.concatenate, (W, T, V, D))
            inb = (W >= 0) & (W < nw) & (T >= 0) & (T < nt) & (V > 0)
            W, T, V, D = W[inb], T[inb], V[inb], D[inb]
            nct = -(-nt // 16)
            cell = (W // 2).astype(np.int64) * nct + T // 16
            tr = trk[D]; grp = d2g[D]
            # particle plane totals and centroids (pixel units)
            pu, pi = np.unique(tr, return_inverse=True)
            ptot = np.bincount(pi, weights=V)
            pcw = np.bincount(pi, weights=V * W) / np.maximum(ptot, 1e-30)
            pct = np.bincount(pi, weights=V * T) / np.maximum(ptot, 1e-30)
            # window cells: per (cell, particle)
            wc = wcells.get(gid, np.zeros(0, np.int64))
            m = np.isin(cell, wc)
            ck = cell[m] * (len(pu) + 1) + pi[m]
            uk, iv = np.unique(ck, return_inverse=True)
            wq = np.bincount(iv, weights=V[m])
            # part B: per-cell stats over all cells with complete charge > 25 e-
            cq = np.bincount(cell, weights=V, minlength=-(-nw // 2) * nct)
            big = np.flatnonzero(cq > 25.0)
            gp = (grp.astype(np.int64) * nw + W) * nt + T                # one per-group x wire x tick sample
            us, isv = np.unique(gp, return_inverse=True)
            sv = np.bincount(isv, weights=V)
            scell = ((us // nt) % nw // 2) * nct + (us % nt) // 16
            sg = us // (nw * nt)
            keep = np.isin(scell, big)
            scell, sv, sg = scell[keep], sv[keep], sg[keep]
            o = np.argsort(scell, kind="stable"); scell, sv, sg = scell[o], sv[o], sg[o]
            pos_ = np.searchsorted(scell, big); end_ = np.searchsorted(scell, big, side="right")
            qthr = np.zeros(len(big)); smax = np.zeros(len(big)); ngr = np.zeros(len(big), np.int64)
            for j, (a_, b_) in enumerate(zip(pos_, end_)):
                s = sv[a_:b_]
                qthr[j] = s[s > 25.0].sum(); smax[j] = s.max() if b_ > a_ else 0
                ngr[j] = len(np.unique(sg[a_:b_]))
            ctk = np.unique(cell[np.isin(cell, big)] * (len(pu) + 1) + pi[np.isin(cell, big)]) // (len(pu) + 1)
            npart = np.bincount(np.searchsorted(big, ctk), minlength=len(big))
            out[gid] = dict(nct=nct, wcell=uk // (len(pu) + 1), wtrk=pu[uk % (len(pu) + 1)], wq=wq,
                            pu=pu, ptot=ptot, pcw=pcw, pct=pct,
                            bcell=big, bq=cq[big], bqthr=qthr, bng=ngr, bsmax=smax, bnp=npart)
    return out, E


def one_event(job):
    f, out_dir = job
    import h5py
    import hdf5plugin  # noqa: F401
    from build_complete_corpus_truth import _init
    from helix.probe.resolution import FT, FW, unkey, window_cells
    name = os.path.basename(f)
    out = os.path.join(out_dir, name)
    if os.path.exists(out):
        return name, "exists"
    t00 = time.time()
    S = _init(JAXTPC)
    z = np.load(f, allow_pickle=True)
    zn = np.load(os.path.join(NEAR, name), allow_pickle=True)
    shard, event = str(z["shard"]), int(z["event"])
    run = os.path.basename(os.path.dirname(shard))
    tag = os.path.basename(shard).replace("sim_wire_coeff_", "").replace(".h5", "")
    meta = json.loads(str(z["meta"])) + json.loads(str(zn["meta"]))
    wq_h = list(np.split(z["wq"], np.searchsorted(z["wid"], np.arange(1, len(json.loads(str(z["meta"]))))))) \
        if len(z["wq"]) else []
    nm_ = len(json.loads(str(zn["meta"])))
    wq_h += list(np.split(zn["wq"], np.searchsorted(zn["wid"], np.arange(1, nm_)))) if nm_ else []
    wins = [(m_["kind"], int(m_["g"]), int(m_["w0"]), int(m_["t0"])) for m_ in meta]
    keys = [window_cells(g, w0, t0) for _, g, w0, t0 in wins]
    wcells = {}
    for k, (_, g0, _, _) in zip(keys, wins):
        g, fw, ft = unkey(k)
        ok = (fw >= 0) & (ft >= 0) & (g == g0)                 # off-grid cells (an origin left of 0) hold nothing
        wcells.setdefault(int(g0), []).append((fw[ok], ft[ok]))
    for tries in range(3):
        try:
            with h5py.File(os.path.join(SOURCE, "step", run, f"sim_wire_step_{tag}.h5"), "r") as fs, \
                    h5py.File(os.path.join(SOURCE, "hits", run, f"sim_wire_hits_{tag}.h5"), "r") as fh:
                ev = f"event_{event:03d}"
                nct = -(-int(S["cfg"].num_time_steps) // 16)
                wc = {g: np.unique(np.concatenate([fw[ft < nct].astype(np.int64) * nct + ft[ft < nct] for fw, ft in v])) for g, v in wcells.items()}
                R, E = rebuild(fs[ev], fh[ev], S, wc)
            break
        except (OSError, KeyError, RuntimeError):
            if tries == 2:
                raise
            time.sleep(30)
    # per window tables
    WT, WC = [], []
    for wi, ((kind, g, w0, t0), k) in enumerate(zip(wins, keys)):
        r = R.get(g)
        if r is None:
            WC.append([0.0, 0.0]); continue
        gg, fw, ft = unkey(k)
        okc = (fw >= 0) & (ft >= 0) & (ft < r["nct"]) & (gg == g)
        code = fw[okc].astype(np.int64) * r["nct"] + ft[okc]
        m = np.isin(r["wcell"], code)
        cells, trks, q = r["wcell"][m], r["wtrk"][m], r["wq"][m]
        cq = np.bincount(np.searchsorted(np.unique(cells), cells), weights=q) if len(cells) else np.zeros(0)
        WC.append([float(q.sum()), float(cq.max()) if len(cq) else 0.0])
        for t_ in np.unique(trks).tolist():
            mt = trks == t_
            j = int(np.searchsorted(r["pu"], t_))
            cqt = np.bincount(np.searchsorted(np.unique(cells[mt]), cells[mt]), weights=q[mt])
            WT.append([wi, t_, float(q[mt].sum()), float(r["ptot"][j]), float(E.get(t_, 0.0)),
                       float(r["pcw"][j] - w0), float(r["pct"][j] - t0), float(cqt.max())])
    # check against the corpus complete truth
    zc = np.load(os.path.join(COMPLETE, run, f"truth_{tag}.npz"))
    i = int(np.nonzero(zc["event"] == event)[0][0])
    CK, CQ = zc["keys"][zc["offsets"][i]:zc["offsets"][i + 1]], zc["q"][zc["offsets"][i]:zc["offsets"][i + 1]]
    allk = np.unique(np.concatenate(keys))
    j = np.clip(np.searchsorted(CK, allk), 0, max(len(CK) - 1, 0))
    ref = np.where(CK[j] == allk, CQ[j], 0.0).sum() if len(CK) else 0.0
    mine = 0.0
    for g, r in R.items():
        mine += r["wq"].sum()
    # part B cells
    from helix.probe.resolution import fkey
    B = {k_: [] for k_ in ("key", "q", "qthr", "ng", "smax", "np")}
    for g, r in R.items():
        B["key"].append(fkey(np.full(len(r["bcell"]), g), r["bcell"] // r["nct"], r["bcell"] % r["nct"]))
        B["q"].append(r["bq"]); B["qthr"].append(r["bqthr"]); B["ng"].append(r["bng"]); B["smax"].append(r["bsmax"]); B["np"].append(r["bnp"])
    np.savez(out + ".tmp.npz",
             win_kind=np.array([KINDS.index(k) for k, *_ in wins]), win_g=np.array([w[1] for w in wins]),
             win_w0=np.array([w[2] for w in wins]), win_t0=np.array([w[3] for w in wins]),
             win_E=np.array([(m_.get("E") or [np.nan])[0] for m_ in meta], float),
             win_qhits=np.array([float(x.sum()) for x in wq_h]) if wq_h else np.zeros(0),
             wc=np.asarray(WC, float).reshape(-1, 2), wt=np.asarray(WT, float).reshape(-1, 8),
             check=np.array([mine, ref]), shard=shard, event=event,
             **{f"cell_{k_}": np.concatenate(v) for k_, v in B.items()})
    os.replace(out + ".tmp.npz", out)
    return name, f"{len(wins)} windows, check {mine:.1f}/{ref:.1f}, {time.time() - t00:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=260)
    ap.add_argument("--last", type=int, default=387)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = [os.path.join(TRUTH, f"ev{i:03d}.npz") for i in range(a.first, a.last + 1)]
    jobs = [(f, a.out) for f in files]
    with get_context("spawn").Pool(a.workers, maxtasksperchild=8) as pool:
        for nm, msg in pool.imap_unordered(one_event, jobs):
            print(nm, msg, flush=True)


if __name__ == "__main__":
    main()
