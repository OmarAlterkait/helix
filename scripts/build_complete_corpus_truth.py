#!/usr/bin/env python3
"""COMPLETE pre-response charge per fine cell, for every event of a coeff corpus run
(the denoiser's training truth without the production's thresholds).

    python scripts/build_complete_corpus_truth.py --corpus <run dir> --out <dir> --jaxtpc <JAXTPC 1ea2282 tree> \
        [--source S] [--workers 32] [--validate <existing truth dir>] [--shards a,b,..] [--part i/n]

Same output format as scripts/build_denoise_truth.py -- ``<out>/truth_<shard>.npz`` per corpus
shard with ``event`` (n,) (corpus event ids in shard order), ``offsets`` (n+1,), ``keys`` int64
(helix.probe.resolution.fkey(plane_gid, wire // 2, tick // 16), sorted per event) and ``q``
float32 electrons -- plus two diagnostics per event: ``q_total`` (all deposit charge of the
event, recombined, before attenuation) and ``q_unplaced`` (the part of it left out, below).

Definition. build_denoise_truth.py reads the simulation's ``hits`` modality, which the
production (JAXTPC commit 1ea2282, tools/track_hits.py + production/save.py) writes after two
cuts: per deposit-group x wire x tick samples <= 1 e- are dropped in the box compaction and
<= 25 e- at the CSR encode. Those cuts drop the diffusion tails of every group, about 1.6% of the
in-window charge, growing with drift (baselines/uboone_sp/results/exact_kernel.md,
baselines/uboone_sp/causes/results/causes_FBC.md, bright_budget.md). This script rebuilds the
same quantity WITHOUT the cuts, from every deposit of the ``step`` file, with JAXTPC 1ea2282's own
functions (imported from --jaxtpc, never modified):
  * drift to each plane: local x = drift_dir * (x_anode - x), minus the plane's distance from the
    anode (drift.correct_drift_for_plane semantics), t_drift = distance / 0.16 cm/us;
  * attenuation exp(-t_drift / 10 ms); readout tick = t_drift + t0 + 405 us (physics.py:177-180);
  * wire index and sub-wire offset: wires.compute_wire_distances;
  * Gaussian-CDF diffusion with sigma(t_drift) over (2K_w + 1) x (2K_t + 1) integer wires x ticks:
    wires.prepare_deposit_with_diffusion -- the very function the production's hits use;
  * summed per (plane, wire // 2, tick // 16); no threshold at any stage.
Deposit times are not stored in the step files. They are recovered per deposit group from the
hits: the hits of a group are its charge, Gaussian-diffused about tick_us = t_drift + t0 + 405 us,
so t0 = 0.5 us x (charge-weighted mean hits bin + 0.5) - charge-weighted mean (t_drift + 405 us)
of the group's deposits, pooled over the three planes. A group without any hits takes the
charge-weighted t0 of its track's groups; deposits of tracks with no hits at all have no time and
are left out (``q_unplaced``): they lie outside the readout window (re-simulating the stored
sensor without them reproduces its total to 0.9998; exact_kernel.md section 2).

Validation (--validate <dir of the existing truth files>): the same rebuild WITH the production's
per-group > 25 e- cut must reproduce the existing truth; the per-shard report gives the median /
95th-percentile relative L2 over (event, plane) and the complete / hits charge ratio. Written to
``<out>/validate_<shard>.json``.

Resumable and atomic: existing outputs are skipped; each file is written to a temporary name and
renamed. JAX runs on CPU (set JAX_PLATFORMS=cpu); chunks are padded to a fixed size so each worker
compiles once. No pimm_data import.
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNK = 16384
PRE_WINDOW_US, V_CM_US, LIFETIME_US, DT_US = 405.0, 0.16, 10000.0, 0.5
D_L, D_T, MAX_DRIFT_CM = 7.2e-6, 12.0e-6, 216.0
_STATE = {}


def _init(jaxtpc):
    """Per worker: JAXTPC functions, the detector config and the jitted diffusion."""
    if _STATE:
        return _STATE
    sys.path.insert(0, jaxtpc)
    import jax
    import yaml
    from tools.config import create_sim_config
    from tools.geometry import calculate_max_diffusion_sigmas
    from tools.wires import compute_wire_distances, prepare_deposit_with_diffusion
    cfg = create_sim_config(yaml.safe_load(open(os.path.join(jaxtpc, "config", "cubic_wireplane_config.yaml"))),
                            total_pad=1000, include_digitize=True, num_s=16)
    _, _, sig_t, sig_l = calculate_max_diffusion_sigmas(MAX_DRIFT_CM, V_CM_US, D_T, D_L, 0.3, DT_US)
    Kw, Kt = max(1, int(np.ceil(3.0 * sig_t))), max(1, int(np.ceil(3.0 * sig_l)))
    vm = jax.vmap(prepare_deposit_with_diffusion, in_axes=(0,) * 11 + (None,) * 9)
    _STATE.update(jax=jax, cfg=cfg, Kw=Kw, Kt=Kt, cwd=compute_wire_distances,
                  prep=jax.jit(vm, static_argnums=tuple(range(11, 20))))
    return _STATE


def _local(cfg, v, pos_mm):
    vg = cfg.volumes[v]
    p = pos_mm.astype(np.float64).copy()
    p[:, 0] = vg.drift_direction * (vg.x_anode_cm * 10.0 - pos_mm[:, 0])
    p[:, 1] -= vg.yz_center_cm[0] * 10.0
    p[:, 2] -= vg.yz_center_cm[1] * 10.0
    return p / 10.0                                                  # cm, local frame


def _t0(cfg, hv, pl_cm, q, v):
    """Per-deposit t0 (us): group t0 from its hits, else its track's, else NaN."""
    from helix.probe.truth import PLANES, decode_hits_plane
    d2g = hv["deposit_to_group"][:].astype(np.int64)
    g2t = hv["group_to_track"][:].astype(np.int64)
    ng = len(g2t)
    num, den = np.zeros(ng), np.zeros(ng)
    qd = np.bincount(d2g, weights=q, minlength=ng)
    for p, P in enumerate(PLANES):
        if P not in hv:
            continue
        h = decode_hits_plane(hv[P])
        qh = np.bincount(h["group"], weights=h["q"], minlength=ng)[:ng]
        th = np.bincount(h["group"], weights=h["q"] * (h["tick"] + 0.5) * DT_US, minlength=ng)[:ng]
        tk = (pl_cm[:, 0] - cfg.volumes[v].plane_distances_cm[p]) / V_CM_US + PRE_WINDOW_US
        td = np.bincount(d2g, weights=q * tk, minlength=ng)
        ok = (qh > 0) & (qd > 0)
        num[ok] += qh[ok] * (th[ok] / qh[ok] - td[ok] / qd[ok])
        den[ok] += qh[ok]
    okg = den > 0
    t0g = np.where(okg, num / np.maximum(den, 1e-30), np.nan)
    tu, ti = np.unique(g2t, return_inverse=True)
    tn = np.bincount(ti, weights=np.where(okg, qd * np.nan_to_num(t0g), 0.0), minlength=len(tu))
    tdn = np.bincount(ti, weights=np.where(okg, qd, 0.0), minlength=len(tu))
    t0trk = np.where(tdn > 0, tn / np.maximum(tdn, 1e-30), np.nan)
    return np.where(okg, t0g, t0trk[ti])[d2g], d2g


def event_complete(step_ev, hits_ev, S, want_thr=False):
    """-> ((keys, q[, keys_thr, q_thr]), q_total, q_unplaced) for one event (all volumes, planes)."""
    import jax.numpy as jnp
    from helix.probe.resolution import fkey
    cfg, Kw, Kt, prep, cwd = S["cfg"], S["Kw"], S["Kt"], S["prep"], S["cwd"]
    nt = int(cfg.num_time_steps)
    K, Q, KT, QT = [], [], [], []
    q_total = q_unpl = 0.0
    for vk in sorted(x for x in step_ev if x.startswith("volume_")):
        v = int(vk.split("_")[1])
        vg = step_ev[vk]
        if "positions" not in vg or vk not in hits_ev:
            continue
        org = np.array([vg.attrs["pos_origin_x"], vg.attrs["pos_origin_y"], vg.attrs["pos_origin_z"]])
        pos = vg["positions"][:].astype(np.float64) * float(vg.attrs["pos_step_mm"]) + org
        q = vg["charge"][:].astype(np.float64)
        pl = _local(cfg, v, pos)
        t0, d2g = _t0(cfg, hits_ev[vk], pl, q, v)
        q_total += float(q.sum()); q_unpl += float(q[~np.isfinite(t0)].sum())
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
            W, T, V, G = [], [], [], []
            for a in range(0, n, CHUNK):
                m = min(n, a + CHUNK) - a
                pad = lambda x, dt=np.float32: np.pad(np.asarray(x[a:a + m], dt), (0, CHUNK - m))
                one = np.ones(CHUNK, np.float32)
                wi, ti, val = prep(pad(qq), pad(dt_), pad(tt), pad(dr), pad(ci, np.int32), pad(cd), pad(att),
                                   one * 0.5, one * 0.5, one, np.arange(CHUNK) < m,
                                   Kw, Kt, float(vgeo.wire_spacings_cm[p]), DT_US, D_L, D_T, V_CM_US, nw, nt)
                k_ = (2 * Kw + 1) * (2 * Kt + 1)
                W.append(np.asarray(wi)[:m].reshape(-1)); T.append(np.asarray(ti)[:m].reshape(-1))
                V.append(np.asarray(val, np.float64)[:m].reshape(-1))
                if want_thr:
                    G.append(np.repeat(d2g[ok][a:a + m], k_))
            W, T, V = np.concatenate(W), np.concatenate(T), np.concatenate(V)
            inb = (W >= 0) & (W < nw) & (T >= 0) & (T < nt) & (V > 0)
            ncw, nct = -(-nw // 2), -(-nt // 16)
            cq = np.bincount((W[inb] // 2) * nct + T[inb] // 16, weights=V[inb], minlength=ncw * nct)
            nz = np.flatnonzero(cq)                                   # dense per-plane cell sums: no sort
            K.append(fkey(np.full(len(nz), gid), nz // nct, nz % nct)); Q.append(cq[nz])
            if want_thr:
                G = np.concatenate(G)[inb]
                pix = W[inb].astype(np.int64) * nt + T[inb]
                key = G.astype(np.int64) * (nw * nt) + pix
                uk, iv = np.unique(key, return_inverse=True)
                s = np.bincount(iv, weights=V[inb])
                kp = s > 25.0                                          # production corr_threshold
                px = uk[kp] % (nw * nt)
                KT.append(fkey(np.full(int(kp.sum()), gid), (px // nt) // 2, (px % nt) // 16)); QT.append(s[kp])

    def red(Ks, Qs):
        if not Ks:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        uk, inv = np.unique(np.concatenate(Ks), return_inverse=True)
        return uk.astype(np.int64), np.bincount(inv, weights=np.concatenate(Qs)).astype(np.float32)

    # complete truth: planes are visited in gid order and each plane's keys are sorted and unique,
    # so the concatenation is already the sorted, unique key list
    if K:
        k, qv = np.concatenate(K).astype(np.int64), np.concatenate(Q).astype(np.float32)
        assert np.all(np.diff(k) > 0)
    else:
        k, qv = np.zeros(0, np.int64), np.zeros(0, np.float32)
    out = (k, qv) + (red(KT, QT) if want_thr else ())
    return out, q_total, q_unpl


def _compare(kt, qt, kh, qh):
    """Relative L2 of thresholded-rebuild vs existing truth, per plane gid."""
    from helix.probe.resolution import unkey
    res = {}
    uk = np.union1d(kt, kh)
    a = np.zeros(len(uk)); b = np.zeros(len(uk))
    a[np.searchsorted(uk, kt)] = qt; b[np.searchsorted(uk, kh)] = qh
    g = unkey(uk)[0]
    for gid in np.unique(g):
        m = g == gid
        res[int(gid)] = float(np.linalg.norm(a[m] - b[m]) / max(np.linalg.norm(b[m]), 1e-9))
    return res, float(a.sum()), float(b.sum())


def one_shard(job, tries=3):
    """Build one shard; a transient filesystem error (CFS EIO) is retried, then reported, never fatal to the pool."""
    for k in range(tries):
        try:
            return _one_shard(job)
        except (OSError, KeyError, RuntimeError) as e:  # h5py reports CFS EIO as any of these
            err = f"{type(e).__name__}: {e}"
            time.sleep(30 * (k + 1))
    return job[3], os.path.basename(job[0]), f"FAILED after {tries} tries ({err}); rerun to resume"


def _one_shard(job):
    shard, out_dir, source, run, jaxtpc, validate = job
    import h5py
    import hdf5plugin  # noqa: F401
    tag = os.path.basename(shard).replace("sim_wire_coeff_", "").replace(".h5", "")
    out = os.path.join(out_dir, f"truth_{tag}.npz")
    vout = os.path.join(out_dir, f"validate_{tag}.json")
    if os.path.exists(out) and (not validate or os.path.exists(vout)):
        return run, tag, "exists"
    t00 = time.time()
    S = _init(jaxtpc)
    with h5py.File(shard, "r") as f:
        events = f["ident"]["event"][:].astype(np.int64)
        srcs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["ident"]["source_file"][:]]
    ref = np.load(os.path.join(validate, f"truth_{tag}.npz")) if validate else None
    keys, qs, offsets, qtot, qunp = [], [], [0], [], []
    rels, s_thr, s_hit, s_full = [], 0.0, 0.0, 0.0
    files = {}
    for i, (ev, sf) in enumerate(zip(events.tolist(), srcs)):
        if sf not in files:
            for fh in files.values():
                fh[0].close(); fh[1].close()
            files = {sf: (h5py.File(os.path.join(source, "step", run, sf.replace("_sensor_", "_step_")), "r"),
                          h5py.File(os.path.join(source, "hits", run, sf.replace("_sensor_", "_hits_")), "r"))}
        fs, fh = files[sf]
        r, qt, qu = event_complete(fs[f"event_{ev:03d}"], fh[f"event_{ev:03d}"], S, want_thr=ref is not None)
        k, q = r[0], r[1]
        keys.append(k); qs.append(q); offsets.append(offsets[-1] + len(k)); qtot.append(qt); qunp.append(qu)
        if ref is not None:
            assert int(ref["event"][i]) == ev, (tag, i, ev)
            a, b = int(ref["offsets"][i]), int(ref["offsets"][i + 1])
            pr, st, sh = _compare(r[2], r[3], ref["keys"][a:b], ref["q"][a:b])
            rels += list(pr.values()); s_thr += st; s_hit += sh; s_full += float(q.sum())
    for fh in files.values():
        fh[0].close(); fh[1].close()
    if not os.path.exists(out):
        tmp = out + ".tmp.npz"
        np.savez(tmp, event=events, offsets=np.asarray(offsets, np.int64),
                 keys=np.concatenate(keys) if keys else np.zeros(0, np.int64),
                 q=np.concatenate(qs) if qs else np.zeros(0, np.float32),
                 q_total=np.asarray(qtot), q_unplaced=np.asarray(qunp))
        os.replace(tmp, out)
    msg = f"{len(events)} events, {offsets[-1]:,} cells, unplaced {sum(qunp) / max(sum(qtot), 1e-9):.3f}, {time.time() - t00:.0f}s"
    if ref is not None:
        rep = dict(shard=tag, run=run, n_events=len(events), n_event_planes=len(rels),
                   thr_vs_hits_rel_l2_median=float(np.median(rels)), thr_vs_hits_rel_l2_p95=float(np.percentile(rels, 95)),
                   thr_over_hits=s_thr / s_hit, complete_over_hits=s_full / s_hit,
                   unplaced_frac=sum(qunp) / max(sum(qtot), 1e-9), seconds=time.time() - t00)
        json.dump(rep, open(vout + ".tmp", "w"), indent=1); os.replace(vout + ".tmp", vout)
        msg += f" | thr/hits L2 med {rep['thr_vs_hits_rel_l2_median']:.4f} p95 {rep['thr_vs_hits_rel_l2_p95']:.4f}, " \
               f"complete/hits {rep['complete_over_hits']:.4f}"
    return run, tag, msg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, help="one corpus run directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jaxtpc", required=True, help="JAXTPC tree at commit 1ea2282 (read-only)")
    ap.add_argument("--source", default=None, help="dir holding step/ and hits/; default: parent of HELIX_SENSOR_ROOT")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--validate", default=None, help="existing truth dir of this run: write validate_<shard>.json")
    ap.add_argument("--shards", default="", help="comma list of shard tags to process (default: all)")
    ap.add_argument("--part", default="0/1", help="i/n: process every n-th shard starting at i (spread over nodes)")
    a = ap.parse_args()
    if a.source is None:
        from helix.paths import root
        source = str(root("HELIX_SENSOR_ROOT").parent)
    else:
        source = a.source
    run = os.path.basename(os.path.normpath(a.corpus))
    os.makedirs(a.out, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(a.corpus, "sim_wire_coeff_[0-9]*.h5")))
    if a.shards:
        want = set(a.shards.split(","))
        shards = [s for s in shards if os.path.basename(s).replace("sim_wire_coeff_", "").replace(".h5", "") in want]
    i, n = (int(x) for x in a.part.split("/"))
    shards = shards[i::n]
    jobs = [(s, a.out, source, run, a.jaxtpc, a.validate) for s in shards]
    with get_context("spawn").Pool(a.workers, maxtasksperchild=8) as pool:
        for r, tag, msg in pool.imap_unordered(one_shard, jobs):
            print(f"{r} {tag}: {msg}", flush=True)


if __name__ == "__main__":
    main()
