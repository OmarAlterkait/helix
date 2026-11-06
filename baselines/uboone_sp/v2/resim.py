#!/usr/bin/env python3
"""Re-simulate the stored clean sensor of a few events from the STEP files with
JAXTPC's own functions at the production commit (1ea2282): drift, plane
correction, wire geometry, deposit preparation, diffusion-table kernel
interpolation, dense accumulation, digitisation and the sparse-save threshold.

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/v2/resim.py --events 260-263 \
        --out baselines/uboone_sp/v2/results/resim.json

Per (event, plane gid) it reports ||stored - resim|| / ||stored|| for:
  resim        positions as stored (uint16 grid, 0.3 mm step), deposit charge
               from the step file, t0 recovered per group from the hits file
               (deposits whose group left no hits are dropped: their time is unknown),
               attenuation exp(-t_drift / 10 ms) applied (and, as a check, not applied)
  t0           the step files carry no deposit time; t0 is recovered per deposit
               group from the hits file (recover_t0); 'resim_t0zero' shows t0 = 0
  jitter       the same with each position moved by an independent U(-0.15, 0.15) mm
               in x, y, z -- the size of the step file's position quantisation; the
               resim-vs-jitter residual is the floor this test can reach
and the pixel-level agreement (fraction of stored nonzero samples reproduced exactly).
Event lookup as sp.data.Event (truth file -> shard -> source file + event id).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
JAXTPC = "/pscratch/sd/o/oalter/helix_work/uboone_sp/jaxtpc_1ea2282"
SRC = "/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02"


def setup():
    sys.path.insert(0, JAXTPC)
    import jax
    import yaml
    from tools.config import create_sim_config
    from tools.geometry import calculate_max_diffusion_sigmas
    from tools.kernels import load_response_kernels
    cfgd = yaml.safe_load(open(os.path.join(JAXTPC, "config", "cubic_wireplane_config.yaml")))
    cfg = create_sim_config(cfgd, total_pad=200_000, include_digitize=True, num_s=16)
    _, _, sig_t, sig_l = calculate_max_diffusion_sigmas(216.0, 0.16, 12.0e-6, 7.2e-6, 0.3, 0.5)
    RK = load_response_kernels(response_path=os.path.join(JAXTPC, "tools", "responses"), num_s=16, time_spacing=0.5,
                               max_sigma_trans_unitless=sig_t, max_sigma_long_unitless=sig_l)
    return jax, cfg, RK


def plane_ticks(cfg, pos_global_mm, v, p):
    """Readout tick (us, t0 = 0) and plane drift distance per deposit, as compute_plane_physics."""
    vg = cfg.volumes[v]
    pos = pos_global_mm.astype(np.float64).copy()
    pos[:, 0] = vg.drift_direction * (vg.x_anode_cm * 10.0 - pos[:, 0])
    d = pos[:, 0] / 10.0 - vg.plane_distances_cm[p]
    return d / 0.16 + 405.0, d


def recover_t0(cfg, hits_vol, pos, q, v):
    """Deposit times are not in the step files. Recover one t0 per deposit GROUP from
    the hits file: the hits truth of a group is its charge, Gaussian-diffused about
    tick_us = t_drift + t0 + 405 us (wires.py:318-345), so t0 = dt * (charge-weighted
    mean hits bin + 0.5) - charge-weighted mean (t_drift + 405) over the group's deposits,
    pooled over the three planes."""
    from helix.probe.truth import decode_hits_plane
    d2g = hits_vol["deposit_to_group"][:].astype(np.int64)
    ng = int(d2g.max()) + 1 if len(d2g) else 0
    num, den = np.zeros(ng), np.zeros(ng)
    for p, P in enumerate("UVY"):
        if P not in hits_vol:
            continue
        h = decode_hits_plane(hits_vol[P])
        qh = np.bincount(h["group"], weights=h["q"], minlength=ng)[:ng]
        th = np.bincount(h["group"], weights=h["q"] * (h["tick"] + 0.5) * 0.5, minlength=ng)[:ng]
        tk, _ = plane_ticks(cfg, pos, v, p)
        qd = np.bincount(d2g, weights=q, minlength=ng)[:ng]
        td = np.bincount(d2g, weights=q * tk, minlength=ng)[:ng]
        ok = (qh > 0) & (qd > 0)
        num[ok] += qh[ok] * (th[ok] / qh[ok] - td[ok] / qd[ok])
        den[ok] += qh[ok]
    t0g = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)   # no hits: time unknown (outside the window)
    return t0g[d2g], t0g, den > 0


def resim_plane(jax, cfg, RK, pos_global_mm, charge, v, p, atten=True, chunk=4000, t0_us=None):
    import jax.numpy as jnp
    from tools.drift import correct_drift_for_plane
    from tools.wires import compute_wire_distances, prepare_deposit_for_response, accumulate_response_signals
    from tools.kernels import interpolate_diffusion_kernel_batch
    from tools.electronics import _digitize_signal
    vg = cfg.volumes[v]
    pos = pos_global_mm.astype(np.float64).copy()                       # global -> local (loader.py:739-748)
    pos[:, 0] = vg.drift_direction * (vg.x_anode_cm * 10.0 - pos[:, 0])
    pos[:, 1] -= vg.yz_center_cm[0] * 10.0
    pos[:, 2] -= vg.yz_center_cm[1] * 10.0
    pcm = pos / 10.0
    vel = 0.16
    d_anode, t_anode = pcm[:, 0], pcm[:, 0] / vel                      # compute_drift_to_plane, local frame, no SCE
    dd, dt = correct_drift_for_plane(jnp.asarray(d_anode), jnp.asarray(t_anode), vel, vg.plane_distances_cm[p])
    dd, dt = np.asarray(dd), np.asarray(dt)
    att = np.exp(-np.where(np.isnan(dt), 0, dt) / 10000.0) if atten else np.ones(len(dt))
    pre = 405.0
    tick = dt + (0.0 if t0_us is None else t0_us) + pre
    nt = cfg.num_time_steps
    q = np.where(np.isfinite(tick), charge, 0.0) * ((np.nan_to_num(tick, nan=-1.0) >= 0) & (np.nan_to_num(tick, nan=-1.0) < nt * 0.5))
    tick = np.nan_to_num(tick, nan=0.0)
    ci, cd = compute_wire_distances(jnp.asarray(pcm[:, 1:3]), vg.angles_rad[p], vg.wire_spacings_cm[p],
                                    vg.max_wire_indices[p], vg.index_offsets[p])
    prep = jax.vmap(prepare_deposit_for_response, in_axes=(0, 0, 0, 0, 0, None, None, None, None))
    wi, wo, ti, to, inten = prep(jnp.asarray(q, jnp.float32), jnp.asarray(tick, jnp.float32), ci, cd,
                                 jnp.asarray(att, jnp.float32), True, vg.wire_spacings_cm[p], 0.5, vg.num_wires[p])
    pl = cfg.plane_names[v][p]
    K = RK[pl]
    s = np.clip(np.sqrt(np.maximum(dd, 0) / 216.0), 0, 1)
    nw = vg.num_wires[p]
    sig = jnp.zeros((nw, nt))
    for a in range(0, len(q), chunk):
        sl = slice(a, a + chunk)
        con = interpolate_diffusion_kernel_batch(K.DKernel, jnp.asarray(s[sl], jnp.float32), wo[sl], to[sl],
                                                 K.wire_spacing, K.num_wires)
        sig = sig + accumulate_response_signals(wi[sl], ti[sl], inten[sl], con, nw, nt, K.num_wires, K.kernel_height,
                                                K.wire_zero_bin, K.time_zero_bin)
    ped = 410.0 if pl == "Y" else 1843.0
    d = np.array(_digitize_signal(sig, 1.0, ped, 4095.0))
    d[np.abs(d) < 2.0] = 0.0                                           # save.py:236  |v| >= threshold_adc kept
    return d, np.asarray(sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-263")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401
    from sp.data import Event, registry
    jax, cfg, RK = setup()
    reg = registry()
    lo, hi = (int(x) for x in a.events.split("-"))
    res = {}
    rng = np.random.default_rng(0)
    for ev in range(lo, hi + 1):
        E = Event(ev)
        step = os.path.join(SRC, "step", E.run, E.source_file.replace("_sensor_", "_step_"))
        hitsf = os.path.join(SRC, "hits", E.run, E.source_file.replace("_sensor_", "_hits_"))
        fhh = h5py.File(hitsf, "r")
        with h5py.File(step, "r") as fh:
            g = fh[f"event_{E.event:03d}"]
            vols = {}
            for v in (0, 1):
                vg = g[f"volume_{v}"]
                if "positions" not in vg:
                    continue
                org = np.array([vg.attrs["pos_origin_x"], vg.attrs["pos_origin_y"], vg.attrs["pos_origin_z"]])
                pos_ = vg["positions"][:].astype(np.float64) * float(vg.attrs["pos_step_mm"]) + org
                q_ = vg["charge"][:].astype(np.float64)
                t0d, t0g, okg = recover_t0(cfg, fhh[f"event_{E.event:03d}"][f"volume_{v}"], pos_, q_, v)
                vols[v] = (pos_, q_, t0d)
                res[f"ev{ev}_v{v}_t0"] = dict(n_groups=int(len(t0g)), n_with_hits=int(okg.sum()),
                                              t0_us_p5_50_95=[float(x) for x in np.percentile(t0g[okg], [5, 50, 95])])
        for gid in range(6):
            v, p = divmod(gid, 3)
            C = E.clean(gid, reg)
            if v not in vols:
                continue
            pos, q, t0d = vols[v]
            r = {}
            for lab, pp, att, t0 in (("resim", pos, True, t0d), ("resim_noatten", pos, False, t0d),
                                     ("resim_t0zero", pos, True, None),
                                     ("jitter", pos + rng.uniform(-0.15, 0.15, pos.shape), True, t0d)):
                d, raw = resim_plane(jax, cfg, RK, pp, q, v, p, atten=att, t0_us=t0)
                d = d[:C.shape[0], :C.shape[1]]
                r[lab] = d
            out = {}
            nC = np.linalg.norm(C)
            for lab in r:
                out[f"{lab}_resid"] = float(np.linalg.norm(C - r[lab]) / nC)
                nz = C != 0
                out[f"{lab}_exact_frac"] = float(np.mean(r[lab][nz] == C[nz]))
                out[f"{lab}_sum_ratio"] = float(r[lab].sum() / C.sum()) if C.sum() else None
            out["resim_vs_jitter"] = float(np.linalg.norm(r["resim"] - r["jitter"]) / nC)
            res[f"ev{ev}_g{gid}"] = out
            print(ev, gid, {k: round(x, 4) for k, x in out.items() if x is not None}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
