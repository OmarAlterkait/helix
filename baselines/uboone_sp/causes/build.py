#!/usr/bin/env python3
"""Per-cell evidence for hypotheses F (dense haze = truth artefact?), B (sparse faint
false charge) and C (gating zeroes true charge), one npz per test event.

    JAX_PLATFORMS=cpu scripts/helix_run.sh python baselines/uboone_sp/causes/build.py --events 260-291 \
        --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/causes/cells

Per plane gid, on the 2 wire x 16 tick fine cells (readout only):
  Th      the evaluation's truth: fine_cells(hits) (sp.data.Event.truth)
  Tf      COMPLETE truth rebuilt with JAXTPC 1ea2282's own hits code
          (wires.prepare_deposit_with_diffusion: attenuated charge, Gaussian-CDF
          diffusion with sigma(t_drift), integer wires x ticks) from EVERY deposit of
          the step file, with NO thresholds (the production drops per group x wire x
          tick samples <= 25 e-, and box entries <= 1 e-). Deposit times are not in the
          step files: t0 per deposit group from the hits (v2/resim.recover_t0); groups
          without hits take the charge-weighted t0 of their track's groups; deposits of
          tracks without any hits have no time and are counted, not placed.
  Tthr    the same rebuild WITH the production's thresholds (> 1 e-, then > 25 e- per
          group x wire x tick): must reproduce Th (machinery check)
  dq      hits-truth charge in the +-8-cell box, cell excluded (dense/ definition); dq_f same on Tf
  dist_h / dist_f   chessboard distance (cells) to the nearest cell with Th > 0 / Tf >= 25 e-
  model   M6d / M6dc raw charge and presence (dense3 maps, CFS), coverage
  cls     classical tuned ROI charge and tuned best statistic (dense cls_fit / cls_fix)
  Kmax    max over the corpus's KEPT coefficients mapped to the cell (bands A4..D2, as the
          model reads them; tick_of_tau) of |value| / sigma (the event's per gid x band
          sigma, sigma_threshold); Kband, Kn: its band and the number of kept coefficients
  Smax    the same on the noise-free signal: DWT of the stored clean sensor (what the
          corpus threshold acts on, minus noise), |coefficient| / sigma
  Kmax3 / Smax3   3 x 3-cell maxima (the response and the band footprints spread over cells)
Records: every cell with Th > 0, Tf > 0, a model or classical prediction > 50 e-, presence
> 0.05 or a kept coefficient; the rest sampled at 1% (weight 100).
Also per plane: 16 x 8-cell tiles (32 x 128 regions) of Th, Tf and the charge maps, and the
event's unplaceable deposit charge.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "v2")); sys.path.insert(0, os.path.dirname(os.path.dirname(BASE)))
SRC = "/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02"
MODEL3 = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/dense3/model"
CLS = ["/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/cls_fit", "/global/cfs/cdirs/m5238/users/oalter/helix_runs/dense2/cls_fix"]
PS_CORPUS, CFS_CORPUS = "/pscratch/sd/o/oalter/coeff_tpc_r1", "/global/cfs/cdirs/m5238/users/oalter/coeff_tpc_r1"
COLS = ["plane", "w", "Th", "Tf", "Tthr", "dq", "dq_f", "dist_h", "dist_f", "cov", "M6d_q", "M6d_p", "M6dc_q", "M6dc_p",
        "cls_q", "cls_any", "Kmax", "Kband", "Kn", "Smax", "Kmax3", "Smax3"]


OST61 = set()          # /pscratch files on the hung OST 61, probed on the HOST with `lfs getstripe -i` (no lfs in the
                       # container): --ost61 file with one path per line; those are read from the CFS copies instead


def ost_ok(path):
    return path not in OST61


def fine(X, FW=2, FT=16):
    nw, nt = X.shape
    Y = np.pad(X, ((0, (-nw) % FW), (0, (-nt) % FT)))
    return Y.reshape(Y.shape[0] // FW, FW, Y.shape[1] // FT, FT).sum((1, 3))


def box(X, r=8):
    P = np.pad(np.asarray(X, np.float64), ((r + 1, r), (r + 1, r)))
    S = P.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]


def max3(X):
    from scipy.ndimage import maximum_filter
    return maximum_filter(X, size=3, mode="constant")


def dist_to(mask):
    from scipy.ndimage import distance_transform_cdt
    return distance_transform_cdt(~mask, metric="chessboard").astype(np.float32) if mask.any() else np.full(mask.shape, 999, np.float32)


def rebuild(ev, E, cfg, nshape):
    """Complete and thresholded truth pixel maps per gid from the step + hits files."""
    import h5py
    import hdf5plugin  # noqa: F401
    import jax
    import jax.numpy as jnp
    import resim as RS
    from tools.wires import compute_wire_distances, prepare_deposit_with_diffusion
    from tools.geometry import calculate_max_diffusion_sigmas
    _, _, sig_t, sig_l = calculate_max_diffusion_sigmas(216.0, 0.16, 12.0e-6, 7.2e-6, 0.3, 0.5)
    Kw, Kt = max(1, int(np.ceil(3.0 * sig_t))), max(1, int(np.ceil(3.0 * sig_l)))
    prep = jax.vmap(prepare_deposit_with_diffusion, in_axes=(0,) * 11 + (None,) * 9)
    fs = h5py.File(os.path.join(SRC, "step", E.run, E.source_file.replace("_sensor_", "_step_")), "r")
    fh = h5py.File(os.path.join(SRC, "hits", E.run, E.source_file.replace("_sensor_", "_hits_")), "r")
    full, thr, info = {}, {}, {}
    for v in (0, 1):
        vg, hv = fs[f"event_{E.event:03d}"][f"volume_{v}"], fh[f"event_{E.event:03d}"][f"volume_{v}"]
        if "positions" not in vg:
            for p in range(3):
                full[3 * v + p] = np.zeros(nshape[3 * v + p]); thr[3 * v + p] = np.zeros(nshape[3 * v + p])
            continue
        org = np.array([vg.attrs["pos_origin_x"], vg.attrs["pos_origin_y"], vg.attrs["pos_origin_z"]])
        pos = vg["positions"][:].astype(np.float64) * float(vg.attrs["pos_step_mm"]) + org
        q = vg["charge"][:].astype(np.float64)
        d2g = hv["deposit_to_group"][:].astype(np.int64)
        g2t = hv["group_to_track"][:].astype(np.int64)
        t0d, t0g, okg = RS.recover_t0(cfg, hv, pos, q, v)
        # groups without hits: their track's charge-weighted t0
        qg = np.bincount(d2g, weights=q, minlength=len(t0g))
        trk = g2t[: len(t0g)]
        tu, ti = np.unique(trk, return_inverse=True)
        num = np.bincount(ti, weights=np.where(okg, qg * np.nan_to_num(t0g), 0.0), minlength=len(tu))
        den = np.bincount(ti, weights=np.where(okg, qg, 0.0), minlength=len(tu))
        t0trk = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
        t0g2 = np.where(okg, t0g, t0trk[ti])
        t0 = t0g2[d2g]
        info[f"v{v}_q_total"] = float(q.sum())
        info[f"v{v}_q_group_hits"] = float(q[okg[d2g]].sum())
        info[f"v{v}_q_track_t0"] = float(q[~okg[d2g] & np.isfinite(t0)].sum())
        info[f"v{v}_q_unplaced"] = float(q[~np.isfinite(t0)].sum())
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
                m = min(n, a + 20000) - a
                one = jnp.ones(m, jnp.float32)
                wi, ti_, val = prep(jnp.asarray(q[ok][sl], jnp.float32), jnp.asarray(dt_[sl], jnp.float32),
                                    jnp.asarray(tick[ok][sl], jnp.float32), jnp.asarray(dd[ok][sl], jnp.float32),
                                    ci[sl], cd[sl], jnp.asarray(att[sl], jnp.float32), one * 0.5, one * 0.5, one,
                                    jnp.ones(m, bool), Kw, Kt, vgeo.wire_spacings_cm[p], 0.5, 7.2e-6, 12.0e-6, 0.16, nw, nt)
                W.append(np.asarray(wi)); Tt.append(np.asarray(ti_)); V.append(np.asarray(val, np.float64))
            K = (2 * Kw + 1) * (2 * Kt + 1)
            W, Tt, V = (np.concatenate(x).reshape(-1) for x in (W, Tt, V))
            G = np.repeat(d2g[ok], K)
            inb = (W >= 0) & (W < nw) & (Tt >= 0) & (Tt < nt) & (V > 0)
            W, Tt, V, G = W[inb], Tt[inb], V[inb], G[inb]
            F = np.zeros(nw * nt); np.add.at(F, W * nt + Tt, V)
            full[gid] = F.reshape(nw, nt)
            key = G.astype(np.int64) * (nw * nt) + W.astype(np.int64) * nt + Tt
            uk, inv = np.unique(key, return_inverse=True)
            s = np.bincount(inv, weights=V)
            keep = s > 25.0                                   # corr_threshold (inter_thresh 1 e- is implied)
            pix = uk[keep] % (nw * nt)
            Hh = np.zeros(nw * nt); np.add.at(Hh, pix, s[keep])
            thr[gid] = Hh.reshape(nw, nt)
    return full, thr, info


def coeff_cells(ce, sig, gid, shape, toff_cfg):
    """Max |kept| / sigma per fine cell (bands 0-3), its band, and the count."""
    from helix.model.tokenize import tick_of_tau
    m = (ce.plane_gid == gid) & (ce.band < 4)
    b = ce.band[m].astype(np.int64)
    a = np.abs(ce.value[m]) / sig[gid, b]
    tk = tick_of_tau(ce.tau[m], ce.plane_gid[m], b, toff_cfg)
    fw, ft = ce.wire[m].astype(np.int64) // 2, np.floor(tk / 16).astype(np.int64)
    okk = (fw >= 0) & (fw < shape[0]) & (ft >= 0) & (ft < shape[1])
    fw, ft, a, b = fw[okk], ft[okk], a[okk], b[okk]
    flat = fw * shape[1] + ft
    K = np.zeros(shape[0] * shape[1]); np.maximum.at(K, flat, a)
    Kb = np.full(shape[0] * shape[1], -1.0)
    o = np.argsort(a)                                          # last write wins = largest amplitude
    Kb[flat[o]] = b[o]
    Kn = np.bincount(flat, minlength=shape[0] * shape[1])
    return K.reshape(shape), Kb.reshape(shape), Kn.reshape(shape).astype(np.float32)


def signal_cells(C, sig, gid, shape, toff_cfg):
    """Max |DWT(clean sensor)| / sigma per fine cell over bands A4..D2 (corpus layout)."""
    from helix.core import backend
    backend.set_backend("numpy")
    from helix.core.wavelet import wavedec
    from helix.tpc.config import DetectorConfig
    from helix.model.tokenize import tick_of_tau
    dc = DetectorConfig()
    nt = C.shape[1]
    x = np.pad(np.asarray(C, np.float32), ((0, 0), (0, (-nt) % (1 << dc.dwt_level))))
    co, _ = wavedec(x, wavelet=dc.wavelet, level=dc.dwt_level, mode=dc.dwt_mode)
    S = np.zeros(shape)
    for b in range(4):
        arr = np.abs(np.asarray(co[b])) / sig[gid, b]
        tk = tick_of_tau(np.arange(arr.shape[1]), np.full(arr.shape[1], gid), np.full(arr.shape[1], b), toff_cfg)
        ft = np.floor(tk / 16).astype(np.int64)
        okt = (ft >= 0) & (ft < shape[1])
        fw = np.arange(arr.shape[0]) // 2
        okw = fw < shape[0]
        A = arr[np.ix_(okw, okt)]
        flat = (fw[okw][:, None] * shape[1] + ft[okt][None, :]).ravel()
        Sf = S.ravel(); np.maximum.at(Sf, flat, A.ravel()); S = Sf.reshape(shape)
    return S


def per_event(job):
    ev, out_dir = job
    if os.environ.get("CAUSES_OST61"):
        OST61.update(x.strip() for x in open(os.environ["CAUSES_OST61"]) if x.strip())
    out = os.path.join(out_dir, f"ev{ev:03d}.npz")
    if os.path.exists(out):
        return ev, "exists"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    t00 = time.time()
    import h5py
    import yaml
    import resim as RS
    sys.path.insert(0, RS.JAXTPC)
    from tools.config import create_sim_config
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents
    from sp.data import Event, registry
    cfg = create_sim_config(yaml.safe_load(open(os.path.join(RS.JAXTPC, "config", "cubic_wireplane_config.yaml"))),
                            total_pad=1000, include_digitize=True, num_s=16)
    tcfg = DenoiseEvents("", [], "", items=[]).cfg
    reg = registry()
    E = Event(ev)
    shard = E.shard.replace(PS_CORPUS, CFS_CORPUS)
    with h5py.File(shard, "r") as fh:
        pos = int(np.searchsorted(fh["ident"]["event"][:], E.event))
    ce = read_coeff_event(shard, pos)
    sig = np.asarray(ce.sigma_threshold, np.float64)
    M3 = np.load(os.path.join(MODEL3, f"ev{ev:03d}.npz"))
    to = int(M3["toff"])
    nshape = {g: (reg[g]["n_wires"], reg[g]["n_ticks"]) for g in range(6)}
    Tfull, Tthr, info = rebuild(ev, E, cfg, nshape)
    rng = np.random.default_rng(5000 + ev)
    recs, tiles, chk = [], [], {}
    for g in range(6):
        p = g % 3
        C = E.clean(g, reg)
        Tp = E.truth(g, C.shape).astype(np.float64)
        Th, Tf, Tt = fine(Tp), fine(Tfull[g]), fine(Tthr[g])
        shape = Th.shape
        crop = lambda X: np.asarray(X[:shape[0], to:to + shape[1]], np.float64)
        f_cls = next(x for x in (os.path.join(d, f"ev{ev:03d}_g{g}.npz") for d in CLS) if os.path.exists(x) and ost_ok(x))
        Fc = np.load(f_cls)
        assert np.allclose(Fc["truth"], Th, rtol=1e-4, atol=1e-2), (ev, g, "classical truth differs")
        K, Kb, Kn = coeff_cells(ce, sig, g, shape, tcfg)
        S = signal_cells(C, sig, g, shape, tcfg)
        has_h = Th > 0
        dq = box(Th); dq = np.where(has_h, dq - Th, dq); dq[dq < 1] = 0
        dqf = box(Tf); dqf = np.where(Tf > 0, dqf - Tf, dqf); dqf[dqf < 1] = 0
        maps = dict(plane=np.full(shape, p, np.float64), Th=Th, Tf=Tf, Tthr=Tt, dq=dq, dq_f=dqf,
                    dist_h=dist_to(has_h), dist_f=dist_to(Tf >= 25.0), cov=crop(M3[f"cov_g{g}"]).astype(np.float64),
                    M6d_q=crop(M3[f"q_M6d_g{g}"]), M6d_p=crop(M3[f"p_M6d_g{g}"]),
                    M6dc_q=crop(M3[f"q_M6dc_g{g}"]), M6dc_p=crop(M3[f"p_M6dc_g{g}"]),
                    cls_q=np.asarray(Fc["tuned_q"], np.float64), cls_any=np.asarray(Fc["tuned_any"], np.float64),
                    Kmax=K, Kband=Kb, Kn=Kn, Smax=S, Kmax3=max3(K), Smax3=max3(S))
        sel = (Th > 0) | (Tf > 0) | (maps["M6d_q"] > 50) | (maps["M6dc_q"] > 50) | (maps["cls_q"] > 50) \
            | (maps["M6d_p"] > 0.05) | (maps["M6dc_p"] > 0.05) | (K > 0)
        samp = ~sel & (rng.random(shape) < 0.01)
        w = np.where(sel, 1.0, np.where(samp, 100.0, 0.0))
        keep = (sel | samp).ravel()
        maps["w"] = w
        recs.append(np.stack([maps[c].ravel()[keep] for c in COLS], 1).astype(np.float32))
        # 32 x 128 regions (16 x 8 cells)
        nw16, nt8 = (shape[0] // 16) * 16, (shape[1] // 8) * 8
        tl = lambda X: X[:nw16, :nt8].reshape(nw16 // 16, 16, nt8 // 8, 8).sum((1, 3)).ravel()
        Mg = lambda t: np.where(maps[f"{t}_p"] > 0.5, maps[f"{t}_q"], 0.0)
        tiles.append(np.stack([np.full(len(tl(Th)), p), tl(Th), tl(Tf), tl(Mg("M6d")), tl(Mg("M6dc")), tl(maps["cls_q"])], 1))
        chk[f"g{g}"] = dict(Th=float(Th.sum()), Tf=float(Tf.sum()), Tthr=float(Tt.sum()),
                            thr_vs_hits_rel_l2=float(np.linalg.norm(Tt - Th) / max(np.linalg.norm(Th), 1e-9)),
                            thr_vs_hits_pix_rel_l2=float(np.linalg.norm(Tthr[g] - Tp) / max(np.linalg.norm(Tp), 1e-9)),
                            cls_file=f_cls)
    np.savez_compressed(out + ".tmp.npz", rec=np.concatenate(recs), tiles=np.concatenate(tiles).astype(np.float32),
                        info=json.dumps(dict(info, **chk)))
    os.replace(out + ".tmp.npz", out)
    return ev, f"{time.time() - t00:.0f}s " + json.dumps({k: round(v["thr_vs_hits_rel_l2"], 4) for k, v in chk.items()})


def main():
    from classical import parse_events
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-291")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ost61", default="", help="file listing /pscratch paths on OST 61 (probed on the host)")
    a = ap.parse_args()
    if a.ost61:
        os.environ["CAUSES_OST61"] = a.ost61
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(cols=COLS), open(os.path.join(a.out, "columns.json"), "w"))
    evs = parse_events(a.events)
    if a.workers <= 1:
        for e in evs:
            print(*per_event((e, a.out)), flush=True)
        return
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.workers, maxtasksperchild=2) as pool:
        for ev, msg in pool.imap_unordered(per_event, [(e, a.out) for e in evs]):
            print(f"ev{ev:03d}: {msg}", flush=True)


if __name__ == "__main__":
    main()
