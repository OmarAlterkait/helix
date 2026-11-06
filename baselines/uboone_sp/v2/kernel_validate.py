#!/usr/bin/env python3
"""Forward-fold the per-wire hits truth with each response kernel and compare with
the stored clean sensor (held-out test events; zero-padded linear convolution,
then the sensor writer's storage: integer ADC, |v| < 2 dropped).

    scripts/helix_run.sh python baselines/uboone_sp/v2/kernel_validate.py --events 260-263 \
        --out baselines/uboone_sp/v2/results/kernel_validate.json

Kernels: fit (least squares on 8 probe-train events), example (avg_response),
exact (JAXTPC-derived, drift-averaged), exact0 (JAXTPC kernel averaged over
sub-wire / sub-tick position, no diffusion) and 'family': each hit sample folded
with exact_<P>_s<k> for its own diffusion level -- s from the charge-weighted
drift of the deposits in its group (step positions, plane-corrected), which needs
the step file and is more truth than any per-wire method has (diagnostic only).
Metrics per plane: ||C - store(H (*) R)|| / ||C||, correlation, sum ratio.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, HERE)
SRC = "/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02"
RESP = os.path.join(BASE, "response")


def kern(name):
    d = np.load(os.path.join(RESP, name) if name.endswith(".npz") else name)
    return d["R_avg"].astype(np.float64), d["wire_offsets"].astype(np.int64), int(d["time_zero_bin"])


def fwd(H, R, wo, tzb, pad=400):
    from sp.response import forward
    Hp = np.pad(H, ((0, 0), (pad, pad)))
    return forward(Hp, R, wo, tzb)[:, pad:-pad]


def one(job):
    ev, g = job
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import h5py
    import hdf5plugin  # noqa: F401
    from sp.data import Event, registry, PLANES
    from sp.response import load_kernel, store_like_sensor
    from helix.probe.truth import decode_hits_plane
    import resim as RS
    reg = registry()
    E = Event(ev)
    pl = PLANES[g % 3]
    C = E.clean(g, reg).astype(np.float64)
    H = E.truth(g, C.shape).astype(np.float64)
    nC = np.linalg.norm(C)
    out = {}
    ks = {"fit": load_kernel(pl, "fit"), "example": load_kernel(pl, "example"),
          "exact": kern(f"exact_{pl}.npz"), "exact0": kern(f"exact0_{pl}.npz")}
    for k, (R, wo, tzb) in ks.items():
        S = store_like_sensor(fwd(H, R, wo, tzb))
        out[k] = dict(resid=float(np.linalg.norm(C - S) / nC), corr=float(np.corrcoef(C.ravel(), S.ravel())[0, 1]),
                      sum_ratio=float(S.sum() / C.sum()) if abs(C.sum()) > 0 else None)
    # drift-dependent family (oracle drift per hit group)
    v, p = divmod(g, 3)
    sys.path.insert(0, RS.JAXTPC)
    import yaml
    from tools.config import create_sim_config
    cfg = create_sim_config(yaml.safe_load(open(os.path.join(RS.JAXTPC, "config", "cubic_wireplane_config.yaml"))),
                            total_pad=1000, include_digitize=True, num_s=16)
    step = os.path.join(SRC, "step", E.run, E.source_file.replace("_sensor_", "_step_"))
    hitsf = os.path.join(SRC, "hits", E.run, E.source_file.replace("_sensor_", "_hits_"))
    with h5py.File(step, "r") as fs, h5py.File(hitsf, "r") as fh:
        vg = fs[f"event_{E.event:03d}"][f"volume_{v}"]
        hv = fh[f"event_{E.event:03d}"][f"volume_{v}"]
        org = np.array([vg.attrs["pos_origin_x"], vg.attrs["pos_origin_y"], vg.attrs["pos_origin_z"]])
        pos = vg["positions"][:].astype(np.float64) * float(vg.attrs["pos_step_mm"]) + org
        q = vg["charge"][:].astype(np.float64)
        d2g = hv["deposit_to_group"][:].astype(np.int64)
        h = decode_hits_plane(hv[pl])
    _, dd = RS.plane_ticks(cfg, pos, v, p)
    ng = int(d2g.max()) + 1
    qd = np.bincount(d2g, weights=q, minlength=ng)
    dg = np.bincount(d2g, weights=q * np.maximum(dd, 0), minlength=ng) / np.maximum(qd, 1e-30)
    s_h = np.sqrt(np.clip(dg[h["group"]] / 216.0, 0, 1))
    grid = np.linspace(0, 1, 9)
    kidx = np.clip(np.round(s_h * 8).astype(int), 0, 8)
    Sfam = np.zeros_like(C)
    ok = (h["wire"] >= 0) & (h["wire"] < C.shape[0]) & (h["tick"] >= 0) & (h["tick"] < C.shape[1])
    for k in range(9):
        m = ok & (kidx == k)
        if not m.any():
            continue
        Hk = np.zeros_like(C)
        np.add.at(Hk, (h["wire"][m], h["tick"][m]), h["q"][m])
        R, wo, tzb = kern(f"exact_{pl}_s{k}.npz")
        Sfam += fwd(Hk, R, wo, tzb)
    Sf = store_like_sensor(Sfam)
    out["family"] = dict(resid=float(np.linalg.norm(C - Sf) / nC), corr=float(np.corrcoef(C.ravel(), Sf.ravel())[0, 1]),
                         sum_ratio=float(Sf.sum() / C.sum()) if abs(C.sum()) > 0 else None)
    out["hits_check"] = float(np.abs(np.bincount(np.ravel_multi_index((h["wire"][ok], h["tick"][ok]), C.shape),
                                                 weights=h["q"][ok], minlength=C.size).reshape(C.shape) - H).max())
    return f"ev{ev}_g{g}", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-263")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.events.split("-"))
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        res = dict(pool.map(one, [(e, g) for e in range(lo, hi + 1) for g in range(6)]))
    summ = {}
    for P, gs in (("U", (0, 3)), ("V", (1, 4)), ("Y", (2, 5))):
        rows = [r for k, r in res.items() if int(k.split("_g")[1]) in gs]
        summ[P] = {k: dict(resid=float(np.mean([r[k]["resid"] for r in rows])), corr=float(np.mean([r[k]["corr"] for r in rows])))
                   for k in ("fit", "example", "exact", "exact0", "family")}
    json.dump(dict(per_plane=res, summary=summ), open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
