#!/usr/bin/env python3
"""Per-particle fine-cell truth for the test events: which particle carries the
charge of every fine cell (2 wires x 16 ticks), from the simulation's hits
(group -> track) and step (deposited energy per track), exactly as
scripts/dump_resolution_truth.py identifies particles (id = volume * 1e7 + track).

    scripts/helix_run.sh python baselines/uboone_sp/dense/particle_truth.py --events 260-387 \
        --out /pscratch/sd/o/oalter/helix_work/uboone_sp/dense/ptruth

Per event one npz; per plane gid g (readout pixels only, like sp.data.Event.truth):
  c_g      flat fine-cell index fw * n_cols + ft of every charged cell (sorted)
  q_g      total truth charge (e-)            qmax_g  charge of the majority particle
  tmax_g   majority particle id               n10_g   number of particles with >= 10% of the cell
  pc_g, pt_g, pq_g   every (cell, particle, charge) triplet
  trk_id, trk_E      particle ids and their deposited energy (MeV, from step `de`)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "v2"))
FW, FT = 2, 16


def process(job):
    ev, out_dir = job
    out = os.path.join(out_dir, f"ev{ev:03d}.npz")
    if os.path.exists(out):
        return ev, "exists"
    import h5py
    import hdf5plugin  # noqa: F401
    from helix.probe.truth import decode_hits_plane
    from sp.data import Event, registry, PLANES

    reg = registry()
    E = Event(ev)
    step = E.hits.replace("/hits/", "/step/").replace("_hits_", "_step_")
    key = f"event_{E.event:03d}"
    res, Eid = {}, {}
    with h5py.File(step, "r") as fs, h5py.File(E.hits, "r") as fh:
        for vk in sorted(x for x in fh[key] if x.startswith("volume_")):
            vi = int(vk.split("_")[1])
            de = fs[key][vk]["de"][:].astype(np.float64)
            d2g = fh[key][vk]["deposit_to_group"][:].astype(np.int64)
            g2t = fh[key][vk]["group_to_track"][:].astype(np.int64)
            tu, ti = np.unique(g2t[d2g], return_inverse=True)
            for t_, e_ in zip(tu.tolist(), np.bincount(ti, weights=de).tolist()):
                Eid[vi * 10_000_000 + t_] = e_
            for pi, pl in enumerate(PLANES):
                g = vi * 3 + pi
                if pl not in fh[key][vk]:
                    continue
                s = decode_hits_plane(fh[key][vk][pl])
                nw, nt = reg[g]["n_wires"], reg[g]["n_ticks"]
                ncol = -(-nt // FT)
                w, t, q = s["wire"], s["tick"], s["q"]
                trk = vi * 10_000_000 + g2t[s["group"]]
                ok = (w >= 0) & (w < nw) & (t >= 0) & (t < nt)
                w, t, q, trk = w[ok], t[ok], q[ok], trk[ok]
                c = (w // FW) * ncol + t // FT
                tid, tinv = np.unique(trk, return_inverse=True)
                pk = c.astype(np.int64) * len(tid) + tinv
                up, pinv = np.unique(pk, return_inverse=True)
                pq = np.bincount(pinv, weights=q)
                pc, pt = up // len(tid), tid[up % len(tid)]
                # per cell: total, majority particle (pairs sorted by cell, then charge descending)
                o = np.lexsort((-pq, pc))
                pc, pt, pq = pc[o], pt[o], pq[o]
                uc, first, cnt = np.unique(pc, return_index=True, return_counts=True)
                qt = np.add.reduceat(pq, first)
                share = pq / np.repeat(qt, cnt)
                n10 = np.add.reduceat((share >= 0.10).astype(np.int64), first)
                res.update({f"c_{g}": uc.astype(np.int64), f"q_{g}": qt.astype(np.float32),
                            f"qmax_{g}": pq[first].astype(np.float32), f"tmax_{g}": pt[first],
                            f"n10_{g}": n10.astype(np.int16), f"pc_{g}": pc.astype(np.int64),
                            f"pt_{g}": pt, f"pq_{g}": pq.astype(np.float32)})
    ids = np.array(sorted(Eid), np.int64)
    np.savez(out + ".tmp.npz", trk_id=ids, trk_E=np.array([Eid[i] for i in ids], np.float64), **res)
    os.replace(out + ".tmp.npz", out)
    return ev, f"{len(ids)} particles"


def main():
    from classical import parse_events
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for ev, msg in pool.imap_unordered(process, [(e, a.out) for e in parse_events(a.events)]):
            print(f"ev{ev:03d}: {msg}", flush=True)


if __name__ == "__main__":
    main()
