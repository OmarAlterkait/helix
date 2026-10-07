#!/usr/bin/env python3
"""Pre-response charge per fine cell, for every event of a coeff corpus run.

    python scripts/build_denoise_truth.py --corpus <run dir> --out <dir> [--source S] [--workers 32]

The denoising model (helix.model.denoise) is trained to predict, for each fine
cell ``(plane_gid, wire // FW, tick // FT)`` of :mod:`helix.probe.resolution`, the
pre-response charge the simulation deposited there: the ``hits`` modality summed
over particle groups. That is the same truth, on the same grid, the floor
evaluation scores (scripts/dump_resolution_truth.py), decoded the same way.

Writes ``<out>/truth_<shard>.npz`` per corpus shard: ``event`` (n,) -- the
corpus event ids in shard order, ``offsets`` (n+1,), ``keys`` int64 and ``q``
float32 -- the event's cells with charge, sorted by key. Existing files are
skipped, so an interrupted build resumes. Hits are read from
``<source>/hits/<run>/sim_wire_hits_<tag>.h5``, joined on the corpus identity
(source_file, event); ``--source`` defaults to the parent of HELIX_SENSOR_ROOT.
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def event_cells(hits_file, event):
    """(keys, q) of one event's charge-carrying fine cells, sorted by key."""
    import h5py
    import hdf5plugin  # noqa: F401  (hits shards use a plugin filter)
    from helix.probe.resolution import FT, FW, fkey
    from helix.probe.truth import PLANES, decode_hits_plane

    K, Q = [], []
    with h5py.File(hits_file, "r") as fh:
        ev = fh[f"event_{int(event):03d}"]
        for vk in sorted(x for x in ev if x.startswith("volume_")):
            vi = int(vk.split("_")[1])
            for pi, pl in enumerate(PLANES):
                if pl not in ev[vk]:
                    continue
                s = decode_hits_plane(ev[vk][pl])
                K.append(fkey(np.full(len(s["q"]), vi * 3 + pi), s["wire"] // FW, s["tick"] // FT))
                Q.append(np.asarray(s["q"], np.float64))
    if not K:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    uk, inv = np.unique(np.concatenate(K), return_inverse=True)
    return uk.astype(np.int64), np.bincount(inv, weights=np.concatenate(Q)).astype(np.float32)


def one_shard(job):
    shard, out_dir, source, run = job
    import h5py
    tag = os.path.basename(shard).replace("sim_wire_coeff_", "").replace(".h5", "")
    out = os.path.join(out_dir, f"truth_{tag}.npz")
    if os.path.exists(out):
        return tag, "exists"
    with h5py.File(shard, "r") as f:
        events = f["ident"]["event"][:].astype(np.int64)
        srcs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["ident"]["source_file"][:]]
    keys, qs, offsets = [], [], [0]
    for ev, sf in zip(events.tolist(), srcs):
        hits = os.path.join(source, "hits", run, sf.replace("_sensor_", "_hits_"))
        k, q = event_cells(hits, ev)
        keys.append(k); qs.append(q); offsets.append(offsets[-1] + len(k))
    tmp = out + ".tmp.npz"
    np.savez(tmp, event=events, offsets=np.asarray(offsets, np.int64),
             keys=np.concatenate(keys) if keys else np.zeros(0, np.int64),
             q=np.concatenate(qs) if qs else np.zeros(0, np.float32))
    os.replace(tmp, out)                                   # complete or absent, never partial
    return tag, f"{len(events)} events, {offsets[-1]:,} cells"


def main():
    from helix.paths import root
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, help="one corpus run directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default=None, help="dir holding hits/; default: parent of HELIX_SENSOR_ROOT")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    source = a.source or str(root("HELIX_SENSOR_ROOT").parent)
    run = os.path.basename(os.path.normpath(a.corpus))
    os.makedirs(a.out, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(a.corpus, "sim_wire_coeff_[0-9]*.h5")))
    with Pool(a.workers) as pool:
        for tag, msg in pool.imap_unordered(one_shard, [(s, a.out, source, run) for s in shards]):
            print(f"{run} {tag}: {msg}", flush=True)


if __name__ == "__main__":
    main()
