#!/usr/bin/env python3
"""Near-activity noise windows for the floor evaluation, as a sidecar.

    python scripts/dump_near_windows.py --truth <truth_v2> --out <truth_v2_near> [--n 60]

For every truth file ``ev*.npz``, charge-free windows that have charge within 16
wires / 128 ticks (:func:`helix.probe.resolution.near_windows`), kind ``bgn``,
written as ``<out>/ev*.npz`` with the truth file's own layout (``wkey``, ``wq``,
``wid``, ``meta``). A sidecar, not a rebuild: the existing windows -- and every
number scored on them -- stay byte-identical; scripts/eval_denoise.py --near adds
these. Seeded per event, so the set is reproducible.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02")
    a = ap.parse_args()

    import h5py
    from helix.core.coeff_io import read_coeff_event
    from helix.probe import resolution as R
    from scripts.build_denoise_truth import event_cells

    os.makedirs(a.out, exist_ok=True)
    counts = []
    for f in sorted(glob.glob(os.path.join(a.truth, "ev*.npz"))):
        ev = int(os.path.basename(f)[2:5])
        z = np.load(f, allow_pickle=True)
        shard, event = str(z["shard"]), int(z["event"])
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        ce = read_coeff_event(shard, pos)
        run = os.path.basename(os.path.dirname(shard))
        tk, tq = event_cells(os.path.join(a.source, "hits", run, ce.source_file.replace("_sensor_", "_hits_")), ce.event)
        wins = R.near_windows(tk, tq, np.random.default_rng((a.seed, ev)), n=a.n)
        wk = [R.window_cells(g, w0, t0) for g, w0, t0 in wins]
        qmap = dict(zip(tk.tolist(), tq.tolist()))
        wq = [np.array([qmap.get(x, 0.0) for x in k.tolist()]) for k in wk]
        if any(q.any() for q in wq):
            raise SystemExit(f"{f}: a near window holds charge")
        meta = [dict(kind="bgn", g=g, w0=w0, t0=t0, E=[], cen=[]) for g, w0, t0 in wins]
        np.savez_compressed(os.path.join(a.out, os.path.basename(f)),
                            wkey=np.concatenate(wk) if wk else np.zeros(0, np.int64),
                            wq=np.concatenate(wq) if wq else np.zeros(0),
                            wid=np.concatenate([np.full(len(k), i) for i, k in enumerate(wk)]) if wk else np.zeros(0, int),
                            meta=json.dumps(meta))
        counts.append(len(wins))
    print(f"[near] {len(counts)} events, {sum(counts)} windows (min {min(counts)}, median {int(np.median(counts))}) -> {a.out}")


if __name__ == "__main__":
    main()
