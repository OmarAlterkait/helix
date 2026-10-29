#!/usr/bin/env python3
"""Simulation truth for every shard of a coeff corpus run: ``{name}_cells_<k>.h5`` beside each
``{name}_coeff_<k>.h5`` (format and label: helix/data/cells.py).

    python scripts/build_truth.py --corpus <run dir>

The events are each shard's own ``/ident``; their ``hits`` and ``step`` are read from the
campaign at HELIX_SIM_ROOT (helix/tpc/io.py). A shard whose cells file exists is skipped, and a
file is renamed into place only once complete, so an interrupted build resumes.
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _one(job):
    from helix.data.cells import build_shard
    shard, source = job
    return os.path.basename(shard), build_shard(shard, source)[1]


def main():
    from helix.paths import root
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, help="one corpus run directory")
    a = ap.parse_args()
    source = str(root("HELIX_SIM_ROOT"))
    shards = sorted(glob.glob(os.path.join(a.corpus, "*_coeff_[0-9]*.h5")))
    if not shards:
        raise SystemExit(f"{a.corpus}: no *_coeff_<k>.h5 shards")
    with Pool(min(len(os.sched_getaffinity(0)), len(shards))) as pool:
        for name, msg in pool.imap_unordered(_one, [(s, source) for s in shards]):
            print(f"{name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
