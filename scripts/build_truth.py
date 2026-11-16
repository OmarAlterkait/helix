#!/usr/bin/env python3
"""Simulation truth for every shard of a coeff corpus run, on a cell grid: one
``{name}_cells_<k>.h5`` per ``{name}_coeff_<k>.h5`` (format and label: helix/data/cells.py).

    python scripts/build_truth.py --corpus <run dir> --grid 2x16                 beside each shard
    python scripts/build_truth.py --corpus <run dir> --grid 1x4 --out <dir>      elsewhere
        [--shards 0-99] [--workers N]
    python scripts/build_truth.py --from <dir of a finer truth> --grid 1x4 --out <dir>

``--from`` needs no simulation: it sums an existing finer truth (``{name}_cells_<k>.h5`` or the
1 x 1 ``{name}_pixels_<k>.h5``) onto ``--grid`` (helix.data.cells.derive_shard), e.g. the 1 x 1
finetruth's 2,000 d1m events to 1 x 4.

The grid (wires x raw ticks per cell) is written into each file's /config and read back with
every record; 2x16 is the d1m truth everything has been scored on, and a finer grid that nests in
it sums back to it exactly (helix.data.cells.aggregate). Beside the shards there is room for one
truth only: a different grid goes to ``--out``, e.g. ``<root>/cells`` next to ``<root>/coeff`` ->
the corpus run, which the dataset reads as its two modalities.

The events are each shard's own ``/ident``; their ``hits``, ``step`` and ``labl`` are read from
the campaign at HELIX_SIM_ROOT (helix/tpc/io.py). A shard whose cells file exists on the same
grid is skipped (on another grid, refused), and a file is renamed into place only once complete,
so an interrupted build resumes.
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _one(job):
    from helix.data.cells import build_shard
    shard, source, out, grid = job
    return os.path.basename(shard), build_shard(shard, source, out, grid=grid)[1]


def _derive(job):
    from helix.data.cells import derive_shard
    src, out, grid = job
    return os.path.basename(src), derive_shard(src, out, grid)[1]


def _finer(src_dir, spec):
    """``[(source file, output name)]``: each ``{name}_{cells|pixels}_<k>.h5`` of ``src_dir``
    whose index is in ``spec``, written as ``{name}_cells_<k>.h5``."""
    out = []
    for p in sorted(glob.glob(os.path.join(src_dir, "*.h5"))):
        m = re.match(r"^(.*)_(?:cells|pixels)_(\d+)\.h5$", os.path.basename(p))
        if m:
            out.append((p, f"{m.group(1)}_cells_{m.group(2)}.h5", int(m.group(2))))
    want = _indices(spec)
    return [(p, name) for p, name, k in out if want is None or k in want]


def _indices(spec):
    """The shard indices of ``spec`` ("a-b" inclusive, or "a,b,c"); None (all) if empty."""
    if not spec:
        return None
    want = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        want.update(range(int(lo), int(hi or lo) + 1))
    return want


def _select(shards, spec):
    """Coeff shards whose index is in ``spec``."""
    want = _indices(spec)
    return [s for s in shards if want is None or int(re.search(r"_coeff_(\d+)\.h5$", s).group(1)) in want]


def main():
    from helix.data.cells import parse_grid
    from helix.paths import root
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", help="one corpus run directory")
    ap.add_argument("--from", dest="src", help="a finer truth's directory, summed onto --grid (needs --out)")
    ap.add_argument("--grid", required=True, type=parse_grid, help="cell size, WIRESxTICKS (d1m: 2x16)")
    ap.add_argument("--out", default=None, help="output directory (default: beside the coeff shards)")
    ap.add_argument("--shards", default="", help="shard indices, 'a-b' or 'a,b,c' (default: all)")
    ap.add_argument("--workers", type=int, default=0, help="processes (default: one per CPU)")
    a = ap.parse_args()
    if bool(a.corpus) == bool(a.src):
        raise SystemExit("give one of --corpus (build from the simulation) and --from (sum a finer truth)")
    if a.src:
        if not a.out:
            raise SystemExit("--from needs --out")
        jobs = [(p, os.path.join(a.out, name), a.grid) for p, name in _finer(a.src, a.shards)]
        if not jobs:
            raise SystemExit(f"{a.src}: no *_cells_<k>.h5 / *_pixels_<k>.h5{' in ' + a.shards if a.shards else ''}")
        os.makedirs(a.out, exist_ok=True)
        with Pool(min(a.workers or len(os.sched_getaffinity(0)), len(jobs))) as pool:
            for name, msg in pool.imap_unordered(_derive, jobs):
                print(f"{name}: {msg}", flush=True)
        return
    source = str(root("HELIX_SIM_ROOT"))
    shards = _select(sorted(glob.glob(os.path.join(a.corpus, "*_coeff_[0-9]*.h5"))), a.shards)
    if not shards:
        raise SystemExit(f"{a.corpus}: no *_coeff_<k>.h5 shards{' in ' + a.shards if a.shards else ''}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    n = min(a.workers or len(os.sched_getaffinity(0)), len(shards))
    with Pool(n) as pool:
        for name, msg in pool.imap_unordered(_one, [(s, source, a.out, a.grid) for s in shards]):
            print(f"{name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
