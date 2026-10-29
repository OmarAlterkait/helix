"""Bring a corpus built before the format stamp to ``helix-coeff-2``, in place, metadata only.

    python scripts/migrate_format.py SOURCE CORPUS_ROOT RUN_DIR... [--events-per-task 500] [--dry-run]

SOURCE is a build script (sim, sbnd, or a path, as for scripts/corpus.sh); RUN_DIR... are the
input runs the corpus was built from, in its order. No coefficient changes. Per coeff /
coeff_clean shard:

* ``format`` is stamped;
* ``removal_json`` and ``basis_digest`` become what the current chain records for this source
  (the coherent gate's schema changed to the readout-groups form); every other basis field
  must already be what the current chain records, or nothing is touched;
* ``noise_json`` loses ``mode`` and ``backend`` (one noise implementation remains).

A run directory is renamed to its input run's name where it differs (``run018251`` ->
``018251``), as ``helix.build.plan`` names them.

Before anything is written, every shard's old ``/config`` attributes go to
``_calib/format1_attrs.json``, and an existing ``_calib/CHUNKS.txt`` to ``CHUNKS.txt.format1``;
then the corpus gets its plan (``_calib/CHUNKS.txt``, ``RUNS.txt``) as
``scripts/corpus.sh plan`` writes it. Shards already stamped are skipped, so a rerun finishes an
interrupted migration. Check the result with ``python -m helix.data.coeff_verify CORPUS_ROOT``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import runpy
import sys

import h5py
import numpy as np

from helix.build import plan
from helix.core.coeff_io import FORMAT, shard_format
from helix.core.provenance import BasisDescriptor
from helix.tpc import chain_coeff
from helix.tpc.chain import TPCChain

HERE = os.path.dirname(os.path.abspath(__file__))


def _text(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def source_class(source):
    path = source if os.path.exists(source) else os.path.join(HERE, f"build_{source}.py")
    ns = runpy.run_path(path)
    return next(v for k, v in ns.items() if k.endswith("Source") and isinstance(v, type) and hasattr(v, "dataset"))


def new_basis(Source, run_dir, cfg):
    """The basis the current chain records for this source, on the shard's own geometry."""
    files = sorted(glob.glob(os.path.join(run_dir, Source.pattern)))
    src = Source(files[:1] if hasattr(Source, "file_plan") else files, None, optical=False)
    chain = TPCChain(src.chain_config(None))
    return chain_coeff.basis(chain, cfg["band_lengths"][:], int(cfg.attrs["n_ticks_raw"]),
                             float(cfg.attrs["sigma_norm"]))


def old_basis(cfg):
    return BasisDescriptor(wavelet=_text(cfg.attrs["wavelet"]), level=int(cfg.attrs["dwt_level"]),
                           mode=_text(cfg.attrs["dwt_mode"]), n_ticks_raw=int(cfg.attrs["n_ticks_raw"]),
                           pad=int(cfg.attrs["pad"]), band_lengths=tuple(int(x) for x in cfg["band_lengths"][:]),
                           removal=json.loads(_text(cfg.attrs["removal_json"])),
                           threshold=json.loads(_text(cfg.attrs["threshold_json"])),
                           sigma_norm=float(cfg.attrs["sigma_norm"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("root")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--events-per-task", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true", help="check and report; write nothing")
    a = ap.parse_args(argv)
    Source = source_class(a.source)
    cal = os.path.join(a.root, "_calib")
    runs = {os.path.basename(d.rstrip("/")): d for d in a.run_dirs}
    shards = {}
    for d in sorted(glob.glob(os.path.join(a.root, "*", ""))):
        name = os.path.basename(d.rstrip("/"))
        if name.startswith("_"):
            continue
        shards[name] = sorted(glob.glob(os.path.join(d, f"{Source.dataset}_coeff_[0-9]*.h5"))
                              + glob.glob(os.path.join(d, f"{Source.dataset}_coeff_clean_[0-9]*.h5")))
    # the corpus's run dirs are named for their input run (sim) or "run" + it (sbnd)
    run_of = {name: next((r for r in runs if name in (r, f"run{r}")), None) for name in shards}
    missing = [n for n, r in run_of.items() if r is None]
    if missing:
        raise SystemExit(f"no input RUN_DIR given for corpus runs {missing}")

    # 1. what changes, checked on every shard before anything is written
    backup, todo, new = {}, [], {}
    for name, paths in shards.items():
        for p in paths:
            with h5py.File(p, "r") as f:
                cfg = f["config"]
                backup[os.path.relpath(p, a.root)] = {k: _text(v) if isinstance(v, (bytes, str)) else
                                                      np.asarray(v).tolist() for k, v in cfg.attrs.items()}
                if shard_format(cfg) == FORMAT:
                    continue
                key = (run_of[name], tuple(cfg["band_lengths"][:].tolist()), int(cfg.attrs["n_ticks_raw"]))
                if key not in new:
                    new[key] = new_basis(Source, runs[run_of[name]], cfg)
                b, o = new[key], old_basis(cfg)
                same = {f: getattr(o, f) == getattr(b, f) for f in
                        ("wavelet", "level", "mode", "n_ticks_raw", "pad", "band_lengths", "threshold", "sigma_norm")}
                if not all(same.values()):
                    raise SystemExit(f"{p}: the current chain records a different basis beyond the gate: "
                                     f"{[f for f, s in same.items() if not s]}; this corpus needs a rebuild")
                if _text(cfg.attrs["basis_digest"]) != o.digest():
                    raise SystemExit(f"{p}: its basis_digest does not describe its own attributes")
                noise = json.loads(_text(cfg.attrs["noise_json"]))
                for k in ("mode", "backend"):
                    noise.pop(k, None)
                todo.append((p, b, json.dumps(noise, sort_keys=True)))
    b0 = next(iter(new.values()), None)
    print(f"{sum(map(len, shards.values()))} shards over {len(shards)} runs; {len(todo)} to migrate")
    if b0 is not None:
        p0 = todo[0][0]
        print(f"removal_json: {backup[os.path.relpath(p0, a.root)]['removal_json']}\n"
              f"          ->  {json.dumps(b0.removal, sort_keys=True)}\n"
              f"basis_digest -> {b0.digest()}")
    eps = {int(h5py.File(ps[0], "r")["config"].attrs["n_events"]) for ps in shards.values()}
    lines = plan(Source, a.root, a.run_dirs, a.events_per_task, max(eps))
    if a.dry_run:
        print(f"dry run: {len(lines)} plan lines, nothing written")
        return 0

    # 2. backups, then the shards, then the run dirs' names, then the plan
    os.makedirs(cal, exist_ok=True)
    bk = os.path.join(cal, "format1_attrs.json")
    renames = {n: r for n, r in run_of.items() if n != r}
    if not os.path.exists(bk):
        with open(bk + ".part", "w") as fh:
            json.dump(dict(attrs=backup, renamed_run_dirs=renames), fh, indent=0, sort_keys=True)
        os.replace(bk + ".part", bk)
    ch = os.path.join(cal, "CHUNKS.txt")
    if os.path.exists(ch) and not os.path.exists(ch + ".format1"):
        os.replace(ch, ch + ".format1")
    for i, (p, b, noise) in enumerate(todo):
        with h5py.File(p, "r+") as f:
            cfg = f["config"]
            cfg.attrs["removal_json"] = json.dumps(b.removal, sort_keys=True)
            cfg.attrs["basis_digest"] = b.digest()
            cfg.attrs["noise_json"] = noise
            cfg.attrs["format"] = FORMAT                # last: a stamped shard is a migrated one
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)
    for n, r in renames.items():
        os.rename(os.path.join(a.root, n), os.path.join(a.root, r))
        print(f"  {n} -> {r}")
    with open(ch, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(cal, "RUNS.txt"), "w") as fh:
        fh.write("\n".join(os.path.basename(d.rstrip("/")) for d in a.run_dirs) + "\n")
    print(f"migrated {len(todo)} shards; plan {ch} ({len(lines)} tasks); old attributes in {bk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
