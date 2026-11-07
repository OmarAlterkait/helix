"""Build shards: the events of a source through the TPC chain and the optical codec.

helix holds no source. A source is a small class in the script that builds from it
(``scripts/build_sim.py``, ``scripts/build_sbnd.py``, ``scripts/build_goop.py``) and only reads;
the script ends with ``helix.build.main(Source)`` (``from helix.build import main``: ``helix.build`` as
an attribute of the package is the function :func:`build`), which gives it this command line:

    python scripts/build_<source>.py --input FILE [FILE ...] --output DIR/PREFIX --device cuda \\
        [--first 0] [--count N | --spread N] [--events-per-shard 100] [--shard-start K] \\
        [--norm-sigma TABLE.npy | --calibrate TABLE.npy] [--config chain.json|yaml] \\
        [--optical-config ...] [--no-optical] [--keep-clean] [--workers 4] [--deflate 4] [--events-log F] [--profile]
    python scripts/build_<source>.py plan OUT RUN_DIR... [--events-per-task 500] [--events-per-shard 100]

and ``scripts/corpus.sh`` runs a whole corpus of it on the site's scheduler. A source:

* ``Source(files, device, optical=True)``, ``len(source)`` events, ``source.systems`` (optical
  systems; empty: TPC only), ``source.dataset`` (the shards' dataset name), ``source.pattern``
  (its input files in a run directory);
* ``source.events(indices, workers)`` yields per event a dict: ``ident`` (run, source_file, event,
  ...), the TPC input as ``readout`` (adc, pedestal, present: :meth:`TPCChain.process_event`) or
  ``planes`` ({gid: image}: :meth:`TPCChain.process_planes`), optionally ``clean`` (noise-free
  planes, written with ``--keep-clean``), ``optical`` ({system: waveforms}) and ``truth`` ({name: array});
* ``source.shard_attrs()``: dict(noise=, provenance=) the coeff shards record about the input;
* for a TPC: ``chain_config(overrides)`` and ``bind(chain)``; for optical: ``optical_chain(overrides)``;
* optionally ``file_plan(path, events_per_shard)`` -> ``(events, first shard)``: plan one task per
  input file, its shards numbered from that.

Everything after the read is shared. Per ``--events-per-shard`` events, shard k
(``--shard-start`` + 0, 1, ...): ``PREFIX_coeff_k.h5`` (:mod:`helix.core.coeff_io`; the dataset
name is PREFIX's last component), ``PREFIX_coeff_clean_k.h5`` (with ``--keep-clean``, when the source gives noise-free planes),
``PREFIX_optical_k.h5`` (:mod:`helix.optical.shard`), joined on (run, source_file, event).
``norm_sigma`` is ``--norm-sigma`` (one table for a whole corpus) or else each shard's own;
``--calibrate TABLE`` writes the table of the selected events and no shards. A JSON summary
(timing, storage, compression) is printed last.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

def _load_config(path):
    if not path:
        return None
    txt = open(path).read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(txt)
    return json.loads(txt)


def normalization_table(events) -> np.ndarray:
    """``norm_sigma`` (n_gid, n_bands) = the mean over ``events`` (CoeffEvents) of their
    per-event threshold sigma."""
    tab = np.stack([e.sigma_threshold for e in events]).mean(axis=0).astype(np.float32)
    if tab.size and float(np.nanmax(tab)) <= 0.0:
        raise ValueError("norm_sigma is all zero: no event carried a threshold sigma")
    return tab


# ------------------------------------------------------------------------------------- shards
def _stack_padded(arrays):
    """Stack arrays of one rank, each axis zero-padded to the largest."""
    shape = np.max([a.shape for a in arrays], axis=0)
    return np.stack([np.pad(a, [(0, s - n) for s, n in zip(shape, a.shape)]) for a in arrays])


class ShardWriter:
    """Shard k of PREFIX per ``events_per_shard`` events: coeff (+ clean) and optical files."""

    def __init__(self, prefix, *, events_per_shard, shard_start, norm, attrs, deflate):
        self.prefix, self.eps, self.k, self.norm, self.attrs, self.deflate = prefix, events_per_shard, shard_start, \
            norm, attrs, deflate
        self.dataset = Path(prefix).name
        self.outputs = []
        self._reset()

    def _reset(self):
        self.noisy, self.clean, self.seeds, self.truth, self.ow, self.n = [], [], [], {}, None, 0

    def add(self, ce, clean, ores, ident, truth):
        from helix.optical.shard import OpticalWriter
        if ce is not None:
            self.noisy.append(ce); self.seeds.append(ident.get("noise_seed", -1))
            if clean is not None:
                self.clean.append(clean)
        if ores:
            if self.ow is None:
                self.ow = OpticalWriter(self.path("optical") + ".part", level=self.deflate)
            self.ow.add(ores, **{k: v for k, v in ident.items() if k != "noise_seed"})
        for name, arr in (truth or {}).items():
            self.truth.setdefault(name, []).append(np.asarray(arr))
        self.n += 1
        if self.n >= self.eps:
            self.close()

    def path(self, kind, k=None):
        return f"{self.prefix}_{kind}_{self.k if k is None else k:04d}.h5"

    def _done(self, kind):
        """Shards are written as ``.part`` and renamed whole, so a file with the final name is complete."""
        os.replace(self.path(kind) + ".part", self.path(kind))
        self.outputs.append(self.path(kind))

    def close(self):
        from helix.core.coeff_io import audit_shard, write_coeff_shard
        if self.noisy:
            if self.norm is not None:
                want = (len(self.noisy[0].gids), self.noisy[0].sigma_threshold.shape[1])
                if self.norm.shape != want or float(np.nanmax(self.norm)) <= 0.0:
                    raise ValueError(f"norm_sigma: shape {self.norm.shape} (expected {want}), all zero or not")
            kw = dict(dataset_name=self.dataset, file_index=self.k,
                      norm_sigma=normalization_table(self.noisy) if self.norm is None else self.norm,
                      noise=self.attrs["noise"], provenance=self.attrs["provenance"],
                      noise_seeds=self.seeds if any(s >= 0 for s in self.seeds) else None)
            if self.clean:
                p = self.path("coeff_clean") + ".part"
                write_coeff_shard(p, self.clean, coords=False, **kw); audit_shard(p); self._done("coeff_clean")
            p = self.path("coeff") + ".part"
            write_coeff_shard(p, self.noisy, **kw); audit_shard(p); self._done("coeff")
        if self.ow is not None:
            for name, arrs in self.truth.items():
                self.ow.f.create_dataset(f"truth/{name}", data=_stack_padded(arrs))
            self.ow.close(); self._done("optical")
        if self.n:
            self.k += 1
        self._reset()


# ------------------------------------------------------------------------------------- build
def select(n, first=0, count=None, spread=None):
    """Event indices: ``count`` from ``first`` (all by default), or ``spread`` evenly over all."""
    if spread:
        return sorted(set(np.linspace(0, n - 1, spread).astype(int).tolist()))
    return list(range(min(first, n), n if count is None else min(n, first + count)))


def build(source, prefix, *, indices, chain=None, ochain=None, events_per_shard=100, shard_start=0,
          norm_sigma=None, calibrate=None, workers=4, deflate=4, events_log=None, keep_clean=False):
    """Run ``indices`` of ``source`` through ``chain`` (TPC; None: optical only) and ``ochain``
    (optical; None: TPC only) into shards at ``prefix`` (none with ``calibrate``: the path the
    normalization table of the events is saved to). ``keep_clean`` also writes the source's clean
    target, when it has one (nothing trains on it). Returns the summary dict."""
    from helix.tpc import chain as tpc_chain
    systems = tuple(source.systems) if ochain is not None else ()
    if chain is not None:
        source.bind(chain)
    device = (chain or ochain).device
    cuda = str(device).startswith("cuda")
    import torch
    sync = torch.cuda.current_stream().synchronize if cuda else (lambda: None)
    # the optical codec is host-heavy (many small kernels, Python per group): on CUDA, with a TPC
    # chain, it runs in a thread on its own stream, overlapping the TPC chain of the same event
    pool = ostream = None
    if cuda and systems and chain is not None:
        import concurrent.futures
        pool = concurrent.futures.ThreadPoolExecutor(1); ostream = torch.cuda.Stream()

    def optical(ev):
        if not systems:
            return []
        if ostream is None:
            return [ochain.process(s, ev["optical"][s]) for s in systems]
        with torch.cuda.stream(ostream):
            out = [ochain.process(s, ev["optical"][s]) for s in systems]
        ostream.synchronize()
        return out

    if calibrate is None:
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    prefix = prefix or ""
    attrs = source.shard_attrs()
    if attrs["provenance"] is not None or chain is not None:      # the GPU generation is part of a build
        attrs["provenance"] = dict(attrs["provenance"] or {},
                                   device=f"torch:{torch.cuda.get_device_name()}" if cuda else "torch:cpu")
    w = ShardWriter(prefix, events_per_shard=events_per_shard, shard_start=shard_start, norm=norm_sigma,
                    attrs=attrs, deflate=deflate)
    # shards are written in a writer thread, overlapping the next event's device work
    q, wstat = queue.Queue(maxsize=4), dict(busy=0.0, error=None)

    def writer():
        while (item := q.get()) is not None:
            tw = time.perf_counter()
            try:
                w.add(*item)
            except Exception as e:                                  # surfaced after the loop
                wstat["error"] = e
            wstat["busy"] += time.perf_counter() - tw

    th = threading.Thread(target=writer, daemon=True)
    if calibrate is None:
        th.start()
    T = dict(wait=0.0, chain=0.0, optical=0.0, extract=0.0)
    okept, osamp = {s: 0 for s in systems}, {s: 0 for s in systems}
    kept, cal, n_done = [], [], 0
    elog = open(events_log, "w") if events_log else None
    t_start = t = time.perf_counter()
    for ev in source.events(indices, workers):
        t1 = time.perf_counter(); T["wait"] += t1 - t
        fut = pool.submit(optical, ev) if pool is not None else None
        res = None
        if chain is not None:
            res = chain.process_event(*ev["readout"]) if "readout" in ev else chain.process_planes(ev["planes"])
        sync(); t2 = time.perf_counter(); T["chain"] += t2 - t1
        ores = fut.result() if fut is not None else optical(ev)
        sync(); t3 = time.perf_counter(); T["optical"] += t3 - t2
        idn = ev["ident"]
        ce = clean = None
        if res is not None:
            ce = tpc_chain.coeff_event(chain, res, run=idn["run"], source_file=idn["source_file"], event=idn["event"])
            if keep_clean and ev.get("clean") is not None and calibrate is None:
                clean = tpc_chain.clean_coeff_event(chain, ce, ev["clean"])
            kept.append(sum(int(r["n_kept"]) for r in res))
        if calibrate is None:
            q.put((ce, clean, ores, idn, ev.get("truth")))
        else:
            cal.append(ce)
        for r in ores:
            okept[r["system"]] += r["n_kept"]; osamp[r["system"]] += r["n_samples"]
        t = time.perf_counter(); T["extract"] += t - t3
        if elog is not None:
            elog.write(json.dumps(dict(**{k: idn[k] for k in ("run", "event")}, chain_s=t2 - t1, optical_s=t3 - t2,
                                       tpc={f"{r['tpc']}{r['plane']}": int(r["n_kept"]) for r in res or []},
                                       optical={r["system"]: dict(kept=int(r["n_kept"]), samples=int(r["n_samples"]),
                                                                  waveforms=int(len(r["length"]))) for r in ores})) + "\n")
        n_done += 1
        if n_done <= 3 or n_done % 50 == 0:
            nk = kept[-1] if res is not None else sum(int(r["n_kept"]) for r in ores)
            print(f"event {idn['run']}:{idn['event']} kept {nk} chain {t2 - t1:.3f}s", flush=True)
    tc = time.perf_counter()
    if calibrate is None:
        q.put(None); th.join(); w.close()
    if pool is not None:
        pool.shutdown()
    T["write_drain"] = time.perf_counter() - tc
    if elog is not None:
        elog.close()
    if wstat["error"] is not None:
        raise wstat["error"]
    n = n_done
    wall = time.perf_counter() - t_start
    summary = dict(events=n, wall_s=wall, events_per_s=n / max(wall, 1e-9),
                   per_event_s={k: v / max(n, 1) for k, v in T.items()},
                   kept_mean=float(np.mean(kept)) if kept else 0.0, device=str(device),
                   writer_thread_busy_s_per_event=wstat["busy"] / max(n, 1), workers=workers, deflate=deflate,
                   outputs=w.outputs, optical_compression={s: osamp[s] / max(okept[s], 1) for s in systems})
    if chain is not None:
        summary.update(config_digest=chain.digest, stages_per_event_s=None if chain.timings is None else
                       {k: v / max(n, 1) for k, v in chain.timings.items()})
    if calibrate is not None:
        table = normalization_table(cal)
        Path(calibrate).parent.mkdir(parents=True, exist_ok=True)
        np.save(calibrate, table)
        summary.update(calibrated=str(calibrate), norm_sigma_shape=list(table.shape))
    if cuda:
        summary.update(gpu_max_memory_gb=torch.cuda.max_memory_allocated() / 1e9, gpu=torch.cuda.get_device_name())
    summary.update(_storage(w.outputs, systems, n))
    return summary


def _storage(outputs, systems, n):
    import h5py

    def stored(path, group=None):
        tot = 0
        with h5py.File(path, "r") as f:
            for o in _datasets(f[group] if group else f):
                tot += o.id.get_storage_size()
        return tot
    tpc = [p for p in outputs if "_coeff_" in p and "_coeff_clean_" not in p]
    opt = [p for p in outputs if "_optical_" in p]
    out = {"tpc": sum(stored(p) for p in tpc) / 1e6 / max(n, 1)}
    for s in systems:
        out[s] = sum(stored(p, f"optical/{s}") for p in opt if s in h5py.File(p, "r")["optical"]) / 1e6 / max(n, 1)
    return dict(stored_mb_per_event=out, output_mb_per_event=sum(os.path.getsize(p) for p in outputs) / 1e6 / max(n, 1))


def _datasets(g):
    import h5py
    out = []
    g.visititems(lambda _, o: out.append(o) if isinstance(o, h5py.Dataset) else None)
    return out


def plan(Source, out, run_dirs, events_per_task, events_per_shard):
    """A corpus's tasks, one line each: ``PREFIX FIRST COUNT EVENTS_PER_SHARD SHARD_START INPUT...``.
    One task per input file when the source numbers shards by file (``Source.file_plan``), else
    every ``events_per_task`` events of a run's files."""
    lines = []
    for d in map(Path, run_dirs):
        prefix = f"{out}/{d.name}/{Source.dataset}"
        files = sorted(str(f) for f in d.glob(Source.pattern))
        if not files:
            raise ValueError(f"{d}: no {Source.pattern} input files")
        if hasattr(Source, "file_plan"):
            for f in files:
                n, k = Source.file_plan(f, events_per_shard)
                lines.append(f"{prefix} 0 {n} {events_per_shard} {k} {f}")
            continue
        n = len(Source(files, None))
        for first in range(0, n, events_per_task):
            if first % events_per_shard:
                raise ValueError("events_per_task must be a multiple of events_per_shard")
            lines.append(f"{prefix} {first} {min(events_per_task, n - first)} {events_per_shard} "
                         f"{first // events_per_shard} {' '.join(files)}")
    return lines


def _complete(prefix, kind, shard_start, count, events_per_shard):
    """Every shard a task would write exists (shards appear only when complete)."""
    return all(Path(f"{prefix}_{kind}_{k:04d}.h5").exists()
               for k in range(shard_start, shard_start + -(-count // events_per_shard)))


def main(Source, argv=None):
    """The command line of a source's build script (the module docstring)."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "plan":
        pp = argparse.ArgumentParser(prog=f"{Path(sys.argv[0]).name} plan", description=plan.__doc__)
        pp.add_argument("out")
        pp.add_argument("run_dirs", nargs="+")
        pp.add_argument("--events-per-task", type=int, default=500)
        pp.add_argument("--events-per-shard", type=int, default=100)
        b = pp.parse_args(argv[1:])
        print("\n".join(plan(Source, b.out, b.run_dirs, b.events_per_task, b.events_per_shard)))
        return None
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output", default=None, help="DIR/PREFIX of the shards (not with --calibrate)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--spread", type=int, default=None, help="N events evenly over all input events")
    ap.add_argument("--events-per-shard", type=int, default=100)
    ap.add_argument("--shard-start", type=int, default=0, help="number of the first shard")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--norm-sigma", default=None, help="the corpus-wide (G, n_bands) table (.npy) for every shard")
    g.add_argument("--calibrate", default=None, help="save the selected events' table here; write no shards")
    ap.add_argument("--config", default=None, help="TPC chain overrides (json / yaml)")
    ap.add_argument("--optical-config", default=None, help="per-system OpticalSpec overrides")
    ap.add_argument("--no-optical", action="store_true", help="TPC only")
    ap.add_argument("--workers", type=int, default=4, help="reader processes (sbnd)")
    ap.add_argument("--deflate", type=int, default=4, help="deflate level of the optical shards")
    ap.add_argument("--events-log", default=None, help="per-event jsonl (kept per plane / system, timing)")
    ap.add_argument("--profile", action="store_true", help="time the TPC chain's stages")
    ap.add_argument("--keep-clean", action="store_true",
                    help="also write PREFIX_coeff_clean_k.h5 from a source's noise-free planes")
    ap.add_argument("--skip-complete", action="store_true",
                    help="do nothing when every shard of this selection exists (a resubmitted task)")
    a = ap.parse_args(argv)
    if (a.output is None) == (a.calibrate is None):
        ap.error("--output (the shards) or --calibrate (a table, no shards): exactly one")
    kind = "coeff" if hasattr(Source, "chain_config") else "optical"
    if a.skip_complete and a.count and a.calibrate is None and _complete(
            a.output, kind, a.shard_start, a.count, a.events_per_shard):
        print(f"complete: {a.output} shards from {a.shard_start}")
        return None

    from helix.tpc.chain import TPCChain
    src = Source(a.input, a.device, optical=not a.no_optical)
    chain = ochain = None
    if hasattr(src, "chain_config"):
        chain = TPCChain(src.chain_config(_load_config(a.config)), device=a.device)
        if a.profile:
            chain.enable_timing()
    if src.systems and not a.no_optical and a.calibrate is None:
        ochain = src.optical_chain(_load_config(a.optical_config))
    norm = None if a.norm_sigma is None else np.load(a.norm_sigma).astype(np.float32)
    summary = build(src, a.output, indices=select(len(src), a.first, a.count, a.spread), chain=chain, ochain=ochain,
                    events_per_shard=a.events_per_shard, shard_start=a.shard_start, norm_sigma=norm,
                    calibrate=a.calibrate, workers=a.workers, deflate=a.deflate, events_log=a.events_log,
                    keep_clean=a.keep_clean)
    print(json.dumps(summary, indent=1))
    return summary
