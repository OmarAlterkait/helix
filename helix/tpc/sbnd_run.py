"""Compress SBND events: the TPC chain and the optical codec, into HELIX's shards.

    python -m helix.tpc.sbnd_run --input RUN_DIR/*.h5 --output OUT/PREFIX --device cuda \\
        [--config overrides.json|yaml] [--optical-config ...] [--workers 4] \\
        [--first 0 --count 100] [--events-per-shard 100] [--profile] \\
        [--norm-sigma TABLE.npy --shard-start K]

Reads the extracted HDF5 files (:mod:`helix.tpc.io_sbnd`, decoded in worker processes), runs
:class:`helix.tpc.chain.TPCChain` (numpy without ``--device``) and the optical codec on each
event and writes, per ``--events-per-shard`` events, the TPC coefficient shard
``PREFIX_coeff_NNNN.h5`` (:mod:`helix.core.coeff_io`, the corpus format the model reads) and the
optical shard ``PREFIX_optical_NNNN.h5`` (:mod:`helix.optical.shard`), joined on (run,
source_file, event). Prints per-event and summary timing.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def _overrides(path):
    if not path:
        return None
    txt = open(path).read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(txt)
    return json.loads(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="take every stride-th event")
    ap.add_argument("--spread", type=int, default=None, help="take N events evenly over all input events")
    ap.add_argument("--events-log", default=None, help="per-event jsonl (kept per plane / system, timing)")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-optical", action="store_true", help="TPC only")
    ap.add_argument("--deflate", type=int, default=4, help="deflate level of the output (0-9)")
    ap.add_argument("--events-per-shard", type=int, default=100)
    ap.add_argument("--dataset-name", default="sbnd")
    ap.add_argument("--norm-sigma", default=None,
                    help="a corpus-wide (G, n_bands) norm_sigma table (.npy) written into every shard "
                         "(default: each shard's own, from its events; one corpus needs one table)")
    ap.add_argument("--shard-start", type=int, default=0,
                    help="number of the first shard (jobs splitting one run's events write disjoint shards)")
    ap.add_argument("--optical-config", default=None, help="per-system OpticalSpec overrides (helix.optical.codec)")
    ap.add_argument("--serial-optical", action="store_true",
                    help="run the optical codec after the TPC chain (default on CUDA: in a thread on its own stream, "
                         "overlapping the TPC chain of the same event)")
    a = ap.parse_args(argv)

    from helix.tpc.chain import TPCChain
    from helix.tpc.io_sbnd import SBNDEvents, loader
    from pathlib import Path
    from helix.core.coeff_io import write_coeff_shard, audit_shard
    from helix.optical.shard import OpticalWriter
    from helix.tpc import chain_coeff
    from helix.tpc.corpus import normalization_table
    from helix.optical.codec import OpticalChain
    systems = () if a.no_optical else ("pmt", "xarapuca")
    ochain = OpticalChain(_overrides(a.optical_config), device=a.device) if systems else None

    norm = None if a.norm_sigma is None else np.load(a.norm_sigma).astype(np.float32)
    files = sorted(a.input)
    n_all = len(SBNDEvents(files))
    last = n_all if a.count is None else min(n_all, a.first + a.count)
    idx = range(a.first, last if a.count is None else min(n_all, a.first + a.count * a.stride), a.stride)
    if a.spread:
        idx = sorted(set(np.linspace(0, n_all - 1, a.spread).astype(int).tolist()))
    chain = TPCChain(_overrides(a.config), device=a.device)
    if a.profile:
        chain.enable_timing()
    cuda = bool(a.device and a.device.startswith("cuda"))
    sync = (lambda: __import__("torch").cuda.current_stream().synchronize()) if cuda else (lambda: None)
    # the optical codec is host-heavy (many small kernels, Python per group): on CUDA it runs in a
    # thread on its own stream while the main thread runs the TPC chain of the same event
    pool = ostream = None
    if cuda and systems and not a.serial_optical:
        import concurrent.futures
        import torch
        pool = concurrent.futures.ThreadPoolExecutor(1); ostream = torch.cuda.Stream()

    def optical(ev):
        if ostream is None:
            return [ochain.process(s, ev[s]) for s in systems]
        import torch
        with torch.cuda.stream(ostream):
            out = [ochain.process(s, ev[s]) for s in systems]
        ostream.synchronize()
        return out
    T = dict(wait=0.0, chain=0.0, optical=0.0, write=0.0)   # write: sparse extraction (main thread)
    okept = {s: 0 for s in systems}; osamp = {s: 0 for s in systems}
    kept, n = [], 0
    t_start = time.perf_counter()
    prefix = Path(a.output); prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []

    class ShardWriter:
        """Per ``--events-per-shard`` events: one coeff_io TPC shard and one optical shard."""

        def __init__(self):
            self.k, self.ces, self.ow = a.shard_start, [], None

        def add_prepared(self, item):
            ce, ores, ident = item
            if self.ow is None and systems:
                self.ow = OpticalWriter(f"{prefix}_optical_{self.k:04d}.h5", level=a.deflate)
            self.ces.append(ce)
            if self.ow is not None:
                self.ow.add(ores, **ident)
            if len(self.ces) >= a.events_per_shard:
                self.close()

        def close(self):
            if self.ces:
                pth = f"{prefix}_coeff_{self.k:04d}.h5"
                write_coeff_shard(pth, self.ces, dataset_name=a.dataset_name, file_index=self.k,
                                  norm_sigma=normalization_table(self.ces) if norm is None else norm)
                audit_shard(pth); outputs.append(pth)
            if self.ow is not None:
                self.ow.close(); outputs.append(f"{prefix}_optical_{self.k:04d}.h5")
            self.ces, self.ow = [], None; self.k += 1
    w = ShardWriter()
    # shard writes run in a writer thread, overlapping the next event's device work (the main
    # thread extracts the sparse entries to the host)
    import queue
    import threading
    q = queue.Queue(maxsize=4)
    wstat = dict(busy=0.0, error=None)

    def writer():
        while True:
            item = q.get()
            if item is None:
                return
            tw = time.perf_counter()
            try:
                w.add_prepared(item)
            except Exception as e:          # surfaced after the loop
                wstat["error"] = e
            wstat["busy"] += time.perf_counter() - tw

    elog = open(a.events_log, "w") if a.events_log else None
    th = threading.Thread(target=writer, daemon=True)
    th.start()
    t = time.perf_counter()
    for ev in loader(files, workers=a.workers, indices=idx, optical_kinds=systems):
        t1 = time.perf_counter(); T["wait"] += t1 - t
        fut = pool.submit(optical, ev) if pool is not None else None
        res = chain.process_event(ev["adc"], ev["pedestal"], ev["present"])
        sync(); t2 = time.perf_counter(); T["chain"] += t2 - t1
        ores = fut.result() if fut is not None else optical(ev)       # overlapped: the wait beyond the TPC chain
        sync(); t3 = time.perf_counter(); T["optical"] += t3 - t2
        m = ev["meta"]
        ident = dict(run=str(m["run"]), source_file=Path(m["file"]).name, event=int(m["event"]), subrun=int(m["subrun"]),
                     time=float(m["time"]))
        ce = chain_coeff.coeff_event(chain, res, run=ident["run"], source_file=ident["source_file"], event=ident["event"])
        q.put((ce, ores, ident))
        for r in ores:
            okept[r["system"]] += r["n_kept"]; osamp[r["system"]] += r["n_samples"]
        t = time.perf_counter(); T["write"] += t - t3
        kept.append(sum(r["n_kept"] for r in res)); n += 1
        if elog is not None:
            elog.write(json.dumps(dict(run=m["run"], event=m["event"], time=m["time"], chain_s=t2 - t1, optical_s=t3 - t2,
                                       tpc={f"{r['tpc']}{r['plane']}": int(r["n_kept"]) for r in res},
                                       optical={r["system"]: dict(kept=int(r["n_kept"]), samples=int(r["n_samples"]),
                                                                  waveforms=int(len(r["length"]))) for r in ores})) + "\n")
        if n <= 3 or n % 50 == 0:
            print(f"event {m['run']}:{m['event']} kept {kept[-1]} chain {t2 - t1:.3f}s", flush=True)
    tc = time.perf_counter()
    q.put(None); th.join(); w.close()
    if pool is not None:
        pool.shutdown()
    T["write_drain"] = time.perf_counter() - tc
    if elog is not None:
        elog.close()
    if wstat["error"] is not None:
        raise wstat["error"]
    wall = time.perf_counter() - t_start
    summary = dict(events=n, wall_s=wall, events_per_s=n / max(wall, 1e-9),
                   per_event_s={k: v / max(n, 1) for k, v in T.items()},
                   stages_per_event_s=None if chain.timings is None else {k: v / max(n, 1) for k, v in chain.timings.items()},
                   kept_mean=float(np.mean(kept)) if kept else 0.0, config_digest=chain.digest, device=a.device,
                   writer_thread_busy_s_per_event=wstat["busy"] / max(n, 1),
                   workers=a.workers, deflate=a.deflate, outputs=outputs,
                   optical_compression={s: osamp[s] / max(okept[s], 1) for s in systems})
    if a.device and a.device.startswith("cuda"):
        import torch
        summary["gpu_max_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        summary["gpu"] = torch.cuda.get_device_name()
    import os
    import h5py

    def stored(path, group=None):
        tot = 0
        with h5py.File(path, "r") as f:
            def v(_, o):
                nonlocal tot
                if isinstance(o, h5py.Dataset):
                    tot += o.id.get_storage_size()
            (f[group] if group else f).visititems(v)
        return tot
    tpc = [p for p in outputs if "_coeff_" in p]; opt = [p for p in outputs if "_optical_" in p]
    summary["stored_mb_per_event"] = {"tpc": sum(stored(p) for p in tpc) / 1e6 / max(n, 1)}
    for s_ in systems:
        summary["stored_mb_per_event"][s_] = sum(stored(p, f"optical/{s_}") for p in opt
                                                 if s_ in h5py.File(p, "r")["optical"]) / 1e6 / max(n, 1)
    summary["output_mb_per_event"] = sum(os.path.getsize(p) for p in outputs) / 1e6 / max(n, 1)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
