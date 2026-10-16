"""Compress SBND events with the configurable TPC chain.

    python -m helix.tpc.sbnd_run --input RUN_DIR/*.h5 --output out.h5 --device cuda \\
        [--config overrides.json|yaml] [--encoding compact|columns] [--workers 4] \\
        [--first 0 --count 100] [--profile]

Reads the extracted HDF5 files (:mod:`helix.tpc.io_sbnd`, decoded in worker processes),
runs :class:`helix.tpc.chain.TPCChain` on each event (numpy without ``--device``) and
writes :mod:`helix.tpc.sparse_io`. Prints per-event and summary timing: waiting for
input, the chain (per stage with ``--profile``), extracting and writing.
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
    ap.add_argument("--encoding", default="compact", choices=["compact", "columns"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="take every stride-th event")
    ap.add_argument("--spread", type=int, default=None, help="take N events evenly over all input events")
    ap.add_argument("--events-log", default=None, help="per-event jsonl (kept per plane / system, timing)")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-optical", action="store_true", help="TPC only")
    ap.add_argument("--deflate", type=int, default=4, help="deflate level of the output (0-9)")
    ap.add_argument("--flush-every", type=int, default=8)
    ap.add_argument("--optical-config", default=None, help="per-system OpticalSpec overrides (helix.optical.codec)")
    ap.add_argument("--serial-optical", action="store_true",
                    help="run the optical codec after the TPC chain (default on CUDA: in a thread on its own stream, "
                         "overlapping the TPC chain of the same event)")
    a = ap.parse_args(argv)

    from helix.tpc.chain import TPCChain
    from helix.tpc.io_sbnd import SBNDEvents, loader
    from helix.tpc.sparse_io import Writer
    from helix.optical.codec import OpticalChain
    systems = () if a.no_optical else ("pmt", "xarapuca")
    ochain = OpticalChain(_overrides(a.optical_config), device=a.device) if systems else None

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
    w = Writer(a.output, chain, encoding=a.encoding, level=a.deflate, flush_every=a.flush_every)
    # compression and HDF5 writes run in a writer thread, overlapping the next event's
    # device work (the main thread only extracts the sparse entries to the host)
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
        q.put(w.prepare(res, ores, run=m["run"], subrun=m["subrun"], event=m["event"], time=m["time"]))
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
                   workers=a.workers, encoding=a.encoding, deflate=a.deflate,
                   optical_compression={s: osamp[s] / max(okept[s], 1) for s in systems})
    if a.device and a.device.startswith("cuda"):
        import torch
        summary["gpu_max_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        summary["gpu"] = torch.cuda.get_device_name()
    import os
    summary["output_mb"] = os.path.getsize(a.output) / 1e6
    import h5py
    with h5py.File(a.output, "r") as f:                          # stored (compressed) bytes per system
        def stored(g):
            tot = 0
            def v(_, o):
                nonlocal tot
                if isinstance(o, h5py.Dataset):
                    tot += o.id.get_storage_size()
            g.visititems(v)
            return tot
        summary["stored_mb_per_event"] = {"tpc": stored(f["coeffs"]) / 1e6 / max(n, 1)}
        for s_ in systems:
            summary["stored_mb_per_event"][s_] = stored(f["optical"][s_]) / 1e6 / max(n, 1)
    summary["output_mb_per_event"] = summary["output_mb"] / max(n, 1)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
