"""CLI entry point for batch processing."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog="helix",
        description="HELIX — coherent noise removal + wavelet sparsification for LArTPC wire data",
    )
    parser.add_argument("--input", required=True, help="Input sensor HDF5 file")
    parser.add_argument("--output", required=True, help="Output processed HDF5 file")
    parser.add_argument("--events", default=None, help="Event range, e.g. '0-19' or '5' (default: all)")
    parser.add_argument("--coh-only", action="store_true", help="Skip wavelet step, output cleaned images")
    parser.add_argument("--backend", choices=["auto", "jax", "numpy"], default="auto")
    args = parser.parse_args()

    from helix.io import config_from_file, count_events, read_sensor_event, write_processed
    from helix.coherent import remove_coherent
    from helix.wavelet import sparsify
    from helix._backend import get_backend

    if args.backend != "auto":
        import helix._backend as _b
        _b._backend = args.backend

    print(f"HELIX v0.1.0 | backend: {get_backend()}")

    config = config_from_file(args.input)

    n_events = count_events(args.input)
    if args.events:
        if "-" in args.events:
            lo, hi = args.events.split("-")
            event_range = range(int(lo), int(hi) + 1)
        else:
            event_range = range(int(args.events), int(args.events) + 1)
    else:
        event_range = range(n_events)

    print(f"Input:  {args.input} ({n_events} events)")
    print(f"Events: {event_range.start}–{event_range.stop - 1}")
    print(f"Config: group_size={config.group_size}, β={config.beta}, "
          f"wavelet={config.wavelet} L={config.dwt_level}\n")

    t0 = time.perf_counter()
    for idx in event_range:
        t_evt = time.perf_counter()
        planes = read_sensor_event(args.input, idx, config)

        results = {}
        for label, image in planes.items():
            cleaned = remove_coherent(image, config)
            if not args.coh_only:
                results[label] = sparsify(cleaned, config)

        if not args.coh_only:
            write_processed(args.output, idx, results, config)

        elapsed = time.perf_counter() - t_evt
        n_planes = len(planes)
        print(f"  event {idx:>4d}: {n_planes} planes, {elapsed*1000:.0f} ms")

    total = time.perf_counter() - t0
    print(f"\nDone. {len(event_range)} events in {total:.1f}s")


if __name__ == "__main__":
    main()
