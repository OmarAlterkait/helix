"""CLI entry point for TPC batch processing (`helix-tpc`)."""
from __future__ import annotations

import argparse
import time


def main():
    parser = argparse.ArgumentParser(
        prog="helix-tpc",
        description="HELIX TPC — coherent noise removal + wavelet sparsification for LArTPC wire data")
    parser.add_argument("--input", required=True, help="Input sensor HDF5 file")
    parser.add_argument("--output", required=True, help="Output processed HDF5 file")
    parser.add_argument("--events", default=None, help="Event range, e.g. '0-19' or '5' (default: all)")
    parser.add_argument("--coh-only", action="store_true", help="Skip wavelet step")
    parser.add_argument("--backend", choices=["numpy", "jax", "torch"], default="numpy")
    args = parser.parse_args()

    from helix.core import backend
    backend.set_backend(args.backend)

    from helix.tpc.io import config_from_file, count_events, read_sensor_event, write_processed
    from helix.tpc.coherent import remove_coherent
    from helix.core.wavelet import sparsify

    print(f"HELIX TPC | backend: {backend.get_backend()}")
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
    print(f"Config: group_size={config.group_size}, wavelet={config.wavelet} L={config.dwt_level}\n")

    t0 = time.perf_counter()
    for idx in event_range:
        t_evt = time.perf_counter()
        planes = read_sensor_event(args.input, idx, config)
        results = {}
        for label, image in planes.items():
            cleaned = remove_coherent(image, config)
            if not args.coh_only:
                results[label] = sparsify(cleaned, wavelet=config.wavelet, level=config.dwt_level,
                                          mode=config.dwt_mode, threshold=config.threshold_spec())
        if not args.coh_only:
            write_processed(args.output, idx, results, config)
        print(f"  event {idx:>4d}: {len(planes)} planes, {(time.perf_counter()-t_evt)*1000:.0f} ms")
    print(f"\nDone. {len(event_range)} events in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
