"""CLI entry point for TPC batch processing (`helix-tpc`).

Two output modes:
  default        — legacy per-plane band-COO (`write_processed`), one group/event.
  --to-coeffs    — the coeff corpus: one flat-columnar shard of CoeffEvents
                   (`write_coeff_shard`), the input for the FM.

Coherent removal is selected by --removal (default: the config's, i.e. the
qualified smart gate).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def _event_range(spec, n):
    if not spec:
        return range(n)
    if "-" in spec:
        lo, hi = spec.split("-")
        return range(int(lo), int(hi) + 1)
    return range(int(spec), int(spec) + 1)


def main():
    p = argparse.ArgumentParser(
        prog="helix-tpc",
        description="HELIX TPC — coherent removal + wavelet sparsification for LArTPC wire data")
    p.add_argument("--input", required=True, help="Input sensor HDF5 file")
    p.add_argument("--output", required=True, help="Output HDF5 file")
    p.add_argument("--events", default=None, help="Event range '0-19' or '5' (default: all)")
    p.add_argument("--removal", choices=["gate", "multipass", "none"], default=None,
                   help="Coherent removal mode (default: config = smart gate)")
    p.add_argument("--to-coeffs", action="store_true",
                   help="Write the coeff corpus shard (CoeffEvents) instead of legacy per-plane output")
    p.add_argument("--backend", choices=["numpy", "jax", "torch"], default="numpy")
    args = p.parse_args()

    from helix.core import backend
    backend.set_backend(args.backend)
    from helix.tpc.io import config_from_file, count_events, read_sensor_event, write_processed
    from helix.tpc.pipeline import process_event, event_coeff_event, canonical_plane_gid
    from helix.core.coeff_io import write_coeff_shard

    config = config_from_file(args.input)
    n = count_events(args.input)
    events = _event_range(args.events, n)
    removal = args.removal or config.removal
    src = Path(args.input)
    print(f"HELIX TPC | backend={backend.get_backend()} | removal={removal} | "
          f"{'coeffs' if args.to_coeffs else 'legacy'}")
    print(f"Input:  {args.input} ({n} events); group_size={config.group_size}, "
          f"wavelet={config.wavelet} L{config.dwt_level}\n")

    t0 = time.perf_counter()
    coeff_events = []
    for idx in events:
        t = time.perf_counter()
        planes = read_sensor_event(args.input, idx, config)
        results = process_event(planes, config, removal=removal)
        if args.to_coeffs:
            by_gid = {canonical_plane_gid(lbl): pp for lbl, pp in results.items()}
            coeff_events.append(event_coeff_event(
                by_gid, config, run=src.parent.name, source_file=src.name, event=idx))
        else:
            write_processed(args.output, idx, {lbl: pp.sparse for lbl, pp in results.items()}, config)
        kept = sum(pp.sparse.n_kept for pp in results.values())
        print(f"  event {idx:>4d}: {len(planes)} planes, {kept:>7d} coeffs, "
              f"{(time.perf_counter()-t)*1000:.0f} ms")

    if args.to_coeffs:
        write_coeff_shard(args.output, coeff_events, dataset_name=src.stem,
                          file_index=0, global_event_offset=0)
        print(f"\nWrote {len(coeff_events)} events → {args.output}")
    print(f"Done. {len(list(events))} events in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
