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


def _event_ids(spec, ids):
    """The event IDS to process, filtered to those actually present.

    This used to be `_event_range(spec, n)` over `range(count_events(...))`,
    which assumes ids are 0..n-1. `list_events`'s own docstring says they are
    not: production files can be missing an id in the middle. Measured on the
    real corpus, 3 of 600 shards of wire_test_00_00_02 have a gap -- e.g.
    run_0027575715/sim_wire_sensor_0065.h5 holds 199 events spanning 0..199
    with 167 ABSENT. range(199) then asks for 167, which raises
    KeyError('event 167 not found'), and never asks for 199, which is there.
    Since write_coeff_shard runs after the loop, --to-coeffs wrote 0 of 199
    events on such a shard.

    `--events lo-hi` is an inclusive range over IDS, not positions, and silently
    yields nothing rather than failing if the range covers only absent ids --
    the caller sees the printed count.
    """
    present = list(ids)
    if not spec:
        return present
    if "-" in spec:
        lo, hi = spec.split("-")
        lo, hi = int(lo), int(hi)
    else:
        lo = hi = int(spec)
    return [e for e in present if lo <= e <= hi]


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
    from helix.tpc.io import config_from_file, list_events, read_sensor_event, write_processed
    from helix.tpc.pipeline import process_event, event_coeff_event, canonical_plane_gid
    from helix.core.coeff_io import write_coeff_shard

    config = config_from_file(args.input)
    ids = list_events(args.input)
    n = len(ids)
    events = _event_ids(args.events, ids)
    if not events:
        raise SystemExit(f"--events {args.events!r} selects none of the {n} ids "
                         f"present in {args.input} (range {min(ids)}..{max(ids)})"
                         if ids else f"{args.input} contains no events")
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
                by_gid, config, run=src.parent.name, source_file=src.name, event=idx, removal=removal))
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
