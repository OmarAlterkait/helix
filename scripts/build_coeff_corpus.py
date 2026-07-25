#!/usr/bin/env python
"""Build the coeff corpus from real doraemon sensor shards (the production plane_fn).

Composes pimm-data (geometry + noise — the COLORED incoherent spectrum the old
build omitted, which defaulted to white) with helix (process_plane gate +
build_corpus). This is a BUILD-TIME composer: it imports both helix and pimm-data
(allowed at build time; the read path never does). Not part of the helix package.

plane_fn(event) -> (noisy_planes, clean_planes):
  noisy = digitize( clean_image + incoherent(colored) + coherent )
  clean = the noise-free digitized image (co-supported target at the noisy mask)

Usage:
  python build_coeff_corpus.py --shard <sensor.h5> --out <dir> --events 100 \
      --npz <noise_spectrum.npz> --geom cubic_wireplane_geometry.json --run <run>

Verified on run_0027575715: ~39k coeffs/plane, D1 ~1.8%, coeff+coeff_clean
co-supported, reads back through pimm_data.CoeffTPCDataset.
"""
import argparse
import hashlib
import sys
import time

import numpy as np


def _add_repo_paths(helix_root, pimm_src):
    for p in (helix_root, pimm_src):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="sensor HDF5 shard")
    ap.add_argument("--out", required=True, help="output corpus dir (<root>/coeff_tpc/<run>/)")
    ap.add_argument("--npz", default="/sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz",
                    help="colored incoherent series spectrum (freqs_hz, shape)")
    ap.add_argument("--geom", default="cubic_wireplane_geometry.json",
                    help="plane registry (pimm_data.geometry.load_plane_registry)")
    ap.add_argument("--dataset-name", default="wire_test_00_00_02")
    ap.add_argument("--run", default="")
    ap.add_argument("--file-index", type=int, default=0)
    ap.add_argument("--events", type=int, default=100)
    ap.add_argument("--event-start", type=int, default=0)
    ap.add_argument("--cal-events", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--white", action="store_true", help="use white incoherent noise (old bug)")
    ap.add_argument("--helix-root", default="/sdf/group/neutrino/omara/helix-consolidate")
    ap.add_argument("--pimm-src", default="/sdf/group/neutrino/omara/pimm-data/src")
    args = ap.parse_args()

    _add_repo_paths(args.helix_root, args.pimm_src)
    from helix.core import backend
    backend.set_backend("numpy")
    from helix.tpc.io import config_from_file, read_sensor_event, count_events
    from helix.tpc.config import DetectorConfig
    from helix.tpc.pipeline import canonical_plane_gid
    from helix.tpc.corpus import build_corpus
    from pimm_data.geometry import load_plane_registry
    from pimm_data.noise import generate_noise, digitize

    spec = None
    if not args.white:
        npz = np.load(args.npz, allow_pickle=True)
        spec = (npz["spectrum_freqs_hz"], npz["spectrum_shape"])
    reg = load_plane_registry(args.geom)
    base = config_from_file(args.shard)
    cfg = DetectorConfig(num_time_steps=base.num_time_steps,
                         plane_labels=base.plane_labels, pedestals=base.pedestals)
    print(f"n_time={cfg.num_time_steps} planes={len(cfg.plane_labels)} "
          f"wavelet={cfg.wavelet} L{cfg.dwt_level} removal={cfg.removal} "
          f"k{cfg.gate_kgate}/np{cfg.gate_npass} noise={'white' if args.white else 'colored'}")

    def plane_fn(ev):
        planes = read_sensor_event(args.shard, ev, cfg)
        seed = int.from_bytes(hashlib.blake2b(f"ev{ev}".encode(), digest_size=8).digest(),
                              "little") & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        noisy, clean = {}, {}
        for label, img in planes.items():
            gid = canonical_plane_gid(label)
            nw = img.shape[0]
            wl = np.asarray(reg.get(gid, {}).get("wire_lengths", []), np.float64)
            if wl.size != nw:
                wl = np.full(nw, 2.33, np.float64)
            ped = cfg.pedestals.get(label.split("_")[-1], 0)
            noise = generate_noise(img.shape, rng=rng, wire_lengths_m=wl,
                                   incoherent=True, coherent=True,
                                   series_spectrum=spec, group_size=cfg.group_size)
            noisy[gid] = digitize(img + noise, ped)
            clean[gid] = img.astype(np.float32)
        return noisy, clean

    n = count_events(args.shard)
    events = list(range(args.event_start, min(args.event_start + args.events, n)))
    t0 = time.perf_counter()
    noisy, clean, norm = build_corpus(events, plane_fn, cfg, args.out,
                                      dataset_name=args.dataset_name, run=args.run,
                                      file_index=args.file_index,
                                      cal_events=tuple(args.cal_events))
    dt = time.perf_counter() - t0
    print(f"built {len(noisy)} events in {dt:.1f}s ({dt/max(len(noisy),1):.1f}s/ev); "
          f"coeffs/event={[ce.n_coeff for ce in noisy]}")
    print(f"norm_sigma {None if norm is None else norm.shape}; wrote -> {args.out}")


if __name__ == "__main__":
    main()
