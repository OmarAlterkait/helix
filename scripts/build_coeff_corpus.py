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
import os
import sys
import time

import numpy as np
from pathlib import Path


def _add_repo_paths(helix_root, pimm_src):
    for p in (helix_root, pimm_src):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def _loader_stream(args, cfg, reg, noise_spec):
    """Mode A: pimm-data DataLoader workers + the torch dense tail -> dlpack -> jax.

    The head (`Collect`) runs per-event in worker PROCESSES on sparse COO, so the
    HDF5 decode overlaps GPU work (measured DataLoader wait: 2.5 ms/event). The
    tail densifies/noises/digitizes on-device with pimm-data's tested torch ops;
    the grids cross to jax zero-copy via dlpack.

    Densify runs BEFORE AddNoise, so the clean target is simply the densified
    image captured before noise is added — no second pass over the event.
    """
    import jax
    import torch
    from torch.utils.data import DataLoader
    from pimm_data import JAXTPCDataset
    from pimm_data.collate import collate_fn
    from pimm_data.transform import Compose

    root = args.data_root or str(Path(args.shard).parents[2])
    split = args.split or Path(args.shard).parent.name
    head = [dict(type="Collect", parts={"sensor": dict(
        keys=("wire", "time", "value", "plane_gid"))})]
    dense = Compose([dict(type="ToDevice", device="cuda"),
                     dict(type="Densify", geom=reg, modality="sensor")])
    noise = Compose([dict(type="AddNoise", geom=reg, modality="sensor", coherent=True,
                          incoherent=True, series_spectrum=noise_spec,
                          wire_lengths_m=2.33),
                     dict(type="Digitize", geom=reg, modality="sensor", n_bits=12)])
    ds = JAXTPCDataset(data_root=root, split=split, dataset_name=args.dataset_name,
                       modalities=("sensor",), transform=head)
    # spawn, NOT fork: DataLoader workers fork while jax is initialised in the
    # parent, and jax warns that fork + its threads "will likely lead to a
    # deadlock". spawn re-imports cleanly in each worker.
    dl = DataLoader(ds, batch_size=1, num_workers=args.workers, collate_fn=collate_fn,
                    persistent_workers=args.workers > 0,
                    multiprocessing_context="spawn" if args.workers > 0 else None,
                    prefetch_factor=2 if args.workers > 0 else None)
    n_want = args.events
    for i, b in enumerate(dl):
        if i >= n_want:
            break
        b = dense(b)
        grids = b["sensor_dense"]
        clean = {int(g): jax.dlpack.from_dlpack(t[0].clone().contiguous())
                 for g, t in grids.items()}            # capture BEFORE noise
        b = noise(b)
        noisy = {int(g): jax.dlpack.from_dlpack(t[0].contiguous())
                 for g, t in b["sensor_dense"].items()}
        yield i, noisy, clean


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
    ap.add_argument("--cal-events", type=int, nargs="*", default=list(range(16)))
    ap.add_argument("--white", action="store_true", help="use white incoherent noise (old bug)")
    ap.add_argument("--backend", choices=["numpy", "jax"], default="numpy",
                    help="jax runs noise + DWT + gate + threshold on GPU (~30x)")
    ap.add_argument("--mode", choices=["serial", "loader"], default="serial",
                    help="loader: pimm-data DataLoader workers + the torch dense tail "
                         "(ToDevice/Densify/AddNoise/Digitize) handed to jax via dlpack. "
                         "Fastest measured path (181 vs 536 ms/event); read overlaps the GPU.")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers (loader mode)")
    ap.add_argument("--split", default=None, help="run dir under data_root (loader mode)")
    ap.add_argument("--data-root", default=None, help="dataset root (loader mode)")
    ap.add_argument("--helix-root", default="/sdf/group/neutrino/omara/helix-consolidate")
    ap.add_argument("--pimm-src", default="/sdf/group/neutrino/omara/pimm-data/src")
    args = ap.parse_args()

    if args.backend == "jax":                    # keep XLA from grabbing the whole card
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _add_repo_paths(args.helix_root, args.pimm_src)
    from helix.core import backend
    backend.set_backend(args.backend)
    from helix.tpc.io import (config_from_file, read_sensor_event,
                              read_sensor_event_coo, count_events)
    from helix.tpc.config import DetectorConfig
    from helix.tpc.pipeline import canonical_plane_gid
    from helix.tpc.corpus import build_corpus
    from pimm_data.geometry import load_plane_registry
    from pimm_data.noise import generate_noise, digitize

    noise_spec = None
    if not args.white:
        npz = np.load(args.npz, allow_pickle=True)
        noise_spec = (npz["spectrum_freqs_hz"], npz["spectrum_shape"])
    reg = load_plane_registry(args.geom)
    base = config_from_file(args.shard)
    cfg = DetectorConfig(num_time_steps=base.num_time_steps,
                         plane_labels=base.plane_labels, pedestals=base.pedestals)
    print(f"n_time={cfg.num_time_steps} planes={len(cfg.plane_labels)} "
          f"wavelet={cfg.wavelet} L{cfg.dwt_level} removal={cfg.removal} "
          f"k{cfg.gate_kgate}/np{cfg.gate_npass} noise={'white' if args.white else 'colored'}")

    use_jax = args.backend == "jax"
    if use_jax:
        import jax
        import jax.numpy as jnp
        from pimm_data.noise_jax import generate_noise_jax
        from pimm_data.dense_ops_jax import densify_plane_jax

    def _seed(ev):
        return int.from_bytes(hashlib.blake2b(f"ev{ev}".encode(), digest_size=8).digest(),
                              "little") & 0xFFFFFFFF

    def _digitize_jax(x, ped, n_bits=12):
        """On-device twin of pimm_data.noise.digitize (round -> clip -> unpedestal)."""
        adc_max = (1 << n_bits) - 1
        return jnp.clip(jnp.round(x + ped), 0, adc_max) - ped

    def plane_fn(ev):
        # jax path reads SPARSE COO and densifies ON DEVICE — building the dense
        # image on the CPU and copying it across was pure overhead.
        planes = (read_sensor_event_coo(args.shard, ev, cfg) if use_jax
                  else read_sensor_event(args.shard, ev, cfg))
        seed = _seed(ev)
        rng = None if use_jax else np.random.default_rng(seed)
        key = jax.random.PRNGKey(seed) if use_jax else None
        noisy, clean = {}, {}
        for i, (label, spec) in enumerate(planes.items()):
            gid = canonical_plane_gid(label)
            if use_jax:
                w, t, v, nw, nt = spec
                img = densify_plane_jax(w, t, v, nw, nt)      # on GPU
            else:
                img = spec
                nw = img.shape[0]
            wl = np.asarray(reg.get(gid, {}).get("wire_lengths", []), np.float64)
            if wl.size != nw:
                wl = np.full(nw, 2.33, np.float64)
            ped = cfg.pedestals.get(label.split("_")[-1], 0)
            if use_jax:                                   # noise + digitize on GPU
                k = jax.random.fold_in(key, i)            # per-plane substream
                noise = generate_noise_jax(
                    k, img.shape, wire_lengths_m=wl, incoherent=True, coherent=True,
                    series_spectrum=noise_spec, group_size=cfg.group_size)
                noisy[gid] = _digitize_jax(img + noise, ped)
                clean[gid] = img
            else:
                noise = generate_noise(img.shape, rng=rng, wire_lengths_m=wl,
                                       incoherent=True, coherent=True,
                                       series_spectrum=noise_spec, group_size=cfg.group_size)
                noisy[gid] = digitize(img + noise, ped)
                clean[gid] = img.astype(np.float32)
        return noisy, clean

    t0 = time.perf_counter()
    if args.mode == "loader":
        from helix.tpc.corpus import build_corpus_stream
        stream = _loader_stream(args, cfg, reg, noise_spec)
        noisy, clean, norm = build_corpus_stream(
            stream, cfg, args.out, dataset_name=args.dataset_name, run=args.run,
            file_index=args.file_index, cal_events=tuple(args.cal_events))
    else:
        n = count_events(args.shard)
        events = list(range(args.event_start, min(args.event_start + args.events, n)))
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
