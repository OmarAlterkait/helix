"""Profile the production GPU load path across batch sizes.

Uses the *production* code, nothing bespoke:
  pimm_data.JAXTPCDataset            sparse clean extraction (CPU, h5)
  pimm_data.batch_transforms         BatchDensify -> BatchAddIntrinsicNoise -> BatchDigitize  (on device)
  helix.core.sparsify (torch)        production wavelet (coif3 L4) -> POST-THRESHOLD coeffs

Per-stage timing (median over reps) + peak GPU memory, for batch sizes 1..32.
Breaks noise into coherent (numpy oracle + H2D copy — NOT on GPU) vs incoherent
(torch, on GPU) so the CPU bottleneck is visible.

    python research/profile_load.py --batch-sizes 1 2 4 8 16 32 --reps 5
"""
from __future__ import annotations

import argparse
import sys
import time
from statistics import median

import numpy as np


DATA_ROOT = "/sdf/data/neutrino/omara/JAXTPC_Wire/test_00_00_02"
SPLIT = "run_0027575766"
DATASET_NAME = "sim_wire"
GEOM_JSON = ("/sdf/group/neutrino/omara/particle-imaging-models/"
             "libs/pimm-data/src/pimm_data/data/cubic_wireplane_geometry.json")
WAVELET, LEVEL, MODE = "coif3", 4, "periodization"


def load_geom_canonical(volume):
    """Canonical-id geometry registry (the dense path's `geom`), for one volume."""
    import json
    from pimm_data.jaxtpc import canonical_plane_id
    d = json.load(open(GEOM_JSON))
    nts = int(d["num_time_steps"])
    geom = {}
    for label, e in d["planes"].items():
        if int(label.split("_")[1]) != volume:
            continue
        wl = e.get("wire_lengths_m")
        geom[canonical_plane_id(label)] = {
            "n_wires": int(e["n_wires"]), "n_ticks": nts,
            "pedestal": int(e["pedestal"]),
            "wire_lengths": (np.asarray(wl, np.float32) if wl is not None
                             else np.full(int(e["n_wires"]), 2.33, np.float32)),
        }
    return geom, nts


def build_batch(ds, ev0, B, volume):
    """Collate B consecutive events into the flat dense-path batch dict (CPU torch)."""
    import torch
    from pimm_data.jaxtpc import canonical_plane_id
    keep = {canonical_plane_id(f"volume_{volume}_{t}") for t in ("U", "V", "Y")}
    wires, times, vals, gids, counts, names = [], [], [], [], [], []
    for i in range(B):
        s = ds.get_data(ev0 + i)["sensor"]
        m = np.isin(s["plane_gid"], list(keep))
        wires.append(s["wire"][m]); times.append(s["time"][m])
        vals.append(s["value"][m]); gids.append(s["plane_gid"][m])
        counts.append(int(m.sum())); names.append(s["name"])
    offset = torch.tensor(np.cumsum(counts), dtype=torch.int64)
    return {
        "wire": torch.as_tensor(np.concatenate(wires), dtype=torch.int32),
        "time": torch.as_tensor(np.concatenate(times), dtype=torch.int32),
        "value": torch.as_tensor(np.concatenate(vals), dtype=torch.float32),
        "plane_gid": torch.as_tensor(np.concatenate(gids), dtype=torch.int32),
        "offset": offset, "name": names,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--volume", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=1.0)
    args = ap.parse_args()

    import torch
    from pimm_data import JAXTPCDataset
    from pimm_data.batch_transforms import (
        move_to_device, _batch_seeds, BatchDensify, BatchAddIntrinsicNoise, BatchDigitize)
    from helix.core import backend
    from helix.core.wavelet import sparsify
    from helix.tpc.config import DetectorConfig
    backend.set_backend("torch")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    th = DetectorConfig(threshold_kappa=args.kappa).threshold_spec()
    print(f"device: {dev} | wavelet {WAVELET} L{LEVEL} | threshold {th}")

    geom, nts = load_geom_canonical(args.volume)
    print(f"geom (volume {args.volume}): "
          f"{[(g, e['n_wires']) for g, e in geom.items()]}, n_ticks={nts}")
    pad = (-nts) % (2 ** LEVEL)

    ds = JAXTPCDataset(data_root=DATA_ROOT, split=SPLIT,
                       modalities=("sensor",), dataset_name=DATASET_NAME)
    densify = BatchDensify(geom)
    addnoise = BatchAddIntrinsicNoise(geom, coherent=True, incoherent=True)
    coh_only = BatchAddIntrinsicNoise(geom, coherent=True, incoherent=False)
    incoh_only = BatchAddIntrinsicNoise(geom, coherent=False, incoherent=True)
    digitize = BatchDigitize(geom)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    def sparsify_planes(dense):
        """Production POST-THRESHOLD sparsify per plane (pad T to 2^L)."""
        kept = total = 0
        for gid, g in dense.items():
            x = torch.nn.functional.pad(g, (0, pad))
            r = sparsify(x, wavelet=WAVELET, level=LEVEL, mode=MODE, threshold=th)
            kept += r.n_kept; total += r.n_total
        return kept, total

    print(f"\n{'B':>3} {'read+collate':>13} {'H2D':>7} {'densify':>9} "
          f"{'coherent':>9} {'incoher':>9} {'digitize':>9} {'dwt+thr':>9} "
          f"{'TOTAL ms':>9} {'ms/evt':>7} {'peakMB':>8} {'keep%':>6}")

    ev = 0
    for B in args.batch_sizes:
        # time read+collate (CPU) separately — it varies with disk cache
        t_rc = []
        for _ in range(args.reps):
            t0 = time.perf_counter(); build_batch(ds, ev, B, args.volume)
            t_rc.append((time.perf_counter() - t0) * 1e3); ev += B
        cpu_batch = build_batch(ds, ev, B, args.volume); ev += B

        def stage(fn, reps=args.reps):
            ts = []
            for _ in range(reps):
                b = move_to_device({k: (v.clone() if torch.is_tensor(v) else v)
                                    for k, v in cpu_batch.items()}, dev)
                seeds = _batch_seeds(b, 0, 0, 0, B)
                sync(); t0 = time.perf_counter()
                fn(b, seeds); sync()
                ts.append((time.perf_counter() - t0) * 1e3)
            return median(ts)

        # build cumulative timings by composing stages on fresh batches
        def f_h2d(b, seeds): pass
        def f_dens(b, seeds): densify(b, seeds=seeds)
        def f_coh(b, seeds): densify(b, seeds=seeds); coh_only(b, seeds=seeds)
        def f_inc(b, seeds): densify(b, seeds=seeds); incoh_only(b, seeds=seeds)
        def f_dig(b, seeds): densify(b, seeds=seeds); addnoise(b, seeds=seeds); digitize(b, seeds=seeds)

        t_h2d = stage(f_h2d)
        t_dens = stage(f_dens) - t_h2d
        t_coh = stage(f_coh) - t_dens - t_h2d
        t_inc = stage(f_inc) - t_dens - t_h2d

        # full path incl. sparsify + peak mem
        torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
        tot_ts, keep = [], 0.0
        for _ in range(args.reps):
            b = move_to_device({k: (v.clone() if torch.is_tensor(v) else v)
                                for k, v in cpu_batch.items()}, dev)
            seeds = _batch_seeds(b, 0, 0, 0, B)
            sync(); t0 = time.perf_counter()
            densify(b, seeds=seeds); addnoise(b, seeds=seeds); digitize(b, seeds=seeds)
            kp, tot = sparsify_planes(b["sensor_dense"]); sync()
            tot_ts.append((time.perf_counter() - t0) * 1e3); keep = 100.0 * kp / tot
        t_full = median(tot_ts)
        t_dig = stage(f_dig) - t_dens - t_coh - t_inc - t_h2d  # digitize-only slice
        t_dwt = t_full - (t_dens + t_coh + t_inc + t_dig + t_h2d)
        peak = (torch.cuda.max_memory_allocated() / 1e6) if dev == "cuda" else 0.0
        rc = median(t_rc)
        print(f"{B:>3} {rc:>13.1f} {t_h2d:>7.2f} {t_dens:>9.2f} "
              f"{t_coh:>9.2f} {t_inc:>9.2f} {t_dig:>9.2f} {t_dwt:>9.2f} "
              f"{t_full:>9.1f} {t_full/B:>7.1f} {peak:>8.0f} {keep:>6.2f}")


if __name__ == "__main__":
    for p in ("/sdf/group/neutrino/omara/helix/.pylibs",
              "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src",
              "/sdf/group/neutrino/omara/helix"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import hdf5plugin  # noqa: F401
    main()
