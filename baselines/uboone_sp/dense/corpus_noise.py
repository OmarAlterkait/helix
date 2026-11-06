#!/usr/bin/env python3
"""Regenerate the EXACT noisy planes the corpus was built from (the noise
realisation the model saw), and verify by re-running the corpus DSP and comparing
with the stored coefficients.

scripts/build_coeff_corpus.py (serial mode, torch backend -- the coeff_tpc_r1
setting recorded in every shard's /config/noise_json) adds noise with
helix.tpc.dense_ops.add_intrinsic_noise(seeds=[ident/noise_seed]) on the GPU, then
digitizes. This module replays its ``_plane_fn_torch`` line for line.

    noisy_planes(sensor_file, event, seed, reg, cfg, noise_spec) -> {gid: (nw, nt) float32 ADC}

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/dense/corpus_noise.py --events 260-263
prints, per event, the agreement of re-built vs stored coefficients.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
PS_CORPUS, CFS_CORPUS = "/pscratch/sd/o/oalter/coeff_tpc_r1", "/global/cfs/cdirs/m5238/users/oalter/coeff_tpc_r1"
SOURCE = "/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02"


def setup(sensor_file):
    from helix.core import backend
    backend.set_backend("torch")
    from helix.paths import packaged
    from helix.tpc.config import DetectorConfig
    from helix.tpc.geometry import load_plane_registry
    from helix.tpc.io import config_from_file
    npz = np.load(str(packaged("noise_spectrum.npz")), allow_pickle=True)
    noise_spec = (npz["spectrum_freqs_hz"], npz["spectrum_shape"])
    reg = load_plane_registry("cubic_wireplane_geometry.json")
    base = config_from_file(sensor_file)
    cfg = DetectorConfig(num_time_steps=base.num_time_steps, plane_labels=base.plane_labels, pedestals=base.pedestals)
    return reg, cfg, noise_spec


def noisy_planes(sensor_file, event, seed, reg, cfg, noise_spec, with_clean=False):
    """build_coeff_corpus._plane_fn_torch, verbatim in effect."""
    import torch
    from helix.tpc import dense_ops as _dops
    from helix.tpc.io import read_sensor_event_coo
    from helix.tpc.pipeline import canonical_plane_gid
    from pimm_data.dense_ops import densify as _densify
    spec = read_sensor_event_coo(sensor_file, event, cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w_l, t_l, v_l, p_l, gmap = [], [], [], [], {}
    for label, (w, t, v, nw, nt) in spec.items():
        gid = canonical_plane_gid(label)
        gmap[gid] = label
        w_l.append(torch.as_tensor(np.asarray(w), dtype=torch.long, device=dev))
        t_l.append(torch.as_tensor(np.asarray(t), dtype=torch.long, device=dev))
        v_l.append(torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev))
        p_l.append(torch.full((len(w),), gid, dtype=torch.long, device=dev))
    wire = torch.cat(w_l); time_ = torch.cat(t_l); val = torch.cat(v_l); pid = torch.cat(p_l)
    offset = torch.tensor([wire.numel()], dtype=torch.long, device=dev)
    grids = _densify(wire, time_, val, pid, offset, reg)
    clean = {int(g): x[0].clone() for g, x in grids.items()}
    grids = _dops.add_intrinsic_noise(grids, reg, seeds=[seed], incoherent=True, coherent=True,
                                      series_spectrum=noise_spec, group_size=cfg.group_size)
    peds = {int(g): cfg.pedestals.get(gmap[int(g)].split("_")[-1], 0) for g in grids}
    grids = _dops.digitize(grids, peds, n_bits=12)
    out = {int(g): x[0] for g, x in grids.items()}
    if with_clean:
        return out, clean
    return out


def event_files(ev):
    import h5py
    z = np.load(os.path.join(TRUTH, f"ev{ev:03d}.npz"), allow_pickle=True)
    shard, event = str(z["shard"]).replace(PS_CORPUS, CFS_CORPUS), int(z["event"])
    with h5py.File(shard, "r") as fh:
        pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        seed = int(fh["ident"]["noise_seed"][pos])
        src = fh["ident"]["source_file"][pos]; src = src.decode() if isinstance(src, bytes) else str(src)
        run = fh["ident"]["run"][pos]; run = run.decode() if isinstance(run, bytes) else str(run)
    return shard, pos, event, seed, os.path.join(SOURCE, "sensor", run, src)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-261")
    a = ap.parse_args()
    from helix.core.coeff_io import read_coeff_event
    lo, hi = (int(x) for x in a.events.split("-"))
    for ev in range(lo, hi + 1):
        shard, pos, event, seed, sensor = event_files(ev)
        reg, cfg, spec = setup(sensor)
        from helix.tpc.pipeline import process_plane, event_coeff_event
        planes = noisy_planes(sensor, event, seed, reg, cfg, spec)
        res = {g: process_plane(img, cfg, removal="gate", with_images=False) for g, img in planes.items()}
        ce = event_coeff_event(res, cfg, run="", source_file="", event=event, removal="gate")
        st = read_coeff_event(shard, pos)
        key = lambda c: (np.asarray(c.plane_gid, np.int64) << 48) | (np.asarray(c.band, np.int64) << 40) \
            | (np.asarray(c.wire, np.int64) << 20) | np.asarray(c.tau, np.int64)
        ka, kb = key(ce), key(st)
        common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
        va, vb = np.asarray(ce.value)[ia], np.asarray(st.value)[ib]
        print(f"ev{ev}: rebuilt {len(ka)} stored {len(kb)} common {len(common)} "
              f"({len(common) / len(kb):.5f} of stored); value max|diff| {np.abs(va - vb).max():.3g}, "
              f"median|v| {np.median(np.abs(vb)):.3g}", flush=True)


if __name__ == "__main__":
    main()
