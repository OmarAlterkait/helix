#!/usr/bin/env python3
"""Fresh noise draws of the evaluation events, shared by scripts/eval_denoise.py
(--draws K) and scripts/eval_classical.py, so the model and the classical chain read
the SAME digitised waveforms.

Why: the stored corpus carries ONE noise realisation per event, and its detection
numbers fluctuate draw to draw by about +-0.01 (0.025 per plane), more than the
single-draw event bootstrap says (baselines/uboone_sp/dense/results/draws.md). The
recorded ``ident/noise_seed`` does not reproduce the stored realisation on today's
torch / GPU, so fresh draws are generated instead.

Draw k of an event:
  seed_k  = (ident/noise_seed + k * 0x9E3779B1) mod 2^32
  planes  = scripts/build_coeff_corpus.py's serial torch path (_plane_fn_torch):
            densify the clean sensor event, helix.tpc.dense_ops.add_intrinsic_noise
            (incoherent + coherent, torch Generators seeded from seed_k, the packaged
            measured spectrum), digitize -- run on a CUDA device
  coeffs  = helix.tpc.pipeline.process_plane(removal="gate") on each plane (numpy
            backend: identical kept set to the torch backend), event_coeff_event
Deterministic per (event, k) on the same GPU type: regeneration is bit-identical
(verified on A100; ``--verify`` re-checks a cache entry).

Cache (one npz per event and draw): ``<cache>/ev<NNN>_d<k>.npz`` with the coefficient
arrays the tokenizer needs (band, plane_gid, wire, tau, value), the digitised planes
(int16, ``g<gid>``), ``seed`` and ``sha_g<gid>`` (sha256 of each plane's int16 bytes).

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python scripts/eval_draws.py --events 260-291 --draws 5
"""
import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/eval_draws/coeffs"
SOURCE = "/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02"
GOLDEN = 0x9E3779B1


def draw_seed(seed, k):
    return (int(seed) + int(k) * GOLDEN) & 0xFFFFFFFF


def shard_identity(shard, event):
    """(pos, noise_seed, run, source_file) of ``event`` in a coefficient shard."""
    import h5py
    with h5py.File(shard, "r") as fh:
        ids = fh["ident"]["event"][:]
        pos = int(np.searchsorted(ids, event))
        if ids[pos] != event:
            raise KeyError(f"{shard}: no event {event}")
        dec = lambda x: x.decode() if isinstance(x, bytes) else str(x)
        return pos, int(fh["ident"]["noise_seed"][pos]), dec(fh["ident"]["run"][pos]), dec(fh["ident"]["source_file"][pos])


def _setup(sensor_file):
    from helix.paths import packaged
    from helix.tpc.config import DetectorConfig
    from helix.tpc.geometry import load_plane_registry
    from helix.tpc.io import config_from_file
    npz = np.load(str(packaged("noise_spectrum.npz")), allow_pickle=True)
    spec = (npz["spectrum_freqs_hz"], npz["spectrum_shape"])
    reg = load_plane_registry("cubic_wireplane_geometry.json")
    base = config_from_file(sensor_file)
    cfg = DetectorConfig(num_time_steps=base.num_time_steps, plane_labels=base.plane_labels, pedestals=base.pedestals)
    return reg, cfg, spec


def digitised_planes(sensor_file, event, seed, reg, cfg, spec):
    """build_coeff_corpus._plane_fn_torch: {gid: (nw, nt) float32 digitised ADC} (CUDA)."""
    import torch
    from helix.core import backend
    backend.set_backend("torch")
    from helix.tpc import dense_ops as _dops
    from helix.tpc.io import read_sensor_event_coo
    from helix.tpc.pipeline import canonical_plane_gid
    from pimm_data.dense_ops import densify as _densify
    spec_ = read_sensor_event_coo(sensor_file, event, cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        raise RuntimeError("fresh draws are defined on CUDA (the torch CPU RNG stream differs)")
    w_l, t_l, v_l, p_l, gmap = [], [], [], [], {}
    for label, (w, t, v, nw, nt) in spec_.items():
        gid = canonical_plane_gid(label)
        gmap[gid] = label
        w_l.append(torch.as_tensor(np.asarray(w), dtype=torch.long, device=dev))
        t_l.append(torch.as_tensor(np.asarray(t), dtype=torch.long, device=dev))
        v_l.append(torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev))
        p_l.append(torch.full((len(w),), gid, dtype=torch.long, device=dev))
    wire = torch.cat(w_l); time_ = torch.cat(t_l); val = torch.cat(v_l); pid = torch.cat(p_l)
    offset = torch.tensor([wire.numel()], dtype=torch.long, device=dev)
    grids = _densify(wire, time_, val, pid, offset, reg)
    grids = _dops.add_intrinsic_noise(grids, reg, seeds=[seed], incoherent=True, coherent=True,
                                      series_spectrum=spec, group_size=cfg.group_size)
    peds = {int(g): cfg.pedestals.get(gmap[int(g)].split("_")[-1], 0) for g in grids}
    grids = _dops.digitize(grids, peds, n_bits=12)
    return {int(g): np.asarray(x[0].detach().cpu().numpy(), np.float32) for g, x in grids.items()}


def _sha(x):
    return hashlib.sha256(np.ascontiguousarray(x.astype(np.int16)).tobytes()).hexdigest()


def build(shard, event, k, cache=CACHE, source=SOURCE, ev_tag=None):
    """Generate (or load) draw k of ``event``; returns the cache npz path."""
    tag = ev_tag if ev_tag is not None else f"{os.path.basename(shard)}_{event}"
    path = os.path.join(cache, f"ev{tag}_d{k}.npz") if not str(tag).isdigit() else os.path.join(cache, f"ev{int(tag):03d}_d{k}.npz")
    if os.path.exists(path):
        return path
    from helix.core import backend
    from helix.tpc.pipeline import process_plane, event_coeff_event
    pos, seed0, run, src = shard_identity(shard, event)
    sensor = os.path.join(source, "sensor", run, src)
    reg, cfg, spec = _setup(sensor)
    seed = draw_seed(seed0, k)
    planes = digitised_planes(sensor, event, seed, reg, cfg, spec)
    backend.set_backend("numpy")
    res = {g: process_plane(x, cfg, removal="gate", with_images=False) for g, x in planes.items()}
    ce = event_coeff_event(res, cfg, run="", source_file="", event=event, removal="gate")
    os.makedirs(cache, exist_ok=True)
    arr = {k_: np.asarray(getattr(ce, k_)) for k_ in ("band", "plane_gid", "wire", "tau", "value")}
    np.savez_compressed(path + ".tmp.npz", seed=seed, noise_seed=seed0, draw=k, shard=shard, event=event,
                        **arr, **{f"g{g}": x.astype(np.int16) for g, x in planes.items()},
                        **{f"sha_g{g}": _sha(x) for g, x in planes.items()})
    os.replace(path + ".tmp.npz", path)
    return path


def tokens(path, shard, ds):
    """(None, FM token dict, band_lengths) of a cached draw, tokenized with the shard's own config."""
    from helix.model.tokenize import assemble, to_fm
    z = np.load(path)
    c = ds._shard_cfg(shard)
    tok = assemble(z["band"], z["plane_gid"], z["wire"], z["tau"], z["value"], gids=c["gids"], n_wires=c["n_wires"],
                   band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=ds.cfg)
    return None, to_fm(tok), c["band_lengths"]


def planes(path, verify=True):
    """{gid: float32 digitised plane} of a cached draw (sha256-checked)."""
    z = np.load(path)
    out = {}
    for k in z.files:
        if k.startswith("g") and k[1:].isdigit():
            x = z[k]
            if verify and _sha(x) != str(z[f"sha_{k}"]):
                raise ValueError(f"{path}: plane {k} fails its sha256")
            out[int(k[1:])] = x.astype(np.float32)
    return out


def verify(path, source=SOURCE):
    """Regenerate a cached draw from its seed and check bit identity of every plane."""
    z = np.load(path)
    shard, event = str(z["shard"]), int(z["event"])
    pos, seed0, run, src = shard_identity(shard, event)
    sensor = os.path.join(source, "sensor", run, src)
    reg, cfg, spec = _setup(sensor)
    P = digitised_planes(sensor, event, int(z["seed"]), reg, cfg, spec)
    return {g: _sha(x) == str(z[f"sha_g{g}"]) for g, x in P.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth", default="/pscratch/sd/o/oalter/helix_work/resolution/truth_v2")
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--corpus-map", default="", help="OLD=NEW prefix substitution for shard paths (e.g. a /pscratch->CFS copy)")
    ap.add_argument("--verify", action="store_true", help="re-generate existing entries and check bit identity")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.events.split("-"))
    old, new = a.corpus_map.split("=") if a.corpus_map else ("", "")
    for ev in range(lo, hi + 1):
        z = np.load(os.path.join(a.truth, f"ev{ev:03d}.npz"), allow_pickle=True)
        shard = str(z["shard"]).replace(old, new) if old else str(z["shard"])
        for k in range(a.draws):
            p = build(shard, int(z["event"]), k, a.cache, ev_tag=ev)
            msg = ""
            if a.verify:
                msg = f" verify {verify(p)}"
            print(f"ev{ev:03d} d{k}: {p}{msg}", flush=True)


if __name__ == "__main__":
    main()
