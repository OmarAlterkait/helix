"""HDF5 I/O for JAXTPC production sensor files and processed output.

Reads the CURRENT JAXTPC sensor schema (doraemon productions, e.g.
``wire_test_00_00_02``)::

    /config                      attrs: num_time_steps, n_volumes, readout_type,
                                 pedestal/noise/digitize provenance, ...
    /config/num_wires            (n_volumes, n_planes) int32 — per (volume, plane)
    /config/pedestals            (n_volumes, n_planes) int32
    event_NNN/volume_V/<P>/      P iterates in sorted order (wire: U, V, Y);
        delta_wire, delta_time   int16 delta-encoded sparse COO
        values                   uint16 digitized ADC (hdf5plugin-compressed)
        attrs: wire_start, time_start, pedestal, n_pixels

Plane labels are ``volume_{V}_{P}`` (matching pimm-data's convention). The
legacy flat schema (``event_N/<plane>/{wire,time,values}`` raw COO with
``n_wires``/``pedestal`` attrs) is kept as a fallback branch. Readout geometry
is read from the file itself — no external config needed; this module is the
first consumer of ``/config/num_wires`` (pimm-data instead takes caller-supplied
geometry). Wire readout only; pixel files (``delta_py``/``delta_pz``) are the
point-cloud path — load those via pimm-data.

``values`` datasets in production files are compressed with an HDF5 plugin
filter — ``hdf5plugin`` must be importable (best-effort imported below; a
missing plugin surfaces as ``OSError: can't open directory ... plugin`` on read).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import h5py

try:                       # registers HDF5 compression filters (production files)
    import hdf5plugin      # noqa: F401
except ImportError:        # synthetic/uncompressed files still work without it
    pass

from helix.tpc.config import DetectorConfig
from helix.core.wavelet import SparseResult


def _event_key(f: h5py.File, event_idx: int) -> str:
    """Writers use ``event_{:03d}`` (minimum width); legacy files unpadded."""
    for key in (f"event_{event_idx:03d}", f"event_{event_idx}"):
        if key in f:
            return key
    raise KeyError(f"event {event_idx} not found (tried event_{event_idx:03d})")


def _resolve_plane(evt: h5py.Group, plane_label: str):
    """``'volume_0_U'`` → the nested group (current schema); flat label → direct."""
    if plane_label in evt:
        return evt[plane_label]
    if plane_label.startswith("volume_"):
        vol, plane = plane_label.rsplit("_", 1)
        if vol in evt and plane in evt[vol]:
            return evt[vol][plane]
    raise KeyError(f"plane {plane_label!r} not found in {evt.name}")


def _plane_column(evt: h5py.Group, plane_label: str) -> tuple[int, int]:
    """(volume index, plane column) for ``/config`` (n_volumes, n_planes) tables.

    Columns follow the sorted plane-key order within a volume (wire: U, V, Y) —
    verified against production ``num_wires`` vs decoded wire ranges."""
    vol, plane = plane_label.rsplit("_", 1)
    v = int(vol.split("_")[1])
    keys = sorted(k for k in evt[vol].keys())
    return v, keys.index(plane)


def config_from_file(path: str | Path, **overrides) -> DetectorConfig:
    """Build a DetectorConfig by reading geometry from a sensor HDF5 file.

    Current schema: ``/config`` attrs + tables; plane labels discovered from
    the first event's ``volume_V/P`` nesting. Legacy flat schema: previous
    behavior (labels = event-level groups, pedestal from plane attrs).
    """
    with h5py.File(path, "r") as f:
        evt_keys = sorted(k for k in f.keys() if k.startswith("event_"))
        if not evt_keys:
            raise ValueError(f"No events found in {path}")
        evt = f[evt_keys[0]]

        if "config" in f:                              # ── current schema ──
            cfg = f["config"]
            if str(cfg.attrs.get("readout_type", "wire")) != "wire":
                raise ValueError(
                    "helix.tpc reads wire readout only; pixel files are the "
                    "point-cloud path — load them via pimm-data.")
            num_time_steps = int(cfg.attrs.get("num_time_steps", 4321))
            plane_labels, pedestals = [], {}
            ped_tab = cfg["pedestals"][:] if "pedestals" in cfg else None
            for vol_key in sorted(k for k in evt.keys() if k.startswith("volume_")):
                planes = sorted(evt[vol_key].keys())
                for col, p in enumerate(planes):
                    plane_labels.append(f"{vol_key}_{p}")
                    if ped_tab is not None:
                        v = int(vol_key.split("_")[1])
                        pedestals[p] = int(ped_tab[v, col])   # per plane TYPE
        else:                                          # ── legacy flat schema ──
            plane_labels = [k for k in evt.keys() if isinstance(evt[k], h5py.Group)]
            num_time_steps = int(evt.attrs.get("num_time_steps", 2701))
            pedestals = {}
            for label in plane_labels:
                pt = label.split("_")[-1] if "_" in label else label
                pedestals[pt] = int(evt[label].attrs.get("pedestal", 0))

    kwargs: dict[str, Any] = dict(plane_labels=tuple(plane_labels),
                                  num_time_steps=num_time_steps,
                                  pedestals=pedestals)
    kwargs.update(overrides)
    return DetectorConfig(**kwargs)


def read_sensor_plane(path, event_idx, plane_label, num_time_steps=None,
                      pedestal=0) -> np.ndarray:
    """Read one plane → (n_wires, n_ticks) float32 pedestal-subtracted image.

    Dispatches on the on-disk encoding: delta-encoded sparse COO (current
    schema; pedestal taken from the plane's own attr) vs raw COO (legacy;
    ``pedestal`` argument applies)."""
    with h5py.File(path, "r") as f:
        evt = f[_event_key(f, event_idx)]
        grp = _resolve_plane(evt, plane_label)

        if "delta_wire" in grp:                        # ── current schema ──
            wire = np.cumsum(grp["delta_wire"][:], dtype=np.int32)
            wire += int(grp.attrs["wire_start"])
            time = np.cumsum(grp["delta_time"][:], dtype=np.int32)
            time += int(grp.attrs["time_start"])
            raw = grp["values"][:]
            values = raw.astype(np.float32)
            ped = int(grp.attrs.get("pedestal", pedestal))
            if raw.dtype == np.uint16:                 # digitized → subtract pedestal
                values -= ped
            if num_time_steps is None:
                num_time_steps = int(f["config"].attrs["num_time_steps"]) \
                    if "config" in f else int(time.max(initial=0)) + 1
            n_wires = None
            if "config" in f and "num_wires" in f["config"]:
                v, col = _plane_column(evt, plane_label)
                n_wires = int(f["config"]["num_wires"][v, col])
            if n_wires is None or n_wires <= int(wire.max(initial=0)):
                n_wires = int(wire.max(initial=0)) + 1
            image = np.zeros((n_wires, num_time_steps), dtype=np.float32)
            image[wire, time] = values                 # pedestal already subtracted
            return image

        # ── legacy flat schema ──
        wire = grp["wire"][:]
        time = grp["time"][:]
        values = grp["values"][:].astype(np.float32)
        n_wires = int(grp.attrs.get("n_wires", wire.max(initial=0) + 1))
        if num_time_steps is None:
            num_time_steps = 2701
    image = np.full((n_wires, num_time_steps), float(pedestal), dtype=np.float32)
    image[wire, time] = values
    image -= pedestal
    return image


def read_sensor_event(path, event_idx, config: DetectorConfig) -> dict[str, np.ndarray]:
    planes = {}
    with h5py.File(path, "r") as f:
        evt = f[_event_key(f, event_idx)]
        available = set()
        for k in evt.keys():
            if k.startswith("volume_") and isinstance(evt[k], h5py.Group):
                available.update(f"{k}_{p}" for p in evt[k].keys())
            elif isinstance(evt[k], h5py.Group):
                available.add(k)
    for label in config.plane_labels:
        if label in available:
            pt = label.split("_")[-1] if "_" in label else label
            ped = config.pedestals.get(pt, 0)
            planes[label] = read_sensor_plane(path, event_idx, label,
                                              config.num_time_steps, ped)
    return planes


def count_events(path) -> int:
    with h5py.File(path, "r") as f:
        return sum(1 for k in f.keys() if k.startswith("event_"))


def write_processed(path, event_idx, planes: dict[str, SparseResult], config: DetectorConfig) -> None:
    """Write processed sparse results to HDF5 (numpy list-of-bands coeffs)."""
    with h5py.File(path, "a") as f:
        evt = f.require_group(f"event_{event_idx:03d}")
        evt.attrs["wavelet"] = config.wavelet
        evt.attrs["dwt_level"] = config.dwt_level
        evt.attrs["threshold_kappa"] = config.threshold_kappa
        for label, result in planes.items():
            grp = evt.require_group(label)
            grp.attrs["n_kept"] = result.n_kept
            grp.attrs["n_total"] = result.n_total
            grp.attrs["sparsity"] = result.sparsity
            grp.attrs["sigma_per_band"] = np.asarray(result.sigma_per_band)
            coeffs = result.coeffs
            if not isinstance(coeffs, list):  # flat (jax) — store densely
                arr = np.asarray(coeffs)
                if "coeffs_flat" in grp:
                    del grp["coeffs_flat"]
                grp.create_dataset("coeffs_flat", data=arr.astype(np.float32))
                continue
            for i, c in enumerate(coeffs):
                c = np.asarray(c)
                band_name = "cA" if i == 0 else f"cD_{len(coeffs) - i}"
                if band_name in grp:
                    del grp[band_name]
                nz = np.nonzero(c)
                if len(nz[0]) > 0:
                    bg = grp.require_group(band_name)
                    bg["wire"] = nz[0].astype(np.int32)
                    bg["coeff_idx"] = nz[1].astype(np.int32)
                    bg["values"] = c[nz].astype(np.float32)
                    bg.attrs["shape"] = c.shape
