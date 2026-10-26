"""HDF5 readers for JAXTPC production sensor files.

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

Plane labels are ``volume_{V}_{P}`` (matching pimm-data's convention). Readout geometry
is read from the file itself — no external config needed; this module is the
first consumer of ``/config/num_wires`` (pimm-data instead takes caller-supplied
geometry). Wire readout only; pixel files (``delta_py``/``delta_pz``) are the
point-cloud path — load those via pimm-data.

``values`` datasets in production files are compressed with an HDF5 plugin
filter — ``hdf5plugin`` must be importable (best-effort imported below; a
missing plugin surfaces as ``OSError: can't open directory ... plugin`` on read).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import h5py

try:                       # registers HDF5 compression filters (production files)
    import hdf5plugin      # noqa: F401
except ImportError:        # synthetic/uncompressed files still work without it
    pass

from helix.tpc.readout import canonical_plane_gid


def _event_key(f: h5py.File, event_idx: int) -> str:
    """Writers use ``event_{:03d}`` (minimum width)."""
    key = f"event_{event_idx:03d}"
    if key not in f:
        raise KeyError(f"event {event_idx} not found ({key})")
    return key


def _resolve_plane(evt: h5py.Group, plane_label: str):
    """``'volume_0_U'`` → the nested group."""
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


@dataclass(frozen=True, eq=False)
class SensorLayout:
    """A sensor file's readout layout: what the readers and the TPC chain need.

    ``plane_labels``  ``volume_{V}_{P}`` (current schema) or flat labels (legacy);
    ``num_time_steps`` ticks per plane;
    ``pedestals``      ``{plane type: ADC}`` (U / V / Y);
    ``num_wires``      (n_volumes, n_planes) int array: ``/config/num_wires``, or
                       derived from the first event's plane shapes when the file has
                       no such table (``None`` when the labels do not form a
                       volume x plane grid). It is the chain's layout readout
                       (:func:`helix.tpc.chain.sim_config`).
    """
    plane_labels: tuple
    num_time_steps: int
    pedestals: dict
    num_wires: np.ndarray | None = None

    def pedestal(self, plane_label) -> int:
        pt = plane_label.split("_")[-1] if "_" in plane_label else plane_label
        return int(self.pedestals.get(pt, 0))

    def plane_gid(self, plane_label) -> int:
        """Canonical plane id ``volume * 3 + {U: 0, V: 1, Y: 2}`` (pimm-data's;
        :func:`helix.tpc.readout.canonical_plane_gid`)."""
        vol, plane = _split_label(plane_label)
        return canonical_plane_gid(vol, _PLANE_INDEX[plane])


_PLANE_INDEX = {"U": 0, "V": 1, "Y": 2}


def _split_label(plane_label):
    """``'volume_1_V'`` -> (1, 'V')."""
    vol, plane = plane_label.rsplit("_", 1)
    return int(vol.split("_")[1]), plane


def _num_wires_grid(labels, n_wires):
    """(n_volumes, n_planes) from ``{label: n_wires}``, or None when the labels do
    not form a full volume x plane grid."""
    by_vol = {}
    for lab in labels:
        v, p = _split_label(lab)
        by_vol.setdefault(v, {})[p] = int(n_wires[lab])
    vols = sorted(by_vol)
    cols = [sorted(by_vol[v]) for v in vols]
    if not vols or vols != list(range(len(vols))) or any(c != cols[0] for c in cols):
        return None
    return np.array([[by_vol[v][p] for p in cols[0]] for v in vols], np.int32)


def sensor_layout(path: str | Path) -> SensorLayout:
    """Read a sensor HDF5 file's :class:`SensorLayout`.

    ``/config`` attrs + tables; plane labels discovered from the first event's
    ``volume_V/P`` nesting. ``num_wires`` comes from ``/config/num_wires`` when
    present, otherwise from the first event's planes.
    """
    num_wires = None
    with h5py.File(path, "r") as f:
        evt_keys = sorted(k for k in f.keys() if k.startswith("event_"))
        if not evt_keys:
            raise ValueError(f"No events found in {path}")
        evt = f[evt_keys[0]]

        if "config" not in f:
            raise ValueError(f"{path}: no /config, so not a JAXTPC production sensor file")
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
        if "num_wires" in cfg:
            num_wires = np.asarray(cfg["num_wires"][:], np.int32)
        first = int(evt_keys[0].split("_", 1)[1])

    if num_wires is None:                              # derive from the first event
        nw = {lab: read_sensor_plane_coo(path, first, lab, num_time_steps,
                                         pedestals.get(lab.split("_")[-1], 0))[3]
              for lab in plane_labels}
        num_wires = _num_wires_grid(plane_labels, nw)
    return SensorLayout(plane_labels=tuple(plane_labels), num_time_steps=num_time_steps,
                        pedestals=pedestals, num_wires=num_wires)



def _empty_plane_shape(f, evt, plane_label, num_time_steps):
    """``(n_wires, n_ticks)`` for a plane group that holds no hits.

    An event depositing no charge in one TPC volume is written as three EMPTY
    plane groups — no datasets, no attrs. That is ordinary physics (the
    interaction happened wholly in the other volume), not a damaged file, so it
    must decode as zero hits rather than raise. The shape cannot come from the
    data, so it comes from the file's config.
    """
    n_wires = 1
    if "config" in f and "num_wires" in f["config"]:
        v, col = _plane_column(evt, plane_label)
        n_wires = int(f["config"]["num_wires"][v, col])
    if num_time_steps is None:
        num_time_steps = int(f["config"].attrs["num_time_steps"]) \
            if "config" in f else 1
    return n_wires, int(num_time_steps)


def read_sensor_plane_coo(path, event_idx, plane_label, num_time_steps=None,
                          pedestal=0):
    """Read one plane as SPARSE COO → ``(wire, time, value, n_wires, n_ticks)``.

    The decode half of :func:`read_sensor_plane` without the dense scatter, so a
    GPU pipeline can densify on-device (``pimm_data.dense_ops``) instead of
    building the dense image on the CPU and copying it across. ``value`` is
    pedestal-subtracted, matching the dense reader.
    """
    with h5py.File(path, "r") as f:
        evt = f[_event_key(f, event_idx)]
        grp = _resolve_plane(evt, plane_label)

        if len(grp.keys()) == 0:                       # ── empty: no charge ──
            nw, nt = _empty_plane_shape(f, evt, plane_label, num_time_steps)
            z = np.zeros(0, dtype=np.int32)
            return z, z.copy(), np.zeros(0, dtype=np.float32), nw, nt

        wire = np.cumsum(grp["delta_wire"][:], dtype=np.int32)
        wire += int(grp.attrs["wire_start"])
        time = np.cumsum(grp["delta_time"][:], dtype=np.int32)
        time += int(grp.attrs["time_start"])
        raw = grp["values"][:]
        values = raw.astype(np.float32)
        ped = int(grp.attrs.get("pedestal", pedestal))
        if raw.dtype == np.uint16:                     # digitized → subtract pedestal
            values -= ped
        if num_time_steps is None:
            num_time_steps = int(f["config"].attrs["num_time_steps"])
        n_wires = None
        if "num_wires" in f["config"]:
            v, col = _plane_column(evt, plane_label)
            n_wires = int(f["config"]["num_wires"][v, col])
        if n_wires is None or n_wires <= int(wire.max(initial=0)):
            n_wires = int(wire.max(initial=0)) + 1
        return wire, time, values, n_wires, int(num_time_steps)


def read_sensor_event_coo(path, event_idx, layout: SensorLayout):
    """All planes of one event as sparse COO → ``{label: (wire,time,value,nw,nt)}``."""
    out = {}
    with h5py.File(path, "r") as f:
        evt = f[_event_key(f, event_idx)]
        available = set()
        for k in evt.keys():
            if k.startswith("volume_") and isinstance(evt[k], h5py.Group):
                available.update(f"{k}_{p}" for p in evt[k].keys())
    for label in layout.plane_labels:
        if label in available:
            out[label] = read_sensor_plane_coo(path, event_idx, label,
                                               layout.num_time_steps,
                                               layout.pedestal(label))
    return out


def read_sensor_plane(path, event_idx, plane_label, num_time_steps=None,
                      pedestal=0) -> np.ndarray:
    """Read one plane → (n_wires, n_ticks) float32 pedestal-subtracted image
    (:func:`read_sensor_plane_coo`, scattered)."""
    wire, time, values, n_wires, n_ticks = read_sensor_plane_coo(path, event_idx, plane_label,
                                                                 num_time_steps, pedestal)
    image = np.zeros((n_wires, n_ticks), dtype=np.float32)
    image[wire, time] = values
    return image


def read_sensor_event(path, event_idx, layout: SensorLayout) -> dict[str, np.ndarray]:
    planes = {}
    with h5py.File(path, "r") as f:
        evt = f[_event_key(f, event_idx)]
        available = set()
        for k in evt.keys():
            if k.startswith("volume_") and isinstance(evt[k], h5py.Group):
                available.update(f"{k}_{p}" for p in evt[k].keys())
    for label in layout.plane_labels:
        if label in available:
            planes[label] = read_sensor_plane(path, event_idx, label,
                                              layout.num_time_steps,
                                              layout.pedestal(label))
    return planes


def count_events(path) -> int:
    with h5py.File(path, "r") as f:
        return sum(1 for k in f.keys() if k.startswith("event_"))


def list_events(path) -> tuple[int, ...]:
    """The event ids actually present, ascending.

    Event ids are NOT guaranteed to be ``0..count_events()-1``: production files
    can be missing an id in the middle (``sim_wire_sensor_0065.h5`` of
    run_0027575715 has 199 events spanning 0..199, with 167 absent, while its own
    ``config.n_events`` attr still claims 200). Anything that derives event
    indices from a COUNT therefore both requests an id that does not exist and
    silently skips a real one at the tail — enumerate with this instead.
    """
    with h5py.File(path, "r") as f:
        return tuple(sorted(int(k.split("_", 1)[1]) for k in f.keys()
                            if k.startswith("event_")))
