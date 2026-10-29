"""Readers for a JAXTPC wire production campaign (format 2), e.g. ``doraemon_2026_1M_lartpc``.

A campaign's wire stage writes one task per ``merge`` consecutive generator jobs, each task one
sensor, one step and one hits file::

    <root>/<group>/<task>/{sensor,step,hits}/sim_wire_{kind}_<task>.h5
        root   the stage's data directory (``.../<campaign>/data/jaxtpc_wire``)
        group  ``j00X000``, the tasks of 1000 consecutive jobs
        task   ``jaxtpc_wire_jAAAAAA-BBBBBB`` (jobs AAAAAA..BBBBBB)

Every file::

    /config         attrs: format_version (2), production_version (the campaign), git_commit,
                    num_time_steps, readout_type, ... ; tables num_wires, pedestals (n_volumes, 3)
    /dprod/events   event_id, job_id, job_event (n_entries,): entry i's identity, in entry order
    entry_NNNNNN/   one event; attrs event_id (unique in the campaign), job_id, job_event, ...

and a sensor entry holds per ``volume_V/<P>`` (P = U, V, Y) the digitized readout as
delta-encoded sparse COO: ``delta_wire``, ``delta_time`` (int16), ``values`` (uint16 raw ADC =
pedestal + signal) with attrs ``wire_start``, ``time_start``, ``pedestal``, ``n_pixels``. A volume
with nothing deposited is written as plane groups with no datasets and no attrs: zero hits.

Plane labels are ``volume_{V}_{P}``; the plane id is ``V * 3 + {U: 0, V: 1, Y: 2}``. Everything
here reads the file itself: no external configuration. The production files are compressed with
an HDF5 plugin filter (blosc-lz4), so ``hdf5plugin`` must be importable to read values.

This is JAXTPC's own ``production/entries.py`` + ``load.py`` contract (``find_entry``,
``load_event_sensor``) at its format 2; format 1 (``event_NNN``) files are not read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

try:                       # registers HDF5 compression filters (production files)
    import hdf5plugin      # noqa: F401
except ImportError:        # synthetic/uncompressed files still work without it
    pass

from helix.tpc.readout import canonical_plane_gid

FORMAT_VERSION = 2
PLANES = ("U", "V", "Y")
KINDS = ("sensor", "step", "hits")


def entry_key(pos: int) -> str:
    """The group of entry ``pos`` (its position in the file)."""
    return f"entry_{int(pos):06d}"


def require_format(f: h5py.File):
    """Refuse a file that is not a format-2 wire production file of a campaign."""
    cfg = f.get("config")
    version = None if cfg is None else cfg.attrs.get("format_version")
    if version is None or int(version) != FORMAT_VERSION:
        raise ValueError(f"{f.filename}: not a JAXTPC format-{FORMAT_VERSION} production file "
                         f"(format_version {version})")
    if str(cfg.attrs.get("readout_type", "wire")) != "wire":
        raise ValueError(f"{f.filename}: helix.tpc reads wire readout only; pixel files are the "
                         "point-cloud path -- load them via pimm-data.")
    if "dprod" not in f:
        raise ValueError(f"{f.filename}: no /dprod/events, so no campaign event identity")


def event_ids(f: h5py.File) -> np.ndarray:
    """The campaign event id of every entry, in entry order (``/dprod/events/event_id``)."""
    return f["dprod"]["events"]["event_id"][:].astype(np.int64)


def entry(f: h5py.File, pos: int, event_id: int) -> h5py.Group:
    """Entry ``pos``, checked to be campaign event ``event_id``."""
    g = f[entry_key(pos)]
    if int(g.attrs["event_id"]) != int(event_id):
        raise ValueError(f"{f.filename}: {entry_key(pos)} is event {int(g.attrs['event_id'])}, "
                         f"/dprod/events says {int(event_id)}")
    return g


def task_files(sensor_file) -> dict:
    """``{sensor, step, hits}``: the paths of the task a sensor file belongs to."""
    p = Path(sensor_file)
    task = p.parent.parent
    return {k: str(task / k / p.name.replace("_sensor_", f"_{k}_")) for k in KINDS}


def sensor_path(root, group, source_file) -> str:
    """The sensor file ``source_file`` (a file name) of group ``group`` under ``root``."""
    task = Path(source_file).stem.split("_sensor_", 1)[1]
    return str(Path(root) / group / task / "sensor" / source_file)


@dataclass(frozen=True, eq=False)
class SensorLayout:
    """A sensor file's readout layout: what the reader and the TPC chain need.

    ``plane_labels``   ``volume_{V}_{P}``, volume-major, U V Y;
    ``num_time_steps`` ticks per plane;
    ``pedestals``      ``{plane type: ADC}`` (U / V / Y);
    ``num_wires``      (n_volumes, 3) ``/config/num_wires``: the chain's layout readout
                       (:func:`helix.tpc.chain.sim_config`);
    ``campaign``       ``/config`` production_version.
    """
    plane_labels: tuple
    num_time_steps: int
    pedestals: dict
    num_wires: np.ndarray
    campaign: str = ""

    def pedestal(self, plane_label) -> int:
        return int(self.pedestals[plane_label.rsplit("_", 1)[1]])

    def n_wires(self, plane_label) -> int:
        v, p = _split_label(plane_label)
        return int(self.num_wires[v, PLANES.index(p)])

    def plane_gid(self, plane_label) -> int:
        """Canonical plane id ``volume * 3 + {U: 0, V: 1, Y: 2}`` (pimm-data's;
        :func:`helix.tpc.readout.canonical_plane_gid`)."""
        v, p = _split_label(plane_label)
        return canonical_plane_gid(v, PLANES.index(p))


def _split_label(plane_label):
    """``'volume_1_V'`` -> (1, 'V')."""
    vol, plane = plane_label.rsplit("_", 1)
    return int(vol.split("_")[1]), plane


def sensor_layout(path) -> SensorLayout:
    """The :class:`SensorLayout` of a sensor file, from its ``/config``."""
    with h5py.File(path, "r") as f:
        require_format(f)
        cfg = f["config"]
        nw = np.asarray(cfg["num_wires"][:], np.int32)
        ped = np.asarray(cfg["pedestals"][:], np.int64)
        if nw.ndim != 2 or nw.shape[1] != len(PLANES) or ped.shape != nw.shape:
            raise ValueError(f"{path}: num_wires {nw.shape} / pedestals {ped.shape} are not (n_volumes, 3)")
        if (ped != ped[:1]).any():
            raise ValueError(f"{path}: pedestals differ between volumes: {ped.tolist()}")
        return SensorLayout(
            plane_labels=tuple(f"volume_{v}_{p}" for v in range(nw.shape[0]) for p in PLANES),
            num_time_steps=int(cfg.attrs["num_time_steps"]),
            pedestals={p: int(ped[0, i]) for i, p in enumerate(PLANES)},
            num_wires=nw, campaign=str(cfg.attrs.get("production_version", "")))


def read_event_coo(evt: h5py.Group, layout: SensorLayout) -> dict:
    """One sensor entry -> ``{label: (wire, time, value)}`` for every plane of the layout, as sparse
    COO: int32 wire and tick, float32 pedestal-subtracted ADC. An empty plane is zero hits."""
    out = {}
    for label in layout.plane_labels:
        v, p = _split_label(label)
        g = evt[f"volume_{v}"][p]
        if "values" not in g:                       # nothing deposited in this volume
            z = np.zeros(0, np.int32)
            out[label] = (z, z.copy(), np.zeros(0, np.float32))
            continue
        wire = np.cumsum(g["delta_wire"][:], dtype=np.int32) + np.int32(g.attrs["wire_start"])
        time = np.cumsum(g["delta_time"][:], dtype=np.int32) + np.int32(g.attrs["time_start"])
        value = g["values"][:].astype(np.float32) - np.float32(g.attrs["pedestal"])
        out[label] = (wire, time, value)
    return out


def read_event(path, pos: int, layout: SensorLayout | None = None) -> dict:
    """Entry ``pos`` of a sensor file -> ``{label: (n_wires, n_ticks) float32}`` dense
    pedestal-subtracted images (:func:`read_event_coo`, scattered)."""
    layout = layout or sensor_layout(path)
    with h5py.File(path, "r") as f:
        coo = read_event_coo(f[entry_key(pos)], layout)
    planes = {}
    for label, (w, t, v) in coo.items():
        img = np.zeros((layout.n_wires(label), layout.num_time_steps), np.float32)
        img[w, t] = v
        planes[label] = img
    return planes
