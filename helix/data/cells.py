"""Simulation truth on the fine cell grid: one ``{name}_cells_<k>.h5`` per coefficient shard.

The label every downstream task scores against -- the denoiser's target and the
detection-floor / resolution evaluation -- built once from the simulation's
``hits`` (and ``step``, for deposited energy), separately from the corpus build:
labels change more often than the corpus. Same events, same order as
``{name}_coeff_<k>.h5``; joined to it by identity (``/ident``) as the dataset
modality ``cells`` (``CoeffTPCDataset(modalities=("coeff", "cells"))``).
Pretraining never opens it (tests/test_noisy_training.py).

Grid: 2 wires x 16 raw ticks, the fine cell of :mod:`helix.probe.resolution`
(``FW``, ``FT``, ``fkey``). Not native resolution: every quantity the denoiser
and the evaluation use is decided per cell (see
:func:`helix.probe.resolution.cell_truth`) except each particle's charge
centroid, which is stored per (plane, particle).

Layout (four tables, each a per-event CSR with its own ``event_offset``)::

    /config       attrs: format ("helix-cells-1"), n_events, grid_fw, grid_ft, key,
                         label_json, provenance_json
    /cells        key int64 (fkey), q float32                 one row per cell, sorted by key
    /cell_track   cell int32, track int32, q float32          the cells' charge split by track
                                                              (rows of /cells, /tracks)
    /plane_track  plane_gid int32, track int32,               per (plane, track): its PIXEL
                  wire float64, tick float64                  charge-weighted centroid
    /tracks       volume int16, track_id int32, energy f32    hits group_to_track ids, per volume;
                                                              energy = step de summed, MeV
    /ident        run, source_file, event                     copied from the coeff shard

A per-cell label (e.g. a particle type of the dominant track) is a column of
``/cells``; a per-track one is a column of ``/tracks``.

numpy + h5py only (no pimm-data): ``helix.data.coeff_reader.CellsReader`` serves
it to the dataset, everything else reads :func:`read_event` directly.
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np

FORMAT = "helix-cells-1"

#: (h5 group, column, record name, dtype). The record names are those of
#: :func:`helix.probe.resolution.cell_truth`, plus the track table's.
SCHEMA = (
    ("cells", "key", "key", np.int64),
    ("cells", "q", "q", np.float32),
    ("cell_track", "cell", "ct_cell", np.int32),
    ("cell_track", "track", "ct_track", np.int32),
    ("cell_track", "q", "ct_q", np.float32),
    ("plane_track", "plane_gid", "pt_gid", np.int32),
    ("plane_track", "track", "pt_track", np.int32),
    ("plane_track", "wire", "pt_wire", np.float64),
    ("plane_track", "tick", "pt_tick", np.float64),
    ("tracks", "volume", "trk_volume", np.int16),
    ("tracks", "track_id", "trk_id", np.int32),
    ("tracks", "energy", "trk_energy", np.float32),
)
GROUPS = tuple(dict.fromkeys(g for g, *_ in SCHEMA))

LABEL = dict(
    source="hits",
    charge="hits charges_u16 / 65535 * peak_charges, summed over every sample of every group in the cell (e-)",
    track="hits group_to_track[group], numbered per volume",
    energy="step de summed over the track's deposits (hits deposit_to_group), MeV",
    centroid="per (plane, track): charge-weighted pixel wire and tick",
)
_STR = h5py.string_dtype(encoding="utf-8")


def cells_path(coeff_shard, out_dir=None):
    """``{name}_coeff_<k>.h5`` -> ``{name}_cells_<k>.h5`` (beside it, or in ``out_dir``)."""
    d, b = os.path.split(str(coeff_shard))
    stem, k = b[:-3].rsplit("_coeff_", 1)
    if not k.isdigit():
        raise ValueError(f"{coeff_shard}: not a {{name}}_coeff_<k>.h5 shard")
    return os.path.join(out_dir or d, f"{stem}_cells_{k}.h5")


def require_format(cfg, where=""):
    """Refuse a file that is not :data:`FORMAT` on this grid."""
    from helix.probe.resolution import FT, FW
    got = cfg.attrs.get("format")
    got = got.decode() if isinstance(got, bytes) else got
    if got != FORMAT:
        raise ValueError(f"{where or 'cells shard'}: format {got!r}, not {FORMAT!r}; rebuild it (scripts/build_truth.py)")
    if (int(cfg.attrs["grid_fw"]), int(cfg.attrs["grid_ft"])) != (FW, FT):
        raise ValueError(f"{where}: grid {cfg.attrs['grid_fw']} x {cfg.attrs['grid_ft']}, helix scores on {FW} x {FT}")


# ------------------------------------------------------------------------------------------ build
def event_record(hits_ev, step_ev):
    """One simulation event (its ``hits`` and ``step`` groups) -> a record (:data:`SCHEMA` names).

    Samples are concatenated volume by volume (sorted by name), plane by plane, in
    stored order, so each cell's charge is summed in exactly the order
    ``scripts/build_denoise_truth.py`` sums it.
    """
    from helix.probe.resolution import cell_truth
    from helix.probe.truth import PLANES, decode_hits_plane

    G, W, T, Q, TRK, vol, tid, energy = [], [], [], [], [], [], [], []
    base = 0
    for vk in sorted(x for x in hits_ev if x.startswith("volume_")):
        vi, hv = int(vk.split("_")[1]), hits_ev[vk]
        d2g = hv["deposit_to_group"][:].astype(np.int64)
        g2t = hv["group_to_track"][:].astype(np.int64)
        de = step_ev[vk]["de"][:].astype(np.float64)
        if len(de) != len(d2g):
            raise ValueError(f"{vk}: step has {len(de)} deposits, hits deposit_to_group {len(d2g)}")
        planes = [(pi, decode_hits_plane(hv[pl])) for pi, pl in enumerate(PLANES) if pl in hv]
        tv = np.unique(np.concatenate([g2t[d2g]] + [g2t[s["group"]] for _, s in planes]))
        energy.append(np.bincount(np.searchsorted(tv, g2t[d2g]), weights=de, minlength=len(tv)))
        vol.append(np.full(len(tv), vi)); tid.append(tv)
        for pi, s in planes:
            G.append(np.full(len(s["q"]), vi * 3 + pi)); W.append(s["wire"]); T.append(s["tick"])
            Q.append(np.asarray(s["q"], np.float64)); TRK.append(base + np.searchsorted(tv, g2t[s["group"]]))
        base += len(tv)
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64)
    rec = cell_truth(cat(G), cat(W), cat(T), cat(Q), cat(TRK))
    rec.update(trk_volume=cat(vol), trk_id=cat(tid), trk_energy=cat(energy))
    return {name: np.asarray(rec[name]).astype(dt) for _, _, name, dt in SCHEMA}


def write_shard(path, ident, records, *, label, provenance):
    """Write ``records`` (one per event of ``ident``) to ``path``, whole or not at all."""
    if len(records) != len(ident["event"]):
        raise ValueError(f"{len(records)} records for {len(ident['event'])} events")
    part = str(path) + ".part"
    with h5py.File(part, "w") as f:
        cfg = f.create_group("config")
        from helix.probe.resolution import FT, FW
        cfg.attrs.update(format=FORMAT, n_events=len(records), grid_fw=FW, grid_ft=FT,
                         key="fkey(plane_gid, wire // grid_fw, tick // grid_ft) = g << 40 | fw << 20 | (ft + 4096)",
                         label_json=json.dumps(label, sort_keys=True),
                         provenance_json=json.dumps(provenance, sort_keys=True))
        for grp in GROUPS:
            cols = [(col, name, dt) for g, col, name, dt in SCHEMA if g == grp]
            n = [len(r[cols[0][1]]) for r in records]
            for r in records:
                if any(len(r[name]) != len(r[cols[0][1]]) for _, name, _ in cols):
                    raise ValueError(f"/{grp}: columns of unequal length")
            h = f.create_group(grp)
            h.create_dataset("event_offset", data=np.concatenate([[0], np.cumsum(n)]).astype(np.int64))
            for col, name, dt in cols:
                data = np.concatenate([np.asarray(r[name], dt) for r in records]) if records else np.zeros(0, dt)
                h.create_dataset(col, data=data, compression="gzip", compression_opts=4, shuffle=True)
        idg = f.create_group("ident")
        idg.create_dataset("run", data=np.array(ident["run"], dtype=_STR))
        idg.create_dataset("source_file", data=np.array(ident["source_file"], dtype=_STR))
        idg.create_dataset("event", data=np.asarray(ident["event"], np.int64))
    os.replace(part, path)


def _text(col):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in col]


def read_ident(f):
    """``/ident`` of an open coeff or cells file as ``{run, source_file, event}`` lists/arrays."""
    return dict(run=_text(f["ident"]["run"][:]), source_file=_text(f["ident"]["source_file"][:]),
                event=f["ident"]["event"][:].astype(np.int64))


def build_shard(coeff_shard, source_root, out_dir=None):
    """Write the cells shard of ``coeff_shard``; ``(path, message)``. An existing one is kept.

    Events come from the coeff shard's ``/ident`` (any number of source files);
    each is read from ``<source_root>/{hits,step}/<run>/<source_file>`` with
    ``_sensor_`` replaced by ``_hits_`` / ``_step_``.
    """
    import hdf5plugin  # noqa: F401  (hits / step use a plugin filter)
    from helix.core.coeff_io import _code_version

    out = cells_path(coeff_shard, out_dir)
    if os.path.exists(out):
        return out, "exists"
    with h5py.File(coeff_shard, "r") as f:
        ident = read_ident(f)
    records, threshold, sim, files = [], set(), {}, {}
    try:
        for run, sf, ev in zip(ident["run"], ident["source_file"], ident["event"].tolist()):
            if (run, sf) not in files:
                files[(run, sf)] = [h5py.File(os.path.join(source_root, kind, run, sf.replace("_sensor_", f"_{kind}_")), "r")
                                    for kind in ("hits", "step")]
                c = files[(run, sf)][0]["config"].attrs
                sim[f"{run}/{sf}"] = {k: (c[k].item() if hasattr(c[k], "item") else str(c[k]))
                                      for k in ("production_version", "git_commit", "git_dirty") if k in c}
            fh, fs = files[(run, sf)]
            hits_ev = fh[f"event_{ev:03d}"]
            threshold.add(float(hits_ev.attrs["threshold"]))
            records.append(event_record(hits_ev, fs[f"event_{ev:03d}"]))
    finally:
        for pair in files.values():
            for x in pair:
                x.close()
    if len(threshold) > 1:
        raise ValueError(f"{coeff_shard}: hits written with different sample thresholds {sorted(threshold)}")
    label = dict(LABEL, hits_threshold=threshold.pop() if threshold else None)
    write_shard(out, ident, records, label=label,
                provenance=dict(code=_code_version(), coeff_shard=os.path.basename(str(coeff_shard)), sim=sim))
    n = sum(len(r["key"]) for r in records)
    return out, f"{len(records)} events, {n:,} cells"


# ------------------------------------------------------------------------------------------- read
def read_event(f, pos):
    """Event at write position ``pos`` of an open cells file -> a record (:data:`SCHEMA` names)."""
    out = {}
    for grp in GROUPS:
        off = f[grp]["event_offset"]
        a, b = int(off[pos]), int(off[pos + 1])
        for g, col, name, dt in SCHEMA:
            if g == grp:
                out[name] = f[grp][col][a:b]
    return out


def open_event(path, pos):
    """``(record, ident)`` of event ``pos`` of a cells shard at ``path``, format-checked."""
    with h5py.File(path, "r") as f:
        require_format(f["config"], str(path))
        ident = read_ident(f)
        return read_event(f, pos), dict(run=ident["run"][pos], source_file=ident["source_file"][pos],
                                        event=int(ident["event"][pos]))
