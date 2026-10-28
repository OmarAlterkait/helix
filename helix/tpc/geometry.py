"""Plane geometry registry — the detector description the forward model needs.

DUPLICATE of ``pimm_data.geometry`` -- byte-identical loader, and the bundled
``cubic_wireplane_geometry.json`` is byte-identical too. ``test_forward_mirror.py``
used to compare them element by element; it was deleted with the forward model,
which means this duplication is currently pinned by NOTHING (an open item in
``TODO.md``).

The duplication is deliberate and is the same arrangement as the shard codec:
pimm-data must run with no helix installed (it serves every detector, not just
the coefficient pipeline), and helix must build a corpus with no pimm-data
installed. Neither may depend on the other, so the shared concern is mirrored and
pinned by a test instead of imported.

    geom = load_plane_registry("cubic_wireplane_geometry.json")   # ships in data/

Keyed by canonical plane gid (matching the per-point ``plane_gid`` the sensor
reader surfaces), with per-plane ``n_wires``/``n_ticks``/``pedestal``/
``wire_lengths`` (meters) — what densify / noise / digitize consume.
"""

import json
import os

import numpy as np

from helix.tpc.readout import canonical_plane_gid

_PLANE_INDEX = {'U': 0, 'V': 1, 'Y': 2}
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def _resolve(path):
    for cand in (path, os.path.join(_DATA_DIR, path),
                 os.path.join(_DATA_DIR, str(path) + '.json')):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"plane-geometry file {path!r} not found (looked in cwd and {_DATA_DIR})")


def _wire_lengths(spec, n_wires):
    """Constant planes store a scalar (collection); varying ones a per-wire array."""
    if isinstance(spec, (int, float)):
        return np.full(int(n_wires), float(spec), dtype=np.float32)
    arr = np.asarray(spec, dtype=np.float32)
    if arr.shape != (int(n_wires),):
        raise ValueError(f"wire_lengths length {arr.shape} != n_wires {n_wires}")
    return arr


def _label_gid(label):
    """``'volume_{v}_{U|V|Y}'`` -> ``v*3 + {U:0,V:1,Y:2}`` (matches pimm-data)."""
    vol, plane = label.rsplit('_', 1)
    return canonical_plane_gid(int(vol.split('_')[1]), _PLANE_INDEX[plane])


def load_plane_registry(path):
    """JSON -> ``{plane gid: {label, n_wires, n_ticks, pedestal, wire_lengths}}``."""
    with open(_resolve(path)) as f:
        d = json.load(f)
    nts = int(d['num_time_steps'])
    reg = {}
    for label, e in d['planes'].items():
        nw = int(e['n_wires'])
        reg[_label_gid(label)] = {
            'label': label, 'n_wires': nw, 'n_ticks': nts,
            'pedestal': int(e['pedestal']),
            'wire_lengths': _wire_lengths(e['wire_lengths_m'], nw),
        }
    return reg
