"""Plane geometry registry — the detector description the forward model needs.

MIRROR of ``pimm_data.geometry``. The two are kept identical by
``tests/test_forward_mirror.py``, which compares them element by element.

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

from helix.tpc.pipeline import canonical_plane_gid

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


def load_plane_registry(path):
    """JSON -> ``{canonical_plane_gid(label): {label, n_wires, n_ticks, pedestal, wire_lengths}}``."""
    with open(_resolve(path)) as f:
        d = json.load(f)
    nts = int(d['num_time_steps'])
    reg = {}
    for label, e in d['planes'].items():
        nw = int(e['n_wires'])
        reg[canonical_plane_gid(label)] = {
            'label': label, 'n_wires': nw, 'n_ticks': nts,
            'pedestal': int(e['pedestal']),
            'wire_lengths': _wire_lengths(e['wire_lengths_m'], nw),
        }
    return reg
