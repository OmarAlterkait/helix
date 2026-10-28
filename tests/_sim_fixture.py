"""Synthetic simulation ``hits`` + ``step`` files in the JAXTPC layout, for truth tests.

``step`` carries per deposit ``de``, ``charge`` and quantised ``positions`` (dequantised by
``/config/volume_ranges``).

Per volume: compact particles (tracks) of one to three groups each, scattered over
the plane, some placed on top of one another so cells hold charge from several
tracks; one group's samples may land on the same pixel more than once. A volume listed in
``empty`` has no deposit, written as the simulator writes one: ``n_actual = 0``, planes with no
entries, ``group_to_track`` holding only the null group, and no per-deposit arrays.
"""
import os

import h5py
import numpy as np


def _plane(rng, centres):
    """One plane group's arrays from ``[(group_id, cw, ct)]``."""
    cw, ct, gid, sz, pk, dw, dt, qs = [], [], [], [], [], [], [], []
    for g, w, t in centres:
        n = int(rng.integers(3, 40))
        cw.append(w); ct.append(t); gid.append(g); sz.append(n); pk.append(rng.uniform(50, 2000))
        dw.append(rng.integers(-3, 4, n)); dt.append(rng.integers(-12, 13, n)); qs.append(rng.integers(0, 65536, n))
    return dict(center_wires=np.array(cw, np.int16), center_times=np.array(ct, np.int16),
                group_ids=np.array(gid, np.int32), group_sizes=np.array(sz, np.uint8),
                peak_charges=np.array(pk, np.float32), delta_wires=np.concatenate(dw).astype(np.int8),
                delta_times=np.concatenate(dt).astype(np.int8), charges_u16=np.concatenate(qs).astype(np.uint16))


def write_sim(root, run, tag, n_events, seed=0, n_volumes=2, n_tracks=30, empty=()):
    """``root/{hits,step}/<run>/sim_wire_{hits,step}_<tag>.h5``; returns the hits path."""
    rng = np.random.default_rng(seed)
    paths = {k: os.path.join(root, k, run, f"sim_wire_{k}_{tag}.h5") for k in ("hits", "step")}
    for p in paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
    with h5py.File(paths["hits"], "w") as fh, h5py.File(paths["step"], "w") as fs:
        for f in (fh, fs):
            f.create_group("config").attrs.update(production_version="test", git_commit="abc", git_dirty=False)
        # per volume (x, y, z) x (lo, hi), mm: what dequantises step positions
        fs["config"]["volume_ranges"] = np.array([[[-2000.0, 0.0], [-2000.0, 2000.0], [0.0, 5000.0]],
                                                  [[0.0, 2000.0], [-2000.0, 2000.0], [0.0, 5000.0]]])[:n_volumes]
        for ev in range(n_events):
            eh, es = fh.create_group(f"event_{ev:03d}"), fs.create_group(f"event_{ev:03d}")
            eh.attrs["threshold"] = 25.0
            for v in range(n_volumes):
                if (ev, v) in empty:
                    hv, sv = eh.create_group(f"volume_{v}"), es.create_group(f"volume_{v}")
                    hv.attrs.update(n_actual=0, n_groups=1); sv.attrs["n_actual"] = 0
                    hv["group_to_track"] = np.zeros(1, np.int32)
                    for p in "UVY":
                        hv.create_group(p).attrs.update(n_entries=0, n_groups_plane=0)
                    continue
                n_groups = 0
                g2t, centres = [], {p: [] for p in "UVY"}
                for trk in range(n_tracks):
                    tid = int(rng.integers(0, 10_000))
                    if trk % 5 == 4 and centres["U"]:                 # on top of the previous track
                        base = {p: centres[p][-1][1:] for p in "UVY"}
                    else:
                        base = {p: (int(rng.integers(40, 900)), int(rng.integers(200, 3000))) for p in "UVY"}
                    for _ in range(int(rng.integers(1, 4))):
                        for p in "UVY":
                            centres[p].append((n_groups, base[p][0] + int(rng.integers(-2, 3)),
                                               base[p][1] + int(rng.integers(-8, 9))))
                        g2t.append(tid); n_groups += 1
                n_dep = 4 * n_groups
                hv, sv = eh.create_group(f"volume_{v}"), es.create_group(f"volume_{v}")
                hv.attrs.update(n_actual=n_dep, n_groups=n_groups); sv.attrs["n_actual"] = n_dep
                hv["group_to_track"] = np.array(g2t, np.int32)
                hv["deposit_to_group"] = rng.integers(0, n_groups, n_dep).astype(np.int32)
                sv["de"] = rng.uniform(0, 0.05, n_dep).astype(np.float16)
                sv["charge"] = rng.uniform(0, 5000, n_dep).astype(np.float32)
                sv["positions"] = rng.integers(0, 65536, (n_dep, 3)).astype(np.uint16)
                for p in "UVY":
                    pg = hv.create_group(p)
                    for k, a in _plane(rng, centres[p]).items():
                        pg[k] = a
    return paths["hits"]
