"""Event loading for the classical baseline: the floor evaluation's test events,
their clean sensor planes, their pre-response charge (hits), and forward noise.

Loading follows scripts/snr_stages.py / scripts/noise_vs_hits.py exactly: the
truth file names a coeff shard + event id; the shard's ``ident`` gives the
source_file; sensor and hits live under ``<source>/{sensor,hits}/<run>/``.
"""
from __future__ import annotations

import json
import os

import numpy as np

SOURCE = "/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02"
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
GEOM = "cubic_wireplane_geometry.json"
PLANES = "UVY"


def registry():
    from helix.tpc.geometry import load_plane_registry
    return load_plane_registry(GEOM)


class Event:
    """One floor-evaluation event: sensor/hits paths and the scoring truth."""

    def __init__(self, ev, truth_dir=TRUTH, source=SOURCE):
        import h5py
        self.ev = int(ev)
        self.truth_file = os.path.join(truth_dir, f"ev{self.ev:03d}.npz")
        z = np.load(self.truth_file, allow_pickle=True)
        self.z = {k: z[k] for k in z.files}
        self.meta = json.loads(str(self.z["meta"]))
        self.shard, self.event = str(self.z["shard"]), int(self.z["event"])
        with h5py.File(self.shard, "r") as fh:
            ids = fh["ident"]["event"][:]
            self.pos = int(np.searchsorted(ids, self.event))
            if ids[self.pos] != self.event:
                raise ValueError(f"event {self.event} not in {self.shard}")
            sf = fh["ident"]["source_file"][self.pos]
        self.source_file = sf.decode() if isinstance(sf, bytes) else str(sf)
        self.run = os.path.basename(os.path.dirname(self.shard))
        self.sensor = os.path.join(source, "sensor", self.run, self.source_file)
        self.hits = os.path.join(source, "hits", self.run, self.source_file.replace("_sensor_", "_hits_"))

    def clean(self, g, reg):
        """Clean (noise-free) post-response image, (n_wires, n_ticks) ADC, pedestal-subtracted."""
        from helix.tpc.io import read_sensor_plane
        return np.asarray(read_sensor_plane(self.sensor, self.event, reg[g]["label"]), np.float32)

    def truth(self, g, shape):
        """Pre-response charge per (wire, tick), electrons, summed over particle groups."""
        return hits_plane(self.hits, self.event, g, shape)


def hits_plane(hits_file, event, g, shape):
    import h5py
    import hdf5plugin  # noqa: F401
    from helix.probe.truth import decode_hits_plane
    vi, pi = divmod(int(g), 3)
    out = np.zeros(shape, np.float64)
    with h5py.File(hits_file, "r") as fh:
        ev = fh[f"event_{int(event):03d}"]
        vk = f"volume_{vi}"
        if vk not in ev or PLANES[pi] not in ev[vk]:
            return out.astype(np.float32)
        s = decode_hits_plane(ev[vk][PLANES[pi]])
    w, t, q = s["wire"], s["tick"], s["q"]
    ok = (w >= 0) & (w < shape[0]) & (t >= 0) & (t < shape[1])
    np.add.at(out, (w[ok], t[ok]), q[ok])
    return out.astype(np.float32)


def noisy_planes(clean, L, ped, rng, group_size=64):
    """The corpus's forward model (scripts/build_coeff_corpus.py, replayed as in
    scripts/snr_stages.py): returns (inc, coh) where
      inc = digitize(clean + incoherent)             -- coherent removed perfectly
      coh = digitize(clean + incoherent + coherent)  -- needs removal
    Both share the same incoherent realisation."""
    from helix.tpc.noise import digitize, generate_noise
    nw, nt = clean.shape
    n_inc = generate_noise((nw, nt), rng=rng, wire_lengths_m=L, incoherent=True, coherent=False)
    n_coh = generate_noise((nw, nt), rng=rng, incoherent=False, coherent=True, group_size=group_size)
    return digitize(clean + n_inc, ped), digitize(clean + n_inc + n_coh, ped)
