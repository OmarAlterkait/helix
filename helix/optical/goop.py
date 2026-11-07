"""goop light files as a build source, optical only (the codec's GOOP specs and responses); goop's
photon counts per PMT go into the optical shards as ``truth/pe_counts``.

    python -m helix.build goop --input light.h5 --output OUT/RUN/goop --device cuda
    python -m helix.build goop plan OUT RUN_DIR...

Options and the shards: :mod:`helix.build`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class GoopSource:
    """goop light files: both sides' stored chunks per event (channel = pmt_id, +1000 west)."""

    systems = ("goop",)
    dataset = "goop"
    pattern = "*.h5"

    def __init__(self, files, device, optical=True):
        import h5py
        self.device = device
        self.index = []
        for p in files:
            with h5py.File(p, "r") as f:
                self.index += [(p, k) for k in sorted(k for k in f.keys() if k.startswith("event_"))]

    def __len__(self):
        return len(self.index)

    def optical_chain(self, overrides):
        from helix.optical.codec import GOOP, OpticalChain
        return OpticalChain(overrides or GOOP, device=self.device, templates="goop")

    @staticmethod
    def waveforms(f, key):
        """One event's stored chunks: dict(channel, t_us, length, offset, buf) and pe_counts
        (2, n_pmt) east / west."""
        chans, t0, lens, bufs, pe = [], [], [], [], []
        for si, side in enumerate(("east", "west")):
            g = f[key][side]
            o = g["offsets"][:]
            chans.append(g["pmt_id"][:].astype(np.int32) + 1000 * si)
            t0.append(g["t0_ns"][:] / 1e3)
            lens.append(np.diff(o)); bufs.append(g["adc"][:])
            pe.append(f[key][f"pe_counts_{side}"][:] if f"pe_counts_{side}" in f[key] else np.zeros(0))
        L = np.concatenate(lens).astype(np.int64)
        wf = dict(channel=np.concatenate(chans), t_us=np.concatenate(t0), length=L,
                  offset=np.r_[0, np.cumsum(L)[:-1]].astype(np.int64), buf=np.concatenate(bufs))
        n = max(len(p) for p in pe)
        return wf, np.stack([np.pad(p, (0, n - len(p))) for p in pe]) if n else np.zeros((2, 0))

    def events(self, indices, workers):
        import h5py
        files = {}
        for k in indices:
            path, key = self.index[k]
            f = files.get(path) or files.setdefault(path, h5py.File(path, "r"))
            wf, pe = self.waveforms(f, key)
            yield dict(ident=dict(run=Path(path).stem, source_file=Path(path).name, event=int(key.split("_")[1])),
                       optical={"goop": wf}, truth=dict(pe_counts=pe))
        for f in files.values():
            f.close()

    def shard_attrs(self):
        return dict(noise=None, provenance=None)

