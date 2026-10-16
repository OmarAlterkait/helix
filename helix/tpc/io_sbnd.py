"""Read SBND events from the extracted HDF5 files (SBND-data-reconstruction/extract).

Per file: ``tpc/adc`` int16 (events, 11276, 3427) chunked (1, 256, 3427) gzip 4 +
shuffle, ``tpc/pedestal`` (events, 11276), ``tpc/present`` (events, 11276) and
``events/{run, subrun, event, time_s, time_ns}``.

Decompressing ~77 MB of ADC per event is CPU work, and h5py serialises all HDF5 calls
behind one lock, so threads do not overlap it: :func:`loader` uses a torch DataLoader
with worker PROCESSES (each opens its own file handles) and pinned memory, so the
next events decode while the GPU processes the current one.
"""
from __future__ import annotations

import h5py
import numpy as np


class SBNDEvents:
    """Random access over (file, index) pairs of extracted SBND files."""

    def __init__(self, files, optical_kinds=()):
        self.files = list(files)
        self.optical_kinds = tuple(optical_kinds)
        self._ev = {}
        self.index = []
        for fi, p in enumerate(self.files):
            with h5py.File(p, "r") as f:
                n = f["events/event"].shape[0]
            self.index += [(fi, i) for i in range(n)]
        self._h = {}

    def __len__(self):
        return len(self.index)

    def _f(self, fi):
        if fi not in self._h:
            self._h[fi] = h5py.File(self.files[fi], "r")
        return self._h[fi]

    def optical(self, k, kind):
        """Waveforms of one event: dict(channel, t_us, length, offset (into buf), buf uint16)."""
        fi, i = self.index[k]
        f = self._f(fi)
        key = (fi, kind)
        if key not in self._ev:
            e = f[f"{kind}/wf/event_idx"][:]
            if len(e) > 1 and not np.all(np.diff(e) >= 0):
                raise ValueError(f"{self.files[fi]}: {kind}/wf/event_idx is not sorted")
            self._ev[key] = e
        e = self._ev[key]
        lo, hi = np.searchsorted(e, i, "left"), np.searchsorted(e, i, "right")
        g = f[f"{kind}/wf"]
        off = g["offset"][lo:hi].astype(np.int64); n = g["length"][lo:hi].astype(np.int64)
        if hi == lo:
            return dict(channel=np.zeros(0, np.int32), t_us=np.zeros(0), length=n, offset=off, buf=np.zeros(0, np.uint16))
        a, b = int(off[0]), int(off[-1] + n[-1])
        return dict(channel=g["channel"][lo:hi].astype(np.int32), t_us=g["t_us"][lo:hi], length=n, offset=off - a,
                    buf=f[f"{kind}/adc"][a:b])

    def __getitem__(self, k):
        fi, i = self.index[k]
        f = self._f(fi)
        ev = f["events"]
        meta = dict(file=self.files[fi], index=i, run=int(ev["run"][i]), subrun=int(ev["subrun"][i]),
                    event=int(ev["event"][i]), time=float(ev["time_s"][i]) + 1e-9 * float(ev["time_ns"][i]))
        out = dict(adc=f["tpc/adc"][i], pedestal=f["tpc/pedestal"][i].astype(np.float32),
                   present=f["tpc/present"][i].astype(bool), meta=meta)
        for kind in self.optical_kinds:
            out[kind] = self.optical(k, kind)
        return out

    def __getstate__(self):                      # worker processes reopen files
        s = self.__dict__.copy()
        s["_h"] = {}
        s["_ev"] = {}
        return s


def _collate(x):
    import torch
    out = dict(x)
    out.update(adc=torch.from_numpy(x["adc"]), pedestal=torch.from_numpy(x["pedestal"]),
               present=torch.from_numpy(x["present"]))
    return out


def loader(files, *, workers=4, prefetch=2, indices=None, optical_kinds=()):
    """Iterate events (TPC arrays as pinned CPU tensors, optical waveforms as numpy, meta),
    decoded in worker processes."""
    import torch
    from torch.utils.data import DataLoader, Subset
    ds = SBNDEvents(files, optical_kinds)
    if indices is not None:
        ds = Subset(ds, list(indices))
    return DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers, collate_fn=_collate,
                      pin_memory=torch.cuda.is_available(), prefetch_factor=prefetch if workers > 0 else None,
                      persistent_workers=False)
