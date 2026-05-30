"""HDF5 I/O for goop two-sided (east/west) optical light files.

The wavelet path operates on the STORED chunks (goop's gap-compressed
"stitches") directly — no deslicing — since each chunk is already a contiguous,
pedestal-baselined active segment. ``deslice_side`` is kept only for
visualization / dense views.

NOTE: this reads the east/west schema (event_NNN/{east,west}/{adc,offsets,
t0_ns,pmt_id} + pe_counts_{side}); the upstream goop loader expects label_N
groups and will not read these files.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import h5py

from helix.optical.config import OpticalConfig


def config_from_file(path: str | Path, **overrides) -> OpticalConfig:
    """Build an OpticalConfig from a light file's /config attrs."""
    with h5py.File(path, "r") as f:
        a = dict(f["config"].attrs)
    g = lambda k, d: (a[k].item() if hasattr(a.get(k), "item") else a.get(k, d))
    kw = dict(
        n_pmts_per_side=int(g("n_pmts_per_side", 81)),
        n_channels=int(g("n_channels", 162)),
        tick_ns=float(g("tick_ns", 1.0)),
        pedestal=float(g("pedestal", 0.0)),
        gain=float(g("gain", 1.0)),
        n_bits=int(g("n_bits", 15)),
        baseline_noise_std=float(g("baseline_noise_std", 0.0)),
    )
    kw.update(overrides)
    return OpticalConfig(**kw)


def list_events(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as f:
        return sorted(k for k in f.keys() if k.startswith("event_"))


@dataclass
class EventChunks:
    """Pedestal-subtracted stored chunks for one event, both sides concatenated.

    chunks  : list of 1-D float32 arrays (variable length), the transform units
    side    : (n_chunks,) 'east'/'west'
    pmt_id  : (n_chunks,) per-side PMT index
    t0_ns   : (n_chunks,) chunk start time
    lengths : (n_chunks,) original chunk lengths
    """
    chunks: list[np.ndarray]
    side: np.ndarray
    pmt_id: np.ndarray
    t0_ns: np.ndarray
    lengths: np.ndarray
    pe_counts: dict[str, np.ndarray]


def read_event_chunks(path: str | Path, event_key: str, config: OpticalConfig) -> EventChunks:
    """Read the stored chunks of one event (pedestal-subtracted)."""
    chunks, sides, pmts, t0s, lens = [], [], [], [], []
    pe = {}
    with h5py.File(path, "r") as f:
        evt = f[event_key]
        for side in config.sides:
            if side not in evt:
                continue
            g = evt[side]
            adc = g["adc"][:].astype(np.float32) - config.pedestal
            offsets = g["offsets"][:]
            t0_ns = g["t0_ns"][:]
            pmt_id = g["pmt_id"][:]
            for k in range(len(pmt_id)):
                c = adc[offsets[k]:offsets[k + 1]]
                chunks.append(c)
                sides.append(side); pmts.append(int(pmt_id[k]))
                t0s.append(float(t0_ns[k])); lens.append(len(c))
            pek = f"pe_counts_{side}"
            if pek in evt:
                pe[side] = evt[pek][:]
    return EventChunks(chunks=chunks, side=np.array(sides), pmt_id=np.array(pmts),
                       t0_ns=np.array(t0s, np.float64), lengths=np.array(lens), pe_counts=pe)


def chunk_noise_sigma(chunks: list[np.ndarray]) -> np.ndarray:
    """Per-chunk noise sigma from the UNPADDED chunk's finest db1 detail (MAD).

    Padding-independent — used for the VisuShrink threshold so zero-padding in
    the batch can't collapse the MAD estimate. Returns (n_chunks,) float32.
    """
    import pywt
    out = []
    for c in chunks:
        d = pywt.wavedec(np.asarray(c, np.float64), "db1", level=1, mode="periodization")[-1]
        out.append(np.median(np.abs(d)) / 0.6745)
    return np.asarray(out, np.float32)


def quantize_coeffs(coeffs, bits: int):
    """Per-signal uniform quantization of coefficient bands (bits<=0 or >=32 = no-op)."""
    if bits is None or bits <= 0 or bits >= 32:
        return coeffs
    import numpy as _np
    out = []
    for c in coeffs:
        a = _np.asarray(c)
        amax = _np.abs(a).max(axis=-1, keepdims=True); amax[amax == 0] = 1.0
        step = amax / ((1 << (bits - 1)) - 1)
        out.append((_np.round(a / step) * step).astype(_np.float32))
    return out


def pad_batch(chunks: list[np.ndarray], level: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length chunks to a common length divisible by 2**level.

    Returns (batch (n_chunks, L) float32 zero-padded, lengths (n_chunks,)).
    A single common length keeps the batched GPU DWT to one call; chunk lengths
    here are similar (~36k) so padding waste is small.
    """
    lengths = np.array([len(c) for c in chunks])
    step = 1 << level
    L = int(np.ceil(lengths.max() / step) * step)
    batch = np.zeros((len(chunks), L), np.float32)
    for i, c in enumerate(chunks):
        batch[i, :len(c)] = c
    return batch, lengths


def deslice_side(path: str | Path, event_key: str, side: str, config: OpticalConfig):
    """Dense (n_pmts, n_bins) pedestal-subtracted array for one side (viz only)."""
    with h5py.File(path, "r") as f:
        g = f[event_key][side]
        adc = g["adc"][:].astype(np.float32)
        offsets = g["offsets"][:]
        t0_ns = g["t0_ns"][:].astype(np.float64)
        pmt_id = g["pmt_id"][:]
    lens = np.diff(offsets)
    t0 = t0_ns.min()
    start = np.round((t0_ns - t0) / config.tick_ns).astype(np.int64)
    n_bins = int((start + lens).max())
    dense = np.full((config.n_pmts_per_side, n_bins), config.pedestal, np.float32)
    for k in range(len(pmt_id)):
        d = adc[offsets[k]:offsets[k + 1]]
        dense[int(pmt_id[k]), start[k]:start[k] + len(d)] = d
    return dense - config.pedestal, float(t0)
