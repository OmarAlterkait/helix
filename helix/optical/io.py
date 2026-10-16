"""HDF5 I/O for goop two-sided (east/west) optical light files.

The wavelet path operates on the STORED chunks (goop's gap-compressed
"stitches") directly — no deslicing — since each chunk is already a contiguous,
pedestal-baselined active segment. There was a ``deslice_side`` here that
rebuilt the dense view "for visualization"; nothing called it, in helix or in
pimm-data, so it went with the rest of the unused surface.

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


def chunk_noise_sigma(chunks: list[np.ndarray], method: str = "mad") -> np.ndarray:
    """Per-chunk noise sigma from the UNPADDED chunk's finest db1 detail.

    Padding-independent — used for the VisuShrink threshold so zero-padding in
    the batch can't collapse the estimate. Returns (n_chunks,) float32.

    ``method="mad"`` (default, unchanged): median(|d|)/0.6745. On INTEGER ADC codes
    this is quantised: the db1 detail takes values k/sqrt(2), so the estimate moves
    in steps of 1/(sqrt(2) 0.6745) = 1.048 ADC, and it is exactly 0 when the noise
    is below ~0.6 ADC (measured: every SBND X-ARAPUCA waveform, sigma = 0.46 ADC,
    returns 0 — the threshold vanishes and nothing is compressed).

    ``method="clipped"``: iterative 3.5-sigma clipped RMS of the same detail band,
    corrected for the Gaussian truncation — continuous, unbiased on digitised
    Gaussian noise down to sigma ~ 0.45 ADC. Vectorised over chunks.
    """
    import pywt
    if method == "mad":
        # the original per-chunk pywt path, kept verbatim so the default is bit-identical
        out = []
        for c in chunks:
            d = pywt.wavedec(np.asarray(c, np.float64), "db1", level=1, mode="periodization")[-1]
            out.append(np.median(np.abs(d)) / 0.6745)
        return np.asarray(out, np.float32)
    if method != "clipped":
        raise ValueError(f"unknown noise-sigma method {method!r} (mad|clipped)")
    n = np.array([len(c) // 2 for c in chunks])
    D = np.zeros((len(chunks), max(int(n.max()), 1)), np.float64)
    pad = np.arange(D.shape[1])[None, :] >= n[:, None]
    for i, c in enumerate(chunks):
        c = np.asarray(c, np.float64)[: 2 * n[i]]
        D[i, : n[i]] = (c[0::2] - c[1::2]) / np.sqrt(2.0)
    return _clipped_rms(D, ~pad).astype(np.float32)


def _clipped_rms(D, valid, k: float = 3.5, n_iter: int = 6):
    """Row-wise zero-centred clipped RMS with truncation correction."""
    from math import erf, exp, pi, sqrt
    corr = sqrt(1.0 - 2 * k * exp(-0.5 * k * k) / sqrt(2 * pi) / erf(k / sqrt(2)))
    q = np.where(valid, np.abs(D), np.nan)
    sd = np.maximum(np.nanpercentile(q, 50, axis=1) / 0.6745, 0.3)
    for _ in range(n_iter):
        sel = valid & (np.abs(D) <= k * sd[:, None])
        nn = np.maximum(sel.sum(1), 2)
        sd = np.maximum(np.sqrt((np.where(sel, D, 0.0) ** 2).sum(1) / nn) / corr, 1e-6)
    return sd


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
