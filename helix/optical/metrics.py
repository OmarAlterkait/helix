"""Reconstruction-quality metrics for optical chunk sparsification.

Computed per chunk over the VALID (unpadded) region only, then summarized.
Charge proxy = sum|adc| (∝ photoelectrons via gain); peak = prompt-pulse
amplitude (most-negative sample). All relative to the input chunk.
"""
from __future__ import annotations

import numpy as np


def chunk_metrics(x_batch, recon_batch, lengths, signal_peak_min: float = 50.0) -> dict:
    """x_batch, recon_batch: (n_chunks, L) numpy; lengths: (n_chunks,).

    Vectorized over chunks (length mask zeros the padded tail). rel_rms is over
    all chunks; peak/area errors are averaged over SIGNAL chunks only
    (|x|max > signal_peak_min ADC) — on noise-only chunks the "peak" is
    meaningless and would otherwise dominate the average.
    """
    x = np.asarray(x_batch, np.float64)
    r = np.asarray(recon_batch, np.float64)
    L = x.shape[1]
    mask = np.arange(L)[None, :] < np.asarray(lengths)[:, None]
    xm, rm = x * mask, r * mask
    nx = np.sqrt((xm ** 2).sum(1)) + 1e-9
    rel_rms = np.sqrt(((rm - xm) ** 2).sum(1)) / nx
    ax, ar = np.abs(xm).sum(1), np.abs(rm).sum(1)
    area_err = np.abs(ar - ax) / (ax + 1e-9)
    pk = xm.min(1)
    peak_err = np.abs(rm.min(1) - pk) / (np.abs(pk) + 1e-9)
    sig = np.abs(xm).max(1) > signal_peak_min          # signal chunks
    sel = sig if sig.any() else np.ones(len(lengths), bool)
    return dict(
        rel_rms=float(rel_rms.mean()), rel_rms_med=float(np.median(rel_rms)),
        area_err=float(area_err[sel].mean()), peak_err=float(peak_err[sel].mean()),
        n_chunks=int(len(lengths)), n_signal=int(sig.sum()),
    )
