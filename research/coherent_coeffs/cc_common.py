"""Shared helpers for the coherent-coefficient structure study.

Goal: decompose (per wire, no thresholding) the three additive components of a
noisy wire plane SEPARATELY and study how each sits in wavelet space.

  signal     = clean doraemon truth (pedestal-subtracted ADC, noise-free)
  coherent   = group-correlated noise (tools.coherent_noise): each wire in a
               64-wire block shares the EXACT same waveform; adjacent blocks
               are anti-correlated (beta coupling -> "bleeding").
  intrinsic  = per-wire independent colored+white noise (JAXTPC model)

Production "best case" transform: coif3, level 4, periodization (config default).
Reuses the data/noise generators from research.wire_denoise.common so the
realizations match the rest of the study.
"""
import os
import sys

import numpy as np
import pywt

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '..', 'wire_denoise'))
import common as _wd  # noqa: E402  (research.wire_denoise.common)

GROUP_SIZE = 64
WAVELET = 'coif3'
LEVEL = 4
MODE = 'periodization'
FIGDIR = os.path.join(_HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

PLANES = _wd.PLANES


def components(ptype, event, seed=1):
    """Return (signal, coherent, intrinsic) each (n_wires, N_TICKS) float32.

    Uses the same per-event seeding as common.make_noisy so the components are
    exactly the realization that would be summed into a noisy image:
        intrinsic rng = seed*100003 + event_index
        coherent  rng = seed*100003 + event_index + 7919
    Here event_index is 0 because we load a single event as a length-1 stack."""
    info = PLANES[ptype]
    nw = info['n_wires']
    signal = _wd.load_clean([event], ptype)[0]                     # (nw, T)
    rng_i = np.random.default_rng(seed * 100003 + 0)
    rng_c = np.random.default_rng(seed * 100003 + 0 + 7919)
    intrinsic = _wd.intrinsic_noise(nw, ptype, rng_i)
    coherent = _wd.coherent_noise(nw, rng_c)
    return signal.astype(np.float32), coherent.astype(np.float32), intrinsic.astype(np.float32)


def dwt_bands(img, wavelet=WAVELET, level=LEVEL, mode=MODE):
    """Per-wire DWT -> list [cA_L, cD_L, ..., cD_1], each (n_wires, len_band).

    No thresholding. img is (n_wires, T)."""
    return pywt.wavedec(np.asarray(img, np.float32), wavelet, level=level, mode=mode, axis=-1)


def band_labels(level=LEVEL):
    return [f'A{level}'] + [f'D{j}' for j in range(level, 0, -1)]


def flat_coeffs(bands):
    """Concatenate a list of (nw, len_j) bands into one (nw, n_coeffs) array."""
    return np.concatenate(bands, axis=-1)


def band_slices(bands):
    """List of (start, stop) slices locating each band inside flat_coeffs."""
    out, p = [], 0
    for b in bands:
        out.append((p, p + b.shape[-1]))
        p += b.shape[-1]
    return out


def n_groups(nw, gs=GROUP_SIZE):
    return (nw + gs - 1) // gs


def full_groups(nw, gs=GROUP_SIZE):
    """Number of COMPLETE 64-wire groups (drops the partial tail group)."""
    return nw // gs


def mad_sigma(x, axis=-1):
    return np.median(np.abs(x), axis=axis) / 0.6745
