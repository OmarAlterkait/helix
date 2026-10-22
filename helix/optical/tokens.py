"""Optical model tokens: 128 slots each, like a TPC token (16 wires x 8 coefficients).

A waveform stored by :mod:`helix.optical.codec` (integer Haar to ``level``, coarse sums of
2^level samples) gives two kinds of token:

  * **block** (kind 0): 128 samples with their whole coefficient tree, in this order
    ``A_level: 2^(7-level), D_level: 2^(7-level), D_(level-1): 2^(8-level), ..., D_1: 64``
    (level 6: 2 + 2 + 4 + 8 + 16 + 32 + 64 = 128 slots). One token per 128-sample block holding a
    stored difference; the tree of a block is the block's samples, re-expressed.
  * **coarse** (kind 1): 128 consecutive coarse sums of a waveform (8192 samples at level 6),
    every waveform covered: the light and baseline everywhere.

Values are raw integers with the waveform's baseline removed from the coarse slots (the median
coarse sum); ``mask`` marks the stored slots; ``sigma`` is each slot's noise (the channel's sigma of
that level; coarse slots: the level-``level`` sigma), so the model's input is
``arcsinh(values / sigma)`` as for the TPC (:func:`helix.model.tokenize.normalize_values`).
"""
from __future__ import annotations

import numpy as np

TOKEN = 7                         # log2 samples per block token (128)
SLOTS = 1 << TOKEN


def layout(level):
    """Slots per band ``[A_level, D_level, ..., D_1]`` of a block token."""
    if not 1 <= level <= TOKEN:
        raise ValueError(f"level {level}: block tokens need 1 <= level <= {TOKEN}")
    return [1 << (TOKEN - level)] + [1 << (TOKEN - j) for j in range(level, 0, -1)]


def _medians(x, counts):
    """Median of each consecutive segment of ``x`` (lengths ``counts``; 0 for an empty one)."""
    w = np.repeat(np.arange(len(counts)), counts)
    v = x[np.lexsort((x, w))]
    s = np.cumsum(counts) - counts
    lo, hi = s + np.maximum(counts - 1, 0) // 2, s + counts // 2
    return np.where(counts > 0, (v[np.minimum(lo, len(v) - 1)] + v[np.minimum(hi, len(v) - 1)]) / 2 if len(v) else 0.0, 0.0)


def event_tokens(ev):
    """All windows of :func:`helix.optical.shard.read_event` at once -> dict(waveform, kind,
    first_sample, values (T, 128) float32, mask (T, 128) bool, sigma (T, 128) float32) per token,
    each window's block tokens then its coarse tokens; base per window."""
    lev = ev["level"]
    per = layout(lev)
    off, nco = ev["offset"], ev["n_coarse"].astype(np.int64)
    W, P = len(nco), np.diff(off)
    ws, nbw = np.arange(W), P >> TOKEN                               # block tokens per window
    co = ev["coarse"].astype(np.float64)
    cw = np.repeat(ws, P >> lev)
    ci = np.arange(len(co)) - np.repeat(off[:-1] >> lev, P >> lev)   # coarse index within its window
    valid = ci < nco[cw]
    base = _medians(co[valid], nco)
    sig_lev = np.asarray(ev["sigma_level"], np.float32).reshape(W, lev)
    sig = np.concatenate([np.repeat(sig_lev[:, :1], per[0], 1)] + [np.repeat(sig_lev[:, i:i + 1], k, 1)
                                                                    for i, k in enumerate(per[1:])], 1)
    nb = int(nbw.sum())
    blk = np.unique(np.concatenate([np.flatnonzero(st) // k for st, k in zip(ev["stored"], per[1:])]))
    V = np.concatenate([np.where(valid, co - base[cw], 0).reshape(nb, per[0])[blk]]
                       + [d.reshape(nb, k)[blk] for d, k in zip(ev["details"], per[1:])], 1)
    M = np.concatenate([valid.reshape(nb, per[0])[blk]] + [st.reshape(nb, k)[blk] for st, k in zip(ev["stored"], per[1:])], 1)
    bw = np.repeat(ws, nbw)
    b_first = (blk - np.repeat(np.cumsum(nbw) - nbw, nbw)[blk]) << TOKEN
    ntc = -(-nco // SLOTS)                                           # coarse tokens per window
    t0 = np.cumsum(ntc) - ntc
    nt = int(ntc.sum())
    cv, cm = np.zeros((nt, SLOTS)), np.zeros((nt, SLOTS), bool)
    t, slot = t0[cw[valid]] + ci[valid] // SLOTS, ci[valid] % SLOTS
    cv[t, slot], cm[t, slot] = co[valid] - base[cw[valid]], True
    tw = np.repeat(ws, ntc)
    c_first = ((np.arange(nt) - t0[tw]) * SLOTS) << lev
    kind = np.r_[np.zeros(len(blk), np.int8), np.ones(nt, np.int8)]
    wav = np.r_[bw[blk], tw].astype(np.int64)
    order = np.argsort(2 * wav + kind, kind="stable")
    Mo = np.concatenate([M, cm])[order]
    return dict(waveform=wav[order], kind=kind[order], first_sample=np.r_[b_first, c_first].astype(np.int64)[order],
                values=np.where(Mo, np.concatenate([V, cv])[order], 0).astype(np.float32), mask=Mo,
                sigma=np.concatenate([sig[bw[blk]], np.repeat(sig[:, :1], SLOTS, 1)[tw]])[order].astype(np.float32),
                base=base)


def waveform_tokens(wf):
    """One waveform of :func:`helix.optical.shard.read_optical` -> dict(kind, first_sample,
    values (T, 128) float32, mask (T, 128) bool, sigma (T, 128) float32, base)."""
    lev = wf["level"]
    n_co = -(-wf["length"] // (1 << lev))
    Lw = n_co << lev
    P = -(-Lw // SLOTS) * SLOTS
    pad = lambda a, j: np.r_[np.asarray(a)[:Lw >> j], np.zeros((P - Lw) >> j, np.asarray(a).dtype)]  # noqa: E731
    lv = range(lev, 0, -1)
    tk = event_tokens(dict(level=lev, offset=np.array([0, P]), n_coarse=np.array([n_co]),
                           sigma_level=np.asarray(wf["sigma_level"], np.float32)[None], coarse=pad(wf["coarse"], lev),
                           details=[pad(d, j) for d, j in zip(wf["details"], lv)],
                           stored=[pad(st, j) for st, j in zip(wf["stored"], lv)]))
    tk.pop("waveform")
    tk["base"] = float(tk["base"][0])
    return tk


def from_tokens(tok, length, level):
    """Inverse of :func:`waveform_tokens` for one waveform -> (coarse, details, stored) as
    :func:`helix.optical.shard.read_optical` gives them (coarse sums over the stored windows)."""
    per = layout(level)
    n_co = -(-length // (1 << level))
    Lw = n_co << level
    co = np.zeros(n_co)
    det = [np.zeros(Lw >> j) for j in range(level, 0, -1)]
    st = [np.zeros(Lw >> j, bool) for j in range(level, 0, -1)]
    cut = np.cumsum([0] + per)
    for kd, f, v, m in zip(tok["kind"], tok["first_sample"], tok["values"], tok["mask"]):
        if kd == 1:
            c0 = f >> level; k = min(SLOTS, n_co - c0); co[c0:c0 + k] = v[:k] + tok["base"]
            continue
        b = f >> TOKEN
        for q, k in enumerate(per[1:]):
            n = min(k, len(det[q]) - b * k)
            det[q][b * k:b * k + n] = v[cut[q + 1]:cut[q + 1] + n]
            st[q][b * k:b * k + n] = m[cut[q + 1]:cut[q + 1] + n]
    return co, det, st


def shard_tokens(path, system, event):
    """Tokens of one event's waveforms of ``system`` -> dict(waveform, channel, t_us, kind,
    first_sample, values, mask, sigma) per token."""
    from helix.optical.shard import read_event
    ev = read_event(path, system, event)
    tk = event_tokens(ev)
    tk.pop("base")
    w = tk["waveform"]
    return dict(waveform=w, channel=ev["channel"][w], t_us=ev["t_us"][w], **{k: v for k, v in tk.items() if k != "waveform"})
