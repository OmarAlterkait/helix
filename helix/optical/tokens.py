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


def waveform_tokens(wf):
    """One waveform of :func:`helix.optical.shard.read_optical` -> dict(kind, first_sample,
    values (T, 128) float32, mask (T, 128) bool, sigma (T, 128) float32, base)."""
    lev = wf["level"]
    per = layout(lev)
    n_co = -(-wf["length"] // (1 << lev))
    co = wf["coarse"][:n_co].astype(np.float64)
    base = float(np.median(co)) if n_co else 0.0
    sig_lev = np.asarray(wf["sigma_level"], np.float32)                 # levels coarse..fine
    nb = -(-n_co // per[0])
    pad = lambda a, k, v=0: np.r_[a, np.full(nb * k - len(a), v, a.dtype)]  # noqa: E731
    bands = [pad(co - base, per[0])] + [pad(d.astype(np.float64), k) for d, k in zip(wf["details"], per[1:])]
    masks = [pad(np.ones(n_co, bool), per[0], False)] + [pad(s, k, False) for s, k in zip(wf["stored"], per[1:])]
    sig = np.concatenate([np.full(per[0], sig_lev[0])] + [np.full(k, sig_lev[i]) for i, k in enumerate(per[1:])])
    V = np.concatenate([b.reshape(nb, k) for b, k in zip(bands, per)], 1)
    M = np.concatenate([m.reshape(nb, k) for m, k in zip(masks, per)], 1)
    blk = np.nonzero(M[:, per[0]:].any(1))[0]
    vs, ms, ss, kind, first = [V[blk]], [M[blk]], [np.broadcast_to(sig, (len(blk), SLOTS))], \
        [np.zeros(len(blk), np.int8)], [blk << TOKEN]
    for c0 in range(0, n_co, SLOTS):                                    # coarse tokens
        k = min(SLOTS, n_co - c0)
        vs.append(np.r_[co[c0:c0 + k] - base, np.zeros(SLOTS - k)][None])
        ms.append((np.arange(SLOTS) < k)[None]); ss.append(np.full((1, SLOTS), sig_lev[0]))
        kind.append(np.ones(1, np.int8)); first.append(np.array([c0 << lev]))
    M = np.concatenate(ms)
    return dict(kind=np.concatenate(kind), first_sample=np.concatenate(first).astype(np.int64),
                values=np.where(M, np.concatenate(vs), 0).astype(np.float32), mask=M,
                sigma=np.concatenate(ss).astype(np.float32), base=base)


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
    from helix.optical.shard import read_optical
    out = {k: [] for k in ("waveform", "channel", "t_us", "kind", "first_sample", "values", "mask", "sigma")}
    for w, wf in enumerate(read_optical(path, system, event)):
        tk = waveform_tokens(wf)
        k = len(tk["kind"])
        out["waveform"].append(np.full(k, w)); out["channel"].append(np.full(k, wf["channel"]))
        out["t_us"].append(np.full(k, wf["t_us"]))
        for key in ("kind", "first_sample", "values", "mask", "sigma"):
            out[key].append(tk[key])
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}
