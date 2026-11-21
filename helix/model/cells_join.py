"""The join from a fine cell (:mod:`helix.data.cells`) to the tokens covering it.

The evaluation's frozen probes (:mod:`helix.eval`) and the trainable denoising head
(:mod:`helix.model.denoise`, labels from :mod:`helix.data.denoise`) read exactly these features, so
a frozen and a fine-tuned number differ only in what was trained. Next to the tokenizer whose cell
keys and patch geometry it inverts.

:func:`covering_tokens`, :func:`cell_inputs` and :func:`cell_neighbors` take numpy arrays or torch
tensors (:mod:`helix.core.xp`) and answer in kind: numpy in the DataLoader workers that build the
training labels (:mod:`helix.data.denoise`), tensors on the event's device at inference
(:func:`helix.eval.denoise.load_denoiser`). One implementation; the integer outputs agree exactly
between the two, ``aux`` to float32 rounding (tests/test_cells_join_xp.py).

Every function takes the cell ``grid`` (:class:`helix.data.cells.Grid`) the keys are on: it is
the truth's (``grid_of(cells_record)``), never a constant of this module.
"""
from __future__ import annotations

import numpy as np

from helix.core.xp import namespace
from helix.data.cells import Grid, fkey, unkey
from helix.model.tokenize import cell_key as pack, pixel_cells, sigma_for_rows, tick_of_tau, unpack_cell_key


def _coeff_keys(band, plane_gid, wire, tau, cfg, grid):
    """``(k, key)``: which coefficients are in the tokenized bands, and their cells on ``grid``."""
    grid = Grid(*grid)
    band, plane_gid, wire, tau = (np.asarray(x) for x in (band, plane_gid, wire, tau))
    k = band < cfg.n_bands
    ct = tick_of_tau(tau[k], plane_gid[k], band[k], cfg)
    return k, fkey(plane_gid[k], wire[k] // grid.fw, np.floor(ct / grid.ft).astype(np.int64))


def coeff_cells(band, plane_gid, wire, tau, cfg, grid):
    """Cell keys on ``grid`` (sorted, unique) of an event's kept coefficients in the tokenized
    bands."""
    return np.unique(_coeff_keys(band, plane_gid, wire, tau, cfg, grid)[1])


def coeff_amplitude(c, cfg, grid):
    """``(keys, amp)``: :func:`coeff_cells` of a coeff part ``c`` (``band, plane_gid, wire, tau,
    value`` and its ``_meta``) and, per cell, the largest ``|value| / sigma`` of the kept
    coefficients in it -- the compressor's own detection statistic, in the units the tokenizer
    normalises by (:func:`helix.model.tokenize.sigma_for_rows`)."""
    k, key = _coeff_keys(c["band"], c["plane_gid"], c["wire"], c["tau"], cfg, grid)
    m = c["_meta"]
    band, g = np.asarray(c["band"])[k], np.asarray(c["plane_gid"])[k]
    a = np.abs(np.asarray(c["value"], np.float64).reshape(-1)[k]) \
        / np.maximum(sigma_for_rows(g, band, m["gids"], m["norm_sigma"]), 1e-6)
    keys, inv = np.unique(key, return_inverse=True)
    amp = np.zeros(len(keys))
    np.maximum.at(amp, inv.ravel(), a)
    return keys, amp


def _fourier(x, xp):
    ang = 2 * np.pi * x[..., None] * xp.asarray([1, 2, 3], xp.float32)
    return xp.concat([xp.sin(ang), xp.cos(ang)], -1).reshape(x.shape[0], -1)


def _place(keys, band_lengths, cfg, grid):
    """``(xp, keys, g, w, t, pc)``: a cell's plane, first wire and centre tick, and its covering
    patches' cell keys (n, n_bands) through the tokenizer's own :func:`pixel_cells`."""
    xp = namespace(keys)
    grid = Grid(*grid)
    keys = xp.asarray(keys, xp.int64)
    g, fw, ft = unkey(keys)
    w, t = fw * grid.fw, ft * grid.ft + grid.ft // 2
    return xp, keys, g, w, t, pixel_cells(g, w, t, band_lengths, cfg)


def _rows(xp, cell_key):
    """``row(k)``: the row in ``cell_key`` (the tokens' keys, unique) of each key of ``k``, -1
    where it is absent."""
    ck = xp.asarray(cell_key, xp.int64)
    order = xp.argsort(ck)
    cks = ck[order]

    def row(k):
        if not xp.size(cks):
            return xp.full(tuple(k.shape), -1, xp.int64)
        pos = xp.clip(xp.searchsorted(cks, k), 0, xp.size(cks) - 1)
        return xp.where(cks[pos] == k, order[pos], -1)
    return row


def covering_tokens(keys, cell_key, band_lengths, cfg, grid):
    """Per cell (``keys``, on ``grid``), the row of the token covering it in each band, -1 where
    none: ``(n, n_bands)`` int64 -- :func:`cell_inputs`' ``idx`` alone. A cell is placed by its
    first wire and its centre tick, through the tokenizer's own :func:`pixel_cells`."""
    xp, _, _, _, _, pc = _place(keys, band_lengths, cfg, grid)
    return _rows(xp, cell_key)(pc)


def cell_inputs(keys, cell_key, band_lengths, cfg, grid):
    """Per cell: which token covers it in each band, and where inside it.

    ``keys``: cell keys (:func:`fkey`) on ``grid``; a cell is placed at its first wire and its
    centre tick. ``cell_key``: the tokenized event's per-token cell
    keys (``B["cell_key"]``). Returns ``idx`` (n, n_bands) int64 -- the covering token's row,
    -1 if none -- and ``aux`` (n, 13 * n_bands) float32: a presence bit per band, then the cell's
    continuous offset inside each band's token (wire, tau), Fourier-encoded. The frozen probes
    (:mod:`helix.eval`) and the trainable denoising head (:mod:`helix.model.denoise`) both read
    exactly this, so a frozen and a fine-tuned number differ only in what was trained.
    """
    nb, pw, pt = cfg.n_bands, cfg.pw, cfg.pt
    xp, keys, g, w, t, pc = _place(keys, band_lengths, cfg, grid)
    idx = _rows(xp, cell_key)(pc)
    dec = [float(1 << int(v)) for v in cfg.lev]
    toff = xp.asarray(cfg.toff, xp.float64)[g % 3]
    tf = xp.astype(t, xp.float64) + toff
    wo = (xp.astype(w % pw, xp.float64) + 0.5) / pw
    offs = []
    for b in range(nb):
        tau = tf / dec[b] - cfg.delta[b]
        offs += [wo, (tau / pt) % 1.0]
    aux = xp.concat([xp.astype(idx >= 0, xp.float32), _fourier(xp.astype(xp.stack(offs, 1), xp.float32), xp)], 1)
    return idx, aux


def cell_neighbors(keys, cell_key, band_lengths, cfg, grid, radius=1):
    """Per cell (keys on ``grid``) and band: the tokens of the (2r+1) x (2r+1) patches around the one covering
    it, -> (n, n_bands, (2r+1)**2) int64, -1 where no token.

    Found by shifting the covering PATCH's block coordinates, so a cell with no token of its own
    in a band still sees the tokens beside it. Slot ``(2r+1)**2 // 2`` is the covering token, the
    same index :func:`cell_inputs` returns. Order: wire offset major, tick offset minor, each from
    -r to r.
    """
    xp, _, _, _, _, pc = _place(keys, band_lengths, cfg, grid)
    pg, pb, wb, tb = unpack_cell_key(pc.reshape(-1))
    row = _rows(xp, cell_key)
    cols = []
    for dw in range(-radius, radius + 1):
        for dt in range(-radius, radius + 1):
            w2, t2 = wb + dw, tb + dt
            ok = (w2 >= 0) & (t2 >= 0)
            # a block before the plane's first (w2 or t2 = -1) has no token: packed at 0, then dropped
            k2 = pack(pg, pb, xp.clip(w2, 0, None) * cfg.pw, xp.clip(t2, 0, None) * cfg.pt, cfg)
            cols.append(xp.where(ok, row(k2), -1))
    return xp.stack(cols, 1).reshape(pc.shape[0], pc.shape[1], len(cols))
