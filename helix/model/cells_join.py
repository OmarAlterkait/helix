"""The join from a fine cell (:mod:`helix.data.cells`) to the tokens covering it.

The evaluation's frozen probes (:mod:`helix.eval`) and the trainable denoising head
(:mod:`helix.model.denoise`, labels from :mod:`helix.data.denoise`) read exactly these features, so
a frozen and a fine-tuned number differ only in what was trained. numpy only; next to the
tokenizer whose cell keys and patch geometry it inverts.
"""
from __future__ import annotations

import numpy as np

from helix.data.cells import FT, FW, fkey, unkey
from helix.model.tokenize import cell_key as pack, pixel_cells, sigma_for_rows, tick_of_tau, unpack_cell_key


def _coeff_keys(band, plane_gid, wire, tau, cfg):
    """``(k, key)``: which coefficients are in the tokenized bands, and their fine cells."""
    band, plane_gid, wire, tau = (np.asarray(x) for x in (band, plane_gid, wire, tau))
    k = band < cfg.n_bands
    ct = tick_of_tau(tau[k], plane_gid[k], band[k], cfg)
    return k, fkey(plane_gid[k], wire[k] // FW, np.floor(ct / FT).astype(np.int64))


def coeff_cells(band, plane_gid, wire, tau, cfg):
    """Fine-cell keys (sorted, unique) of an event's kept coefficients in the tokenized bands."""
    return np.unique(_coeff_keys(band, plane_gid, wire, tau, cfg)[1])


def coeff_amplitude(c, cfg):
    """``(keys, amp)``: :func:`coeff_cells` of a coeff part ``c`` (``band, plane_gid, wire, tau,
    value`` and its ``_meta``) and, per cell, the largest ``|value| / sigma`` of the kept
    coefficients in it -- the compressor's own detection statistic, in the units the tokenizer
    normalises by (:func:`helix.model.tokenize.sigma_for_rows`)."""
    k, key = _coeff_keys(c["band"], c["plane_gid"], c["wire"], c["tau"], cfg)
    m = c["_meta"]
    band, g = np.asarray(c["band"])[k], np.asarray(c["plane_gid"])[k]
    a = np.abs(np.asarray(c["value"], np.float64).reshape(-1)[k]) \
        / np.maximum(sigma_for_rows(g, band, m["gids"], m["norm_sigma"]), 1e-6)
    keys, inv = np.unique(key, return_inverse=True)
    amp = np.zeros(len(keys))
    np.maximum.at(amp, inv.ravel(), a)
    return keys, amp


def _fourier(x):
    ang = 2 * np.pi * x[..., None] * np.array([1, 2, 3], np.float32)
    return np.concatenate([np.sin(ang), np.cos(ang)], -1).reshape(x.shape[0], -1)


def cell_inputs(keys, cell_key, band_lengths, cfg):
    """Per fine cell: which token covers it in each band, and where inside it.

    ``keys``: fine-cell keys (:func:`fkey`). ``cell_key``: the tokenized event's per-token cell
    keys (``B["cell_key"]``). Returns ``idx`` (n, n_bands) int64 -- the covering token's row,
    -1 if none -- and ``aux`` (n, 13 * n_bands) float32: a presence bit per band, then the cell's
    continuous offset inside each band's token (wire, tau), Fourier-encoded. The frozen probes
    (:mod:`helix.eval`) and the trainable denoising head (:mod:`helix.model.denoise`) both read
    exactly this, so a frozen and a fine-tuned number differ only in what was trained.
    """
    nb, pw, pt = cfg.n_bands, cfg.pw, cfg.pt
    g, fw, ft = unkey(np.asarray(keys, np.int64))
    w, t = fw * FW, ft * FT + FT // 2
    pc = pixel_cells(g, w, t, band_lengths, cfg)
    ck = np.asarray(cell_key, np.int64)
    order = np.argsort(ck); cks = ck[order]
    if len(cks):
        pos = np.clip(np.searchsorted(cks, pc), 0, len(cks) - 1)
        idx = np.where(cks[pos] == pc, order[pos], -1)
    else:
        idx = np.full(pc.shape, -1, np.int64)
    dec = (1 << np.asarray(cfg.lev)).astype(np.float64)
    toff = np.asarray(cfg.toff)[g % 3]
    offs = []
    for b in range(nb):
        tau = (t + toff) / dec[b] - cfg.delta[b]
        offs += [(w % pw + 0.5) / pw, (tau / pt) % 1.0]
    aux = np.concatenate([(idx >= 0).astype(np.float32), _fourier(np.stack(offs, 1).astype(np.float32))], 1)
    return idx.astype(np.int64), aux.astype(np.float32)


def cell_neighbors(keys, cell_key, band_lengths, cfg, radius=1):
    """Per fine cell and band: the tokens of the (2r+1) x (2r+1) patches around the one covering
    it, -> (n, n_bands, (2r+1)**2) int64, -1 where no token.

    Found by shifting the covering PATCH's block coordinates, so a cell with no token of its own
    in a band still sees the tokens beside it. Slot ``(2r+1)**2 // 2`` is the covering token, the
    same index :func:`cell_inputs` returns. Order: wire offset major, tick offset minor, each from
    -r to r.
    """
    g, fw, ft = unkey(np.asarray(keys, np.int64))
    pc = pixel_cells(g, fw * FW, ft * FT + FT // 2, band_lengths, cfg)          # (n, nb)
    pg, pb, wb, tb = unpack_cell_key(pc.ravel())
    ck = np.asarray(cell_key, np.int64)
    order = np.argsort(ck); cks = ck[order]
    offs = [(dw, dt) for dw in range(-radius, radius + 1) for dt in range(-radius, radius + 1)]
    out = np.full((pc.size, len(offs)), -1, np.int64)
    for j, (dw, dt) in enumerate(offs):
        w2, t2 = wb + dw, tb + dt
        ok = (w2 >= 0) & (t2 >= 0)
        if not ok.any() or not len(cks):
            continue
        k2 = pack(pg[ok], pb[ok], w2[ok] * cfg.pw, t2[ok] * cfg.pt, cfg)
        pos = np.clip(np.searchsorted(cks, k2), 0, len(cks) - 1)
        out[np.nonzero(ok)[0], j] = np.where(cks[pos] == k2, order[pos], -1)
    return out.reshape(pc.shape[0], pc.shape[1], len(offs))

