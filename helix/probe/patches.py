"""Join per-pixel truth to per-cell features, at patch granularity.

A "patch" here is a distinct 4-band cell TUPLE — the set of cells covering a
pixel. Pixels sharing a tuple share every feature the probe can see, so they are
one row; scoring them separately would just weight the metric by how many pixels
happen to fall in a patch.

The label is the charge-weighted mean ``u`` over that patch's DOMINANT pixels
(``f_top >= dom_threshold``), and patches with no dominant pixel are dropped.
That is why ``qtot_min`` and ``dom_threshold`` are target parameters rather than
sample filters, and why both must be echoed into every results row.
"""

from __future__ import annotations

import numpy as np

__all__ = ["patch_rows", "GEO_COLUMNS"]

#: What the geometry-only arm sees: plane one-hot, mean wire, mean tick, and the
#: per-band presence bits. Deliberately includes drift time — so any leakage of
#: ``u`` through timing is already in the FLOOR, and a trained model's gain over
#: it is the genuinely cross-plane part.
GEO_COLUMNS = "plane_onehot[6] + wire_mean + tick_mean + presence[n_bands]"


def patch_rows(cell_rows, plane_gid, wire, tick, qtot, ftop, u, *,
               dom_threshold=0.5, n_planes=6, wire_scale=2000.0,
               tick_scale=4321.0):
    """Reduce pixels to patch rows.

    Returns a dict with ``cells`` (the per-row cell tuple, -1 where absent),
    ``y``, ``geo``, ``plane``, ``n_pixels`` and ``n_dom``. Feature arms are built
    from ``cells`` by :func:`helix.probe.features.gather_cell_features`, so this
    module never touches a model.
    """
    cell_rows = np.asarray(cell_rows, np.int64)
    plane_gid = np.asarray(plane_gid, np.int64)
    qtot = np.asarray(qtot, np.float64)
    ftop = np.asarray(ftop, np.float64)
    u = np.asarray(u, np.float64)
    n_bands = cell_rows.shape[1]

    uniq, grp = np.unique(cell_rows, axis=0, return_inverse=True)
    ng = len(uniq)

    dom = ftop >= dom_threshold
    w = qtot * dom                                   # dominant pixels only
    wsum = np.zeros(ng); usum = np.zeros(ng); ndom = np.zeros(ng)
    np.add.at(wsum, grp, w)
    np.add.at(usum, grp, w * u)
    np.add.at(ndom, grp, dom.astype(float))

    keep = ndom > 0                                  # no dominant pixel -> no label
    y = np.zeros(ng)
    y[keep] = usum[keep] / np.maximum(wsum[keep], 1e-12)

    npix = np.bincount(grp, minlength=ng).astype(np.int64)
    sw = np.zeros(ng); st = np.zeros(ng)
    np.add.at(sw, grp, np.asarray(wire, np.float64))
    np.add.at(st, grp, np.asarray(tick, np.float64))
    wmean = sw / np.maximum(npix, 1)
    tmean = st / np.maximum(npix, 1)

    # A patch's plane: every pixel in it shares one, since plane is part of the
    # cell key -- with ONE exception per event. Pixels that matched no cell in
    # any band all carry the tuple (-1,-1,-1,-1), so they group together
    # regardless of plane: measured, that single row pools ~34k pixels spanning
    # all six planes and both volumes, with all-zero features. It is 1 row of
    # ~6400 and inflates every arm equally (+0.0015), but it is not a patch and
    # its `plane` label is meaningless.
    #
    # `first[grp[order]] = order` is LAST-write-wins, so this took the last
    # pixel while the comment said first. Immaterial for real patches (one plane
    # each) and arbitrary for the all-miss row either way -- but a label that
    # contradicts its own docstring is how the all-miss row went unnoticed.
    # Take the first, as documented, and surface the exception.
    order = np.argsort(grp, kind="stable")
    first = np.full(ng, -1, np.int64)
    first[grp[order][::-1]] = order[::-1]            # reversed => FIRST wins
    plane = plane_gid[first]

    # Which groups actually straddle planes. Callers can drop them; nothing is
    # dropped here, so the row count stays comparable with earlier runs.
    pmin = np.full(ng, np.iinfo(np.int64).max, np.int64)
    pmax = np.full(ng, -1, np.int64)
    np.minimum.at(pmin, grp, np.asarray(plane_gid, np.int64))
    np.maximum.at(pmax, grp, np.asarray(plane_gid, np.int64))
    multiplane = pmin != pmax

    presence = (uniq >= 0).astype(np.float32)
    geo = np.concatenate([
        np.eye(n_planes, dtype=np.float32)[plane],
        (wmean / wire_scale)[:, None].astype(np.float32),
        (tmean / tick_scale)[:, None].astype(np.float32),
        presence,
    ], 1)

    return dict(cells=uniq[keep], y=y[keep].astype(np.float32),
                geo=geo[keep], plane=plane[keep].astype(np.int64),
                n_pixels=npix[keep], n_dom=ndom[keep].astype(np.int64),
                # True where a row's pixels do NOT share one plane -- i.e. the
                # all-miss row, whose `plane` label is arbitrary. Expect exactly
                # one per event, or zero if it had no dominant pixel.
                multiplane=multiplane[keep], n_bands=n_bands)
