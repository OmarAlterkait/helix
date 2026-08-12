"""Per-pixel probe truth: derive it from the simulation, and read it back.

The probe target ``u`` is a function of ``hits`` and ``step`` and **nothing
else** — no corpus, no tokenizer, no checkpoint. That is why the truth is stored
per PIXEL ``(plane_gid, wire, tick)`` rather than per tokenizer cell: a per-cell
dump would bake in ``PatchConfig`` (which is per-CHECKPOINT — m113 trained
``grid_center`` while the dataclass default is ``centroid``) and the shard's
``band_lengths`` (which is per-CORPUS), making it valid for exactly one
checkpoint family. Per-pixel survives a corpus rebuild, a re-shard, and a change
of patch geometry.

What a pixel carries:

    qtot    total charge on it
    f_top   fraction contributed by its largest group
    b1      that group's charge-weighted 3D centroid, mm

``u`` is then ``b1`` projected onto the plane's along-wire axis
(:mod:`helix.probe.alongwire`). Storing ``b1`` rather than ``u`` keeps the axis
out of the label, so the axis can be re-verified independently.

``qtot_min`` and ``dom_threshold`` are TARGET parameters, not sample filters:
the probe label is charge-weighted over dominant pixels and patches with no
dominant pixel are dropped, so changing either changes ``y``. Both are written
into ``/config`` and must be echoed into every results row — the reference
recorded neither (``QTOT_MIN`` was an unlogged env var).
"""

from __future__ import annotations

import numpy as np

__all__ = ["PLANES", "decode_hits_plane", "group_centroids", "pixel_truth"]

#: Plane order within a volume; ``plane_gid = volume * 3 + PLANES.index(p)``.
PLANES = ("U", "V", "Y")


def decode_hits_plane(grp):
    """One ``hits`` plane group -> flat per-sample ``(wire, tick, q, group)``.

    CSR-style: each group owns ``group_sizes[i]`` consecutive samples, stored as
    offsets from its centre. ``charges_u16`` is quantised against the group's own
    peak, so it dequantises as ``q / 65535 * peak_charges[group]``.
    """
    cw = grp["center_wires"][:].astype(np.int64)
    ct = grp["center_times"][:].astype(np.int64)
    dw = grp["delta_wires"][:].astype(np.int64)
    dt = grp["delta_times"][:].astype(np.int64)
    qs = grp["charges_u16"][:].astype(np.float64)
    gid = grp["group_ids"][:].astype(np.int64)
    sz = grp["group_sizes"][:].astype(np.int64)
    pk = grp["peak_charges"][:].astype(np.float64)

    if sz.sum() != qs.shape[0]:
        raise ValueError(
            f"hits plane is inconsistent: sum(group_sizes)={sz.sum()} but "
            f"{qs.shape[0]} charge samples")
    rep = np.repeat(np.arange(len(sz)), sz)          # sample -> its group's row
    return dict(wire=cw[rep] + dw, tick=ct[rep] + dt,
                q=qs / 65535.0 * pk[rep], group=gid[rep])


def group_centroids(pos, q, deposit_to_group):
    """``{global_group: charge-weighted (x, y, z)}`` over deposits, mm.

    Segment reduction over a stable sort of ``deposit_to_group``, matching the
    reference. Groups with no charge fall back to the unweighted mean rather than
    dividing by zero.
    """
    pos = np.asarray(pos, np.float64)
    q = np.asarray(q, np.float64)
    d2g = np.asarray(deposit_to_group, np.int64)
    if d2g.shape[0] != pos.shape[0]:
        raise ValueError(
            f"deposit count mismatch: deposit_to_group has {d2g.shape[0]} "
            f"entries, step has {pos.shape[0]} positions")

    order = np.argsort(d2g, kind="stable")
    dg, p, w = d2g[order], pos[order], q[order]
    uniq, starts = np.unique(dg, return_index=True)
    ends = np.concatenate([starts[1:], [dg.shape[0]]])
    out = {}
    for g, s, e in zip(uniq, starts, ends):
        ww = w[s:e]
        tot = ww.sum()
        out[int(g)] = ((p[s:e] * ww[:, None]).sum(0) / tot if tot > 1e-12
                       else p[s:e].mean(0))
    return out


def pixel_truth(samples, centroids, qtot_min):
    """Per-sample rows -> per-PIXEL truth.

    Returns ``(wire, tick, qtot, f_top, b1)``. A pixel's ``f_top`` is the charge
    share of its largest contributing group, and ``b1`` is that group's centroid
    — so ``f_top`` measures how much a single group dominates the pixel, which is
    what makes ``b1`` a meaningful label for it.

    Pixels below ``qtot_min`` are dropped here; that is a TARGET parameter and
    the caller must record it.
    """
    wire = np.asarray(samples["wire"], np.int64)
    tick = np.asarray(samples["tick"], np.int64)
    q = np.asarray(samples["q"], np.float64)
    grp = np.asarray(samples["group"], np.int64)

    # (pixel, group) -> summed charge, then reduce over groups per pixel.
    key = np.stack([wire, tick, grp], 1)
    uniq_pg, inv_pg = np.unique(key, axis=0, return_inverse=True)
    q_pg = np.zeros(len(uniq_pg))
    np.add.at(q_pg, inv_pg, q)

    pix = uniq_pg[:, :2]
    uniq_pix, inv_pix = np.unique(pix, axis=0, return_inverse=True)
    qtot = np.zeros(len(uniq_pix))
    np.add.at(qtot, inv_pix, q_pg)

    # Largest (pixel, group) contribution per pixel: sort so the last write wins.
    order = np.lexsort((q_pg, inv_pix))
    top_group = np.empty(len(uniq_pix), np.int64)
    top_q = np.empty(len(uniq_pix))
    top_group[inv_pix[order]] = uniq_pg[order, 2]
    top_q[inv_pix[order]] = q_pg[order]

    keep = qtot >= float(qtot_min)
    if not keep.any():
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32),
                np.empty(0, np.float32), np.empty((0, 3), np.float32))

    kw, kt = uniq_pix[keep, 0], uniq_pix[keep, 1]
    kq, ktop, kgrp = qtot[keep], top_q[keep], top_group[keep]
    b1 = np.array([centroids.get(int(g), (np.nan, np.nan, np.nan)) for g in kgrp],
                  np.float64)
    good = np.isfinite(b1).all(1)          # a group with no deposits has no centroid
    return (kw[good].astype(np.int32), kt[good].astype(np.int32),
            kq[good].astype(np.float32),
            (ktop[good] / np.maximum(kq[good], 1e-12)).astype(np.float32),
            b1[good].astype(np.float32))
