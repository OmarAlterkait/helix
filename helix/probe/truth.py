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

    # Vectorised segment reduction. The obvious form is a Python loop over
    # groups, but there are ~24k groups per event and that loop dominated the
    # dump at 5.8 s/event.
    uniq, inv = np.unique(d2g, return_inverse=True)
    ng = len(uniq)
    wsum = np.zeros(ng)
    np.add.at(wsum, inv, q)
    psum = np.zeros((ng, pos.shape[1]))
    np.add.at(psum, inv, pos * q[:, None])
    # Groups with no charge fall back to the unweighted mean rather than 0/0.
    cnt = np.zeros(ng)
    np.add.at(cnt, inv, 1.0)
    plain = np.zeros((ng, pos.shape[1]))
    np.add.at(plain, inv, pos)
    ok = wsum > 1e-12
    cen = np.where(ok[:, None], psum / np.maximum(wsum, 1e-12)[:, None],
                   plain / np.maximum(cnt, 1)[:, None])
    return {int(g): cen[i] for i, g in enumerate(uniq)}


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
    # Pack (wire, tick, group) into ONE int64 rather than calling np.unique on a
    # 2-D array, which sorts row-wise via a void view and is far slower.
    W_BITS, T_BITS = 20, 20
    if wire.min() < 0 or tick.min() < 0:
        raise ValueError("negative wire/tick cannot be packed")
    if wire.max() >= 1 << W_BITS or tick.max() >= 1 << T_BITS:
        raise ValueError(f"wire {wire.max()} / tick {tick.max()} overflow the "
                         f"{W_BITS}/{T_BITS}-bit packing")
    pixkey = (wire << T_BITS) | tick
    pgkey = (pixkey << 24) | np.minimum(grp, (1 << 24) - 1)
    uniq_pg, inv_pg = np.unique(pgkey, return_inverse=True)
    q_pg = np.zeros(len(uniq_pg))
    np.add.at(q_pg, inv_pg, q)

    pix_of_pg = uniq_pg >> 24
    grp_of_pg = uniq_pg & ((1 << 24) - 1)
    uniq_pix, inv_pix = np.unique(pix_of_pg, return_inverse=True)
    qtot = np.zeros(len(uniq_pix))
    np.add.at(qtot, inv_pix, q_pg)

    # Largest (pixel, group) contribution per pixel: sort so the last write wins.
    order = np.lexsort((q_pg, inv_pix))
    top_group = np.empty(len(uniq_pix), np.int64)
    top_q = np.empty(len(uniq_pix))
    top_group[inv_pix[order]] = grp_of_pg[order]
    top_q[inv_pix[order]] = q_pg[order]

    keep = qtot >= float(qtot_min)
    if not keep.any():
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32),
                np.empty(0, np.float32), np.empty((0, 3), np.float32))

    kw = (uniq_pix[keep] >> T_BITS).astype(np.int64)
    kt = (uniq_pix[keep] & ((1 << T_BITS) - 1)).astype(np.int64)
    kq, ktop, kgrp = qtot[keep], top_q[keep], top_group[keep]
    b1 = np.array([centroids.get(int(g), (np.nan, np.nan, np.nan)) for g in kgrp],
                  np.float64)
    good = np.isfinite(b1).all(1)          # a group with no deposits has no centroid
    return (kw[good].astype(np.int32), kt[good].astype(np.int32),
            kq[good].astype(np.float32),
            (ktop[good] / np.maximum(kq[good], 1e-12)).astype(np.float32),
            b1[good].astype(np.float32))
