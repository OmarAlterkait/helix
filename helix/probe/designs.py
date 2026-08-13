"""The two probes, which differ only in what the head is allowed to see.

``mlp``          single-plane. Each row sees its own patch's band features.
                 Arms: geo / raw / random / trained. Answers: is ``u`` present
                 above the geometry floor?
``triangulate``  adds drift-time-matched context from the OTHER two planes — the
                 epipolar constraint. Arms: solo / cross / xwire. Answers: if
                 ``u`` is weak single-plane, is the information gone or merely
                 un-combined?

Every arm is ``[arm features | geo]``, so ``geo`` is the common floor and an
arm's gain over it is the part that needed the representation.

Two deliberate choices:

* **Matching uses the drift tick, never the true wire.** The tick is an
  observable; the wire is (most of) the answer. ``xwire`` — which DOES use the
  partner's true wire — is therefore a ceiling, not an arm to compare models on.
* **Rows are patches, not pixels.** The reference's ``triangulate`` worked on
  capped dominant pixels while its ``mlp`` worked on patches, so the two probes
  scored different row sets. Pixels sharing a cell tuple are indistinguishable
  to the probe, so patches are the honest unit and it makes the two comparable.

``cross`` mean-pools the matched slab. ``probe_3d_select`` hypothesised this was
crippling (RoPE features do not average) and tested selecting the single
drift-nearest partner instead — it scored WORSE on both models tried (0.1077 vs
0.1342, and its own ceiling fell from 0.6979 to 0.6244). The hypothesis was
refuted, so mean-pooling is kept and selection is not implemented.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mlp_designs", "triangulate_designs", "slab_context"]


def mlp_designs(geo, arms):
    """``{name: X}`` for the single-plane probe. ``arms`` maps name -> features."""
    geo = np.asarray(geo, np.float32)
    out = {"geo": geo}
    for name, F in arms.items():
        if F is None:
            continue
        F = np.asarray(F, np.float32)
        if F.shape[0] != geo.shape[0]:
            raise ValueError(f"arm {name!r} has {F.shape[0]} rows, geo has {geo.shape[0]}")
        out[name] = np.concatenate([F, geo], 1)
    return out


def slab_context(plane, tick, feats, wire, event, *, tbin=8, n_planes=6):
    """Drift-time-matched context from the other two planes of the same volume.

    A "slab" is one ``(EVENT, volume, plane, tick // tbin)`` cell. For each row we
    look up the slab at the SAME drift bin in each of the other two planes of its
    volume and return that slab's mean feature and mean true wire.

    ``event`` is REQUIRED and part of the key. Without it, slabs pool across
    events and a row receives context averaged over unrelated events — which
    destroys the correspondence the epipolar match exists to exploit. The
    reference is called once per event, so its slabs are within-event by
    construction; here the rows arrive concatenated, so the event must be in the
    key explicitly.

    Returns ``(ctx_feats, ctx_wire, hit)`` with two partners per row.
    """
    plane = np.asarray(plane, np.int64)
    tick = np.asarray(tick, np.float64)
    wire = np.asarray(wire, np.float64)
    feats = np.asarray(feats, np.float32)

    vol, pl = plane // 3, plane % 3
    tb = np.floor(tick / tbin).astype(np.int64)
    tb -= tb.min()
    ntb = int(tb.max()) + 1
    event = np.asarray(event)
    _, eidx = np.unique(event, return_inverse=True)
    n_pl = int(plane.max()) + 1
    key = ((eidx.astype(np.int64) * n_pl + plane) * ntb) + tb

    uk, inv = np.unique(key, return_inverse=True)
    ng = len(uk)
    fsum = np.zeros((ng, feats.shape[1]), np.float64)
    wsum = np.zeros(ng)
    cnt = np.zeros(ng)
    np.add.at(fsum, inv, feats.astype(np.float64))
    np.add.at(wsum, inv, wire)
    np.add.at(cnt, inv, 1.0)
    fmean = (fsum / np.maximum(cnt, 1)[:, None]).astype(np.float32)
    wmean = wsum / np.maximum(cnt, 1)

    n, fd = feats.shape[0], feats.shape[1]
    ctx = np.zeros((n, 2 * fd), np.float32)
    ctxw = np.zeros((n, 2), np.float32)
    hit = np.zeros((n, 2), np.float32)
    for i, off in enumerate((1, 2)):
        want = ((eidx * n_pl + (vol * 3 + (pl + off) % 3)) * ntb) + tb
        pos = np.searchsorted(uk, want)
        ok = (pos < ng)
        pos = np.clip(pos, 0, max(ng - 1, 0))
        ok &= (uk[pos] == want)
        if ok.any():
            ctx[ok, i * fd:(i + 1) * fd] = fmean[pos[ok]]
            ctxw[ok, i] = wmean[pos[ok]] / 2000.0
            hit[ok, i] = 1.0
    return ctx, ctxw, hit


def triangulate_designs(geo, own_feats, plane, tick, wire, event, *, tbin=8):
    """``{solo, cross, xwire}`` for the cross-plane probe.

    ``solo`` reproduces the single-plane result on the same rows, so the three
    are directly comparable. ``xwire`` replaces the partner FEATURES with their
    true mean wire — a geometric ceiling, not a model measurement.
    """
    geo = np.asarray(geo, np.float32)
    own = np.asarray(own_feats, np.float32)
    ctx, ctxw, hit = slab_context(plane, tick, own, wire, event, tbin=tbin)
    return {
        "solo": np.concatenate([own, geo], 1),
        "cross": np.concatenate([own, ctx, hit, geo], 1),
        "xwire": np.concatenate([own, ctxw, hit, geo], 1),
    }
