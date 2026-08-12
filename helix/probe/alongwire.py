"""The along-wire direction — a detector constant, fitted once and then VERIFIED.

``u``, the along-wire coordinate, is the degree of freedom a single wire plane
cannot determine: a wire fixes only the pitch coordinate, so recovering ``u``
requires combining planes. That is what makes it the right probe target.

Getting ``u`` needs the wire orientation, and it is **not in the geometry
registry** — ``cubic_wireplane_geometry.json`` carries only ``n_wires``,
``pedestal`` and ``wire_lengths_m``. So it has to come from a fit:
``wire ~ a*y + b*z`` over deposits whose wire is known, giving the pitch
direction ``(a, b)``; the along-wire unit vector is its perpendicular
``(-b, a)/|a,b|``.

**It is a constant.** Measured over three events from three shards
(~600k deposits):

    gid 0,3 (U)   (-0.5000, +0.8660)   = +120.00 deg
    gid 1,4 (V)   (-0.5000, -0.8660)   = -120.00 deg
    gid 2,5 (Y)   (-1.0000,  0.0000)   = +-180.00 deg
    pitch 1.51701-1.51703 wires/mm, residual 0.30 wires, every plane

Exactly the +-60/0 degree wire orientations, identical to four decimals across
events, volumes and shards. So this module FITS at dump time and thereafter
VERIFIES, rather than re-fitting per run. Two reasons:

1. The reference re-fitted on every probe run, and did it over ALL events before
   the cross-validation split (``probe_3d_ridge.py:141``) — so the target
   definition saw eval-fold data. Numerically harmless at this precision, but a
   harness leak, and freezing removes it.
2. A fit that silently moves is a corrupted target. Verification turns that into
   a loud failure.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fit_alongwire", "verify_alongwire", "u_of"]

#: Maximum deviation of a re-fit unit vector from the stored one before we refuse.
#: The spread actually measured across events/shards is < 1e-4.
TOL = 1e-3


def fit_alongwire(plane_gid, y, z, wire, n_gid=6, min_points=500):
    """Least-squares along-wire unit vectors, ``{gid: (uy, uz)}``.

    ``wire ~ a*y + b*z + c`` per plane; along-wire is perpendicular to the pitch
    direction ``(a, b)``. Returns the vectors plus per-gid diagnostics, because a
    fit is only trustworthy with its residual attached.
    """
    plane_gid = np.asarray(plane_gid, np.int64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)
    wire = np.asarray(wire, np.float64)

    vecs, diag = {}, {}
    for g in range(n_gid):
        s = plane_gid == g
        n = int(s.sum())
        if n < min_points:
            continue
        A = np.column_stack([y[s], z[s], np.ones(n)])
        coef, *_ = np.linalg.lstsq(A, wire[s], rcond=None)
        a, b = float(coef[0]), float(coef[1])
        pitch = float(np.hypot(a, b))
        if pitch <= 0:
            raise ValueError(f"gid {g}: degenerate wire fit (pitch {pitch})")
        vecs[g] = np.array([-b / pitch, a / pitch], np.float64)
        diag[g] = dict(pitch_wires_per_mm=pitch, n=n,
                       resid_rms_wires=float(np.sqrt(np.mean((A @ coef - wire[s]) ** 2))))
    if not vecs:
        raise ValueError(
            f"no plane had >= {min_points} deposits with a known wire — the "
            f"hits/step join produced nothing to fit")
    return vecs, diag


def verify_alongwire(stored, plane_gid, y, z, wire, tol=TOL, min_points=500):
    """Re-fit and compare against the frozen table; raise if anything moved.

    ``stored`` is ``(n_gid, 2)`` as written into the truth artifact's ``/config``.
    A silent move here would redefine the target midway through a study, so this
    is a hard failure rather than a warning.
    """
    stored = np.asarray(stored, np.float64)
    fresh, diag = fit_alongwire(plane_gid, y, z, wire,
                                n_gid=stored.shape[0], min_points=min_points)
    bad = {}
    for g, v in fresh.items():
        # Sign is arbitrary in the fit (u and -u are the same axis); compare the
        # axis, not the arrow, then report the signed deviation.
        d = min(float(np.linalg.norm(v - stored[g])),
                float(np.linalg.norm(v + stored[g])))
        if d > tol:
            bad[g] = dict(stored=stored[g].tolist(), refit=v.tolist(), delta=d,
                          **diag[g])
    if bad:
        raise ValueError(
            f"along-wire geometry moved beyond {tol}: {bad}. The stored table "
            f"defines the probe target, so a shift redefines what is being "
            f"measured. Re-dump the truth artifact, or investigate the source.")
    return diag


def u_of(plane_gid, b1, alongwire):
    """Project a 3D centroid onto its plane's along-wire axis. ``b1`` is (N, 3) mm.

    Returns metres, matching the reference (``pb_aw.u_target`` divides by 1000).
    """
    plane_gid = np.asarray(plane_gid, np.int64)
    b1 = np.asarray(b1, np.float64)
    aw = np.asarray(alongwire, np.float64)
    v = aw[plane_gid]                      # (N, 2), the (y, z) axis per point
    return ((b1[:, 1] * v[:, 0] + b1[:, 2] * v[:, 1]) / 1000.0).astype(np.float32)
