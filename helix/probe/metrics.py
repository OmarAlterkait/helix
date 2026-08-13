"""Probe scoring: per-(event, plane) correlation, Fisher-z averaged.

The metric both probes report. Correlation is computed WITHIN each
``(event, plane)`` group and only then averaged, which is the point: a global
correlation would be dominated by coarse between-event and between-plane
structure that any position-only baseline reproduces. Scoring within a group
removes that, so what is left is per-pixel content.

Averaging is in Fisher-z (``arctanh``) space because correlations are not
additive; the mean is transformed back with ``tanh``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fisher_r"]


def fisher_r(y, pred, event, plane, min_rows=100):
    """``(r, per_group_rs, info)`` — Fisher-z mean of per-(event, plane) Pearson r.

    Groups with fewer than ``min_rows`` rows, or with no variance in either
    ``y`` or ``pred``, cannot yield a correlation and are skipped. ``info``
    REPORTS how many were skipped and why: the reference dropped them silently,
    so lowering a per-event cap could change the metric without changing
    anything visible.
    """
    y = np.asarray(y, np.float64)
    pred = np.asarray(pred, np.float64)
    event = np.asarray(event)
    plane = np.asarray(plane)

    rs, n_small, n_flat = [], 0, 0
    for e in np.unique(event):
        in_e = event == e
        for g in np.unique(plane[in_e]):
            s = in_e & (plane == g)
            if s.sum() < min_rows:
                n_small += 1
                continue
            if y[s].std() <= 1e-12 or pred[s].std() <= 1e-12:
                n_flat += 1
                continue
            rs.append(np.corrcoef(y[s], pred[s])[0, 1])

    rs = np.asarray(rs, np.float64)
    info = dict(n_groups=int(rs.size), n_skipped_small=int(n_small),
                n_skipped_flat=int(n_flat), min_rows=int(min_rows))
    if rs.size == 0:
        return float("nan"), rs, info
    z = np.arctanh(np.clip(rs, -0.999, 0.999))
    return float(np.tanh(z.mean())), rs, info
