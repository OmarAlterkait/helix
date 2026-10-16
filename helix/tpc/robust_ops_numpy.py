"""NumPy backend of :mod:`helix.tpc.robust` (the reference)."""
from __future__ import annotations

import numpy as np
from scipy.special import erf


def trunc_var_factor(k: float) -> float:
    """Var of N(0,1) truncated to [-k, k]."""
    phi = np.exp(-0.5 * k * k) / np.sqrt(2 * np.pi)
    p = erf(k / np.sqrt(2))
    return 1.0 - 2 * k * phi / p


def _row_quantiles(x, qs=(0.25, 0.5, 0.75)):
    T = x.shape[1]
    ks = [min(T - 1, int(round(q * (T - 1)))) for q in qs]
    p = np.partition(x, ks, axis=1)
    return np.stack([p[:, k] for k in ks])


def clipped_stats(x, mask=None, k=3.5, n_iter=5, center=None, sd0=None, tol=1e-3):
    x = np.asarray(x, np.float32)
    keep = None if mask is None else ~np.asarray(mask, bool)
    if center is None or sd0 is None:
        q = _row_quantiles(x)
        mu = q[1].astype(np.float32) if center is None else np.asarray(center, np.float32)
        sd = (np.maximum((q[2] - q[0]) / 1.349, 0.5) if sd0 is None else np.asarray(sd0)).astype(np.float32)
    else:
        mu, sd = np.asarray(center, np.float32).copy(), np.asarray(sd0, np.float32).copy()
    corr = np.float32(np.sqrt(trunc_var_factor(k)))
    n = np.zeros(x.shape[0], np.int64)
    for _ in range(n_iter):
        d = x - mu[:, None]
        sel = np.abs(d) <= (k * sd)[:, None]
        if keep is not None:
            sel &= keep
        n = sel.sum(1).astype(np.int64)
        d = d * sel
        nn = np.maximum(n, 2).astype(np.float64)
        m1 = d.sum(1, dtype=np.float64) / nn
        m2 = np.einsum("ij,ij->i", d, d).astype(np.float64) / nn
        var = np.maximum(m2 - m1 * m1, 0.0) * nn / (nn - 1)
        mu_new = (mu + m1).astype(np.float32)
        sd_new = np.maximum(np.sqrt(var) / corr, 1e-3).astype(np.float32)
        done = np.nanmax(np.abs(mu_new - mu)) < tol and np.nanmax(np.abs(sd_new - sd)) < tol
        mu, sd = mu_new, sd_new
        if done:
            break
    none = n < 10
    return (np.where(none, np.nan, mu).astype(np.float32), np.where(none, np.nan, sd).astype(np.float32), n)
