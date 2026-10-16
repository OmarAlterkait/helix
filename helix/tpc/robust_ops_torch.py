"""Torch backend of :mod:`helix.tpc.robust`: same iteration, sums in float64."""
from __future__ import annotations

import numpy as np
import torch

from helix.tpc.robust_ops_numpy import trunc_var_factor


def _row_quantiles(x, qs=(0.25, 0.5, 0.75)):
    T = x.shape[1]
    ks = [min(T - 1, int(round(q * (T - 1)))) for q in qs]
    return [torch.kthvalue(x, kk + 1, dim=1).values for kk in ks]


def _arr(v, like):
    if v is None:
        return None
    if not isinstance(v, torch.Tensor):
        v = torch.as_tensor(np.asarray(v, np.float32))
    return v.to(device=like.device, dtype=torch.float32)


def clipped_stats(x, mask=None, k=3.5, n_iter=5, center=None, sd0=None, tol=1e-3):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(np.asarray(x, np.float32))
    x = x.to(torch.float32)
    keep = None if mask is None else ~torch.as_tensor(mask, device=x.device).to(torch.bool)
    center, sd0 = _arr(center, x), _arr(sd0, x)
    if center is None or sd0 is None:
        q = _row_quantiles(x)
        mu = q[1] if center is None else center
        sd = torch.clamp_min((q[2] - q[0]) / 1.349, 0.5) if sd0 is None else sd0
    else:
        mu, sd = center.clone(), sd0.clone()
    corr = float(np.sqrt(trunc_var_factor(k)))
    n = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
    # numpy stops once no row moves by >= tol; here every iteration runs and rows freeze
    # at that point instead (same result, no host sync per iteration)
    done = torch.zeros((), dtype=torch.bool, device=x.device)
    for _ in range(n_iter):
        d = x - mu[:, None]
        sel = d.abs() <= (k * sd)[:, None]
        if keep is not None:
            sel &= keep
        n_new = sel.sum(1)
        d = d * sel
        nn = torch.clamp_min(n_new, 2).to(torch.float64)
        m1 = d.sum(1, dtype=torch.float64) / nn
        m2 = torch.sum(d * d, 1, dtype=torch.float64) / nn
        var = torch.clamp_min(m2 - m1 * m1, 0.0) * nn / (nn - 1)
        mu_new = (mu + m1).to(torch.float32)
        sd_new = torch.clamp_min(torch.sqrt(var) / corr, 1e-3).to(torch.float32)
        conv = torch.maximum(torch.nan_to_num((mu_new - mu).abs(), nan=0.0).max(),
                             torch.nan_to_num((sd_new - sd).abs(), nan=0.0).max()) < tol
        mu, sd, n = torch.where(done, mu, mu_new), torch.where(done, sd, sd_new), torch.where(done, n, n_new)
        done = done | conv
    none = n < 10
    nan = torch.full((), float("nan"), device=x.device)
    return torch.where(none, nan, mu), torch.where(none, nan, sd), n
