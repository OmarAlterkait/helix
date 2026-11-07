"""Robust statistics and hysteresis selection, shared by the TPC and optical chains (torch, any
device).

Statistics:

* :func:`q50` - the median as ``np.median`` defines it (the AVERAGE of the two middle values);
* :func:`masked_median` - the same over the valid entries of each slice;
* :func:`mad` / :func:`masked_mad` - the noise sigma of continuous values: q50(|x|) / 0.6745;
* :func:`clipped_stats` - per-row pedestal and noise of integer ADC samples: iterative k-sigma
  clipped mean / rms.

Never ``torch.median`` / ``torch.nanmedian``: they return the LOWER of the two middles. Every
sigma here feeds a threshold, a discontinuous function of it, so the tie-break decides which
coefficients survive (torch.median kept 1-2 extra coefficients per event, never fewer).

On integer ADC data the median and the MAD are quantized (a median pedestal has a flat ~0.29
ADC error; the MAD of codes collapses below ~0.6 ADC), so per-channel statistics of the RAW
waveform use :func:`clipped_stats`, with the truncation corrected:
  * seed: per-row nearest-rank quartiles (median, IQR / 1.349);
  * iterate: keep |x - mu| <= k sd (and not ``mask``), mu += mean, sd = rms / c(k), with c(k)^2
    the variance of a unit Gaussian truncated at +-k; rows freeze once no row moves by ``tol``;
  * rows with fewer than 10 samples left get NaN (no estimate).

Hysteresis (two-threshold) selection. HELIX's universal threshold keeps an element when it alone
is above kappa sigma sqrt(2 ln N): the largest of N noise values. Hysteresis (Canny) keeps, in
addition, every element above a low ``floor`` that is CONNECTED to such a seed through
elements that also pass the floor. A faint element is kept where it continues a
structure that is certainly there (a track's faint end, a pulse's tail) and dropped
where noise crosses the floor in isolation.

:func:`grow` (seed, candidate) -> kept, connectivity ``conn``:
  * ``"8"``: 2-D, 8-connected (TPC: wire x coefficient within a band);
  * ``"1d"``: along axis 1 only (optical: samples of one waveform).
:func:`window_any`: any(m) in a window along an axis (dilation of a mask).

Torch on any device, by flooding from the seeds (the same kept set as labelling the
components, e.g. scipy.ndimage.label):

  * ``"1d"``: one step. Runs of candidates along a row get ids from a cumulative sum
    of run starts; a run is kept when it holds a kept element (no host sync).
  * ``"8"``: repeat {that row-run fill; one 3x3 dilation within the candidates} until
    nothing changes, checked every ``_CHECK`` steps (one host sync each). Faint
    extensions of seeded structure are short, so this converges in ~10 steps on SBND
    (max 16 over 30 planes x 5 bands), where min-label propagation with pointer
    jumping took up to 71. GPU, SBND event (6 planes x 5 bands, RTX 2080 Ti): 0.18 s
    vs 0.40 s (labelling) and 0.45 s (scipy on the host, transfers included).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

#: reduction length above which q50 sorts instead of selecting. 4096 sits well past the gate's
#: per-block axes (group_size=64) and well below the flattened bands (0.5-4.3M) where kthvalue
#: collapses.
_Q50_SORT_ABOVE = 4096


def q50(x, dim: int):
    """``np.median`` along ``dim``: the average of the two middle values; NaN propagates.

    Two implementations, chosen by the reduction length, because their costs invert:
    ``kthvalue`` (a selection) wins on short axes but is pathological on long ones on CUDA
    (3.5 ms at 0.53M elements, 61 ms at 4.27M, against 1.1 ms for a full sort); a sort has no
    size ceiling (``torch.quantile`` refuses inputs above ~2**24)."""
    n = x.shape[dim]
    if n <= _Q50_SORT_ABOVE:
        lo = torch.kthvalue(x, (n + 1) // 2, dim=dim).values
        out = lo if n % 2 else (lo + torch.kthvalue(x, n // 2 + 1, dim=dim).values) * 0.5
    else:
        s, _ = torch.sort(x, dim=dim)
        m = n // 2
        out = s.select(dim, m) if n % 2 else (s.select(dim, m - 1) + s.select(dim, m)) * 0.5
    # sort pushes NaN to the end and kthvalue selects around it: propagate it as np.median does
    nan = torch.isnan(x).any(dim=dim)
    return torch.where(nan, torch.full_like(out, float("nan")), out)


def masked_median(x, v, dim):
    """``np.median(x[v])`` along ``dim`` per slice (NaN where no valid entry): a sort with
    invalid entries set to +inf and the average of the ``(n-1)//2`` and ``n//2`` order statistics."""
    inf = torch.full((), float("inf"), dtype=x.dtype, device=x.device)
    xs, _ = torch.sort(torch.where(v, x, inf), dim=dim)
    n = v.sum(dim=dim, keepdim=True)
    a = torch.gather(xs, dim, torch.clamp_min((n - 1) // 2, 0))
    b = torch.gather(xs, dim, torch.clamp_min(n // 2, 0))
    med = 0.5 * (a + b)
    return torch.where(n > 0, med, torch.full_like(med, float("nan"))).squeeze(dim)


def mad(x, dim: int, eps: float | None = None):
    """Gaussian sigma from the median absolute value: q50(|x|) / 0.6745 (floored at ``eps``)."""
    s = q50(x.abs(), dim) / 0.6745
    return s if eps is None else torch.clamp_min(s, eps)


def masked_mad(x, v, dim: int, eps: float | None = None):
    """:func:`mad` over the valid entries ``v`` of each slice (NaN where none is valid)."""
    s = masked_median(x.abs(), v, dim) / 0.6745
    return s if eps is None else torch.clamp_min(s, eps)


def trunc_var_factor(k: float) -> float:
    """Var of N(0,1) truncated to [-k, k]."""
    from scipy.special import erf
    phi = np.exp(-0.5 * k * k) / np.sqrt(2 * np.pi)
    p = erf(k / np.sqrt(2))
    return 1.0 - 2 * k * phi / p


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
    """Per row of ``x`` (rows, n): (mean, rms, n_used), sums in float64. See the module docstring."""
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
    # every iteration runs and rows freeze once converged (no host sync per iteration)
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


# ---- hysteresis ----------------------------------------------------------------------------

_CHECK = 4


def _t(x, dev=None):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x, bool), device=dev)


def _fill_runs(keep, cand):
    """keep | every run of ``cand`` along a row that holds a ``keep`` element."""
    st = cand.clone()
    st[:, 1:] &= ~cand[:, :-1]
    N = cand.numel()
    rid = (torch.cumsum(st.reshape(-1).to(torch.int64), 0) * cand.reshape(-1))     # 0 outside cand
    # mark the runs holding a kept element: every element writes True, to its run if kept
    # else to a dump slot (N + 1). Same value everywhere, so no atomics; boolean indexing
    # would sync the host (nonzero), scatter_reduce is ~3x slower (measured, 2080 Ti)
    good = torch.zeros(N + 2, dtype=torch.bool, device=cand.device)
    good.index_put_((torch.where((keep & cand).reshape(-1), rid, N + 1),), torch.ones((), dtype=torch.bool, device=cand.device))
    # good[0] (outside cand) is never written and good[N + 1] never read: no reset (a
    # scalar store would sync the host)
    return keep | good[rid].reshape(cand.shape)


def _dilate8(m):
    dt = torch.float16 if m.is_cuda else torch.float32
    return F.max_pool2d(m[None, None].to(dt), 3, stride=1, padding=1)[0, 0] > 0


def grow(seed, cand, conn="8"):
    """Elements of ``seed | cand`` connected (``conn``) to a ``seed`` element through
    ``cand`` elements. Bool, 2-D (rows, columns)."""
    if conn not in ("8", "1d"):
        raise ValueError(f"conn must be '8' or '1d', got {conn!r}")
    seed = _t(seed)
    cand = _t(cand, seed.device) | seed
    keep = _fill_runs(seed, cand)
    if conn == "1d":
        return keep
    steps = 2 * _CHECK                                 # most bands converge within ~10 steps
    while True:
        prev = keep
        for _ in range(steps):
            keep = _fill_runs(_dilate8(keep) & cand, cand)
        if torch.equal(keep, prev):
            return keep
        steps = _CHECK


def grow_many(seeds, cands, conn="8"):
    """One :func:`grow` per mask: batching them side by side makes every mask wait for
    the slowest to converge (measured slower)."""
    return [grow(s_, c_, conn) for s_, c_ in zip(seeds, cands)]


def window_any(m, left, right, dim=1):
    """out[i] = any(m[i-left : i+right+1]) along ``dim`` (bool, 2-D). Prefix sums: cost
    independent of the window (max_pool1d is O(window) per output)."""
    m = _t(m)
    if dim == 0:
        return window_any(m.T, left, right, 1).T
    T = m.shape[1]
    c = F.pad(torch.cumsum(m.to(torch.int32), dim=1), (1, 0))          # c[:, j] = sum m[:, :j]
    i = torch.arange(T, device=m.device)
    return (c[:, torch.clamp(i + right + 1, max=T)] - c[:, torch.clamp(i - left, min=0)]) > 0
