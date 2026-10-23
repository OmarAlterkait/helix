"""Hysteresis (two-threshold) selection, shared by the TPC and optical chains.

HELIX's universal threshold keeps an element when it alone is above
kappa sigma sqrt(2 ln N): the largest of N noise values. Hysteresis (Canny) keeps, in
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
