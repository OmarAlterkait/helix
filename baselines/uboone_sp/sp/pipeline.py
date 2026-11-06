"""The MicroBooNE (Wire-Cell, arXiv:1802.08709) signal-processing chain, as the
user's example implements it, with the response kernel injected.

Everything except the deconvolution entry point is the example's own code,
imported unchanged from reference/uboone_deconv/uboone_pipeline: the software
filters and their per-plane parameters (filters.py), ROI formation and
refinement (roi.py), oracle ROIs (roi.form_rois_true), ROI application with
induction baseline subtraction (roi.apply_rois), filtered truth
(pipeline.filter_truth) and the example's metrics (metrics.py).

``deconvolve_2d_dual`` here is a line-for-line copy of the example's
deconv.deconvolve_2d_dual with two changes, neither of which changes the
result for a given kernel (tests/check_equivalence.py verifies this to 1e-5):
  * the kernel is an argument instead of being read from avg_response_<plane>.npz;
  * real FFTs (rfft2) instead of complex fft2 -- the input, the kernel and both
    filters are real/Hermitian, so the result is identical and 2x cheaper.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(HERE, "reference", "uboone_deconv")
if REF not in sys.path:
    sys.path.insert(0, REF)

from uboone_pipeline.filters import (  # noqa: E402  (the example's code, unchanged)
    PLANE_PARAMS, build_2d_filter, lf_filter, time_filter_gaussian, time_filter_wiener, wire_filter)
from uboone_pipeline.roi import (  # noqa: E402
    apply_rois, form_rois_collection, form_rois_induction, form_rois_true)
from uboone_pipeline.pipeline import filter_truth  # noqa: E402

from sp.response import kernel_full  # noqa: E402


class Kernel:
    """A response kernel with its full-plane FFT cached per plane shape."""

    def __init__(self, R, wire_offsets, tzb):
        self.R, self.wo, self.tzb = R, wire_offsets, tzb
        self._cache = {}

    def rfft(self, nw, nt):
        if (nw, nt) not in self._cache:
            self._cache[(nw, nt)] = np.fft.rfft2(kernel_full(self.R, self.wo, self.tzb, nw, nt))
        return self._cache[(nw, nt)]


def deconvolve_2d_dual(M, plane, kernel, dt_us=0.5, params=None, noiseless=False):
    """The example's deconv.deconvolve_2d_dual, kernel injected (see module doc)."""
    p = dict(PLANE_PARAMS[plane])
    if params:
        p.update(params)
    is_induction = plane in ("U", "V")
    nw, nt = M.shape
    R_freq = kernel.rfft(nw, nt)
    M_freq = np.fft.rfft2(M.astype(np.float64))
    ntr = R_freq.shape[1]                                   # rfft half of the time axis

    F_w = wire_filter(nw, p["wire_a"])
    wiener_dc_zero = is_induction or (not noiseless)
    gauss_dc_zero = not noiseless
    F_t_wiener = time_filter_wiener(nt, dt_us, p["wiener_a"], p["wiener_b"], dc_zero=wiener_dc_zero)[:ntr]
    F_t_gauss = time_filter_gaussian(nt, dt_us, p["gauss_a"], dc_zero=gauss_dc_zero)[:ntr]

    R_abs = np.abs(R_freq)
    eps = 1e-6 * R_abs.max()
    R_safe = np.where(R_abs > eps, R_freq, eps * np.exp(1j * np.angle(R_freq)))
    base = M_freq / R_safe
    C_wiener = np.fft.irfft2(base * build_2d_filter(F_t_wiener, F_w), s=(nw, nt)).astype(np.float32)
    C_gauss = np.fft.irfft2(base * build_2d_filter(F_t_gauss, F_w), s=(nw, nt)).astype(np.float32)
    return C_wiener, C_gauss


def apply_lf_filter_to_wiener(C_wiener, plane, dt_us=0.5, which="tight", params=None):
    """The example's deconv.apply_lf_filter_to_wiener (rfft; identical result)."""
    p = dict(PLANE_PARAMS[plane])
    if params:
        p.update(params)
    a_mhz = p.get(f"lf_{which}_a")
    if a_mhz is None:
        return C_wiener.copy()
    nw, nt = C_wiener.shape
    F_lf = lf_filter(nt, dt_us, a_mhz)[: nt // 2 + 1]
    C = np.fft.rfft(C_wiener.astype(np.float64), axis=1)
    return np.fft.irfft(C * F_lf[None, :], n=nt, axis=1).astype(np.float32)


def run_sp(M, plane, kernel, truth=None, use_true_roi=False, true_roi_pad=10, noiseless=False,
           dt_us=0.5, params=None):
    """The example's pipeline.run_pipeline (Fig. 13 chain), kernel injected.

    Returns dict(C_wiener, C_gauss, C_charge, roi_mask, rois)."""
    is_induction = plane in ("U", "V")
    C_wiener, C_gauss = deconvolve_2d_dual(M, plane, kernel, dt_us=dt_us, params=params, noiseless=noiseless)
    if use_true_roi:
        rois = form_rois_true(truth, pad_ticks=true_roi_pad)
    elif is_induction:
        C_loose = apply_lf_filter_to_wiener(C_wiener, plane, dt_us=dt_us, which="loose", params=params)
        C_tight = apply_lf_filter_to_wiener(C_wiener, plane, dt_us=dt_us, which="tight", params=params)
        rois = form_rois_induction(C_wiener, C_loose, C_tight, PLANE_PARAMS[plane], noiseless=noiseless)
    else:
        rois = form_rois_collection(C_wiener, PLANE_PARAMS[plane], noiseless=noiseless)
    C_charge, roi_mask = apply_rois(C_gauss, rois, is_induction)
    return dict(C_wiener=C_wiener, C_gauss=C_gauss, C_charge=C_charge, roi_mask=roi_mask, rois=rois)


def fine_cells(C, FW=2, FT=16):
    """Sum a (n_wires, n_ticks) charge image over 2-wire x 16-tick cells, clip at 0."""
    nw, nt = C.shape
    pw, pt = (-nw) % FW, (-nt) % FT
    X = np.pad(C.astype(np.float64), ((0, pw), (0, pt)))
    X = X.reshape(X.shape[0] // FW, FW, X.shape[1] // FT, FT).sum((1, 3))
    return np.maximum(X, 0.0)


def lookup(cellmaps, keys):
    """Per fine-cell key, the value in ``cellmaps[g][fw, ft]`` (0 outside / absent)."""
    from helix.probe.resolution import unkey
    g, fw, ft = unkey(np.asarray(keys, np.int64))
    out = np.zeros(len(g), np.float64)
    for gg, X in cellmaps.items():
        m = (g == gg) & (fw >= 0) & (fw < X.shape[0]) & (ft >= 0) & (ft < X.shape[1])
        out[m] = X[fw[m], ft[m]]
    return out
