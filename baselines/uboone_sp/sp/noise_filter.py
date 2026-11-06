"""Coherent-noise removal ahead of the deconvolution (rung R2).

The example ships no noise filter (it never ran with noise, HANDOVER.md §5), so
R2 uses OUR removal, two ways:
  gate       what the corpus does (helix.tpc.pipeline mode 'gate'): DWT ->
             helix.tpc.coherent_gate -> inverse DWT, with no thresholding --
             the coherent-removed image, before the model's sparsification.
  multipass  helix.tpc.coherent.remove_coherent: image-space per-64-wire-group
             masked mean with signal protection, the closest analogue of
             MicroBooNE's coherent filter (median of the group, signal ROIs
             protected; JINST 12 P08003).
"""
from __future__ import annotations

import numpy as np


def _cfg():
    from helix.tpc.config import DetectorConfig
    return DetectorConfig()


def gate_clean(img, cfg=None):
    from helix.core import backend
    backend.set_backend("numpy")
    from helix.core.wavelet import wavedec
    from helix.core.wavelet_ops_numpy import reconstruct
    from helix.tpc.coherent_gate import coherent_gate
    cfg = cfg or _cfg()
    nt = img.shape[1]
    x = np.pad(np.asarray(img, np.float32), ((0, 0), (0, (-nt) % (1 << cfg.dwt_level))))
    coeffs, lev = wavedec(x, wavelet=cfg.wavelet, level=cfg.dwt_level, mode=cfg.dwt_mode)
    gated = coherent_gate(coeffs, group_size=cfg.group_size, kgate=cfg.gate_kgate, ksig=cfg.gate_ksig,
                          npass=cfg.gate_npass, tau=cfg.gate_tau, gate_approx=True)
    return np.asarray(reconstruct([np.asarray(b) for b in gated], cfg.wavelet, lev, cfg.dwt_mode, nt), np.float32)


def multipass_clean(img, cfg=None):
    from helix.core import backend
    backend.set_backend("numpy")
    from helix.tpc.coherent import remove_coherent
    cfg = cfg or _cfg()
    return np.asarray(remove_coherent(np.asarray(img, np.float32), cfg), np.float32)
