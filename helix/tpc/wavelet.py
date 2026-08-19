"""The (image, DetectorConfig) wavelet API — the TPC's adapter over helix.core.

``helix.core.wavelet`` is detector-agnostic and takes explicit keywords
(``wavelet``, ``level``, ``mode``, ``threshold``); a TPC caller has those in a
``DetectorConfig``. This translates one into the other, which is a TPC concern
and so lives under ``helix.tpc`` rather than at the package root.

It sat at ``helix/wavelet.py`` labelled a back-compat shim while being the only
implementation of this signature — so "shim" read as "safe to delete", which it
was not. It is not a shim; it is the adapter.
"""
from helix.core.wavelet import SparseResult, ThresholdSpec
from helix.core.wavelet import sparsify as _sparsify
from helix.core.wavelet import reconstruct as _reconstruct

__all__ = ["SparseResult", "ThresholdSpec", "sparsify", "reconstruct"]


def sparsify(image, config):
    return _sparsify(image, wavelet=config.wavelet, level=config.dwt_level,
                     mode=config.dwt_mode, threshold=config.threshold_spec())


def reconstruct(result, config, n_time):
    return _reconstruct(result, n_time)
