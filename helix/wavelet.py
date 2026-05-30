"""Back-compat shim for the old (image, config) wavelet API.

New code should use ``helix.core.wavelet.sparsify(image, *, wavelet, level,
mode, threshold)`` directly. These wrappers translate a DetectorConfig into the
core call so existing TPC callers/tests keep working.
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
