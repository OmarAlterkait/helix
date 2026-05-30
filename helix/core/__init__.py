"""helix.core — detector-agnostic signal-processing shared by TPC and optical.

Public surface: backend selection and the wavelet sparsification API. Heavy
frameworks (jax/torch) are imported lazily by the backend dispatcher, never at
``import helix`` time.
"""
from helix.core.backend import get_backend, set_backend
from helix.core.wavelet import SparseResult, ThresholdSpec, sparsify, reconstruct

__all__ = [
    "get_backend", "set_backend",
    "SparseResult", "ThresholdSpec", "sparsify", "reconstruct",
]
