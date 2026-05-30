"""HELIX — Hierarchical Encoding for Learned Inference on eXperimental data.

Signal-processing pipelines for liquid-argon TPC detector data:

  helix.core    — detector-agnostic wavelet sparsification + lazy backend
                  dispatch (numpy / jax / torch, imported only when selected)
  helix.tpc     — wire-plane pipeline: coherent noise removal + wavelet
  helix.optical — PMT optical-waveform pipeline (goop light files)

Top-level names below are kept for backward compatibility and resolve to
helix.tpc / helix.core.
"""
__version__ = "0.2.0"

from helix.core.backend import get_backend, set_backend
from helix.tpc.config import DetectorConfig
from helix.tpc.pipeline import process_plane, process_event
from helix.tpc.io import config_from_file

__all__ = [
    "DetectorConfig", "config_from_file", "process_plane", "process_event",
    "get_backend", "set_backend",
]
