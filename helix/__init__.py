"""HELIX — Hierarchical Encoding for Learned Inference on eXperimental data.

Signal processing pipeline for liquid argon TPC wire data:
coherent noise removal followed by wavelet sparsification.
"""

__version__ = "0.1.0"

from helix.config import DetectorConfig
from helix.pipeline import process_plane, process_event

__all__ = ["DetectorConfig", "process_plane", "process_event"]
