"""helix.optical — PMT optical-waveform pipeline (goop light files).

Reuses helix.core wavelet sparsification; adds the east/west chunk loader,
optical config, per-chunk metrics, and the chunk-batched event pipeline.
"""
from helix.optical.config import OpticalConfig
from helix.optical.io import (config_from_file, list_events, read_event_chunks,
                              chunk_noise_sigma, deslice_side)
from helix.optical.pipeline import process_event, OpticalResult
from helix.optical.viz import decompose, plot_decomposition

__all__ = [
    "OpticalConfig", "config_from_file", "list_events", "read_event_chunks",
    "chunk_noise_sigma", "deslice_side", "process_event", "OpticalResult",
    "decompose", "plot_decomposition",
]
