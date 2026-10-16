"""helix.optical — optical (PMT / X-ARAPUCA) waveforms, data and simulation alike.

  * codec (helix.optical.codec): one method for every light system - SBND PMT and X-ARAPUCA
    (helix.tpc.sbnd_run) and goop's simulated PMTs (helix.optical.goop_run, codec.GOOP):
    deconvolve with the system's measured response, keep the integer-Haar differences where the
    light (or the departure from it) reaches threshold, coarse sums always; optical shards
    (helix.tpc.sparse_io) and 128-slot tokens (helix.optical.tokens);
  * the earlier goop pipeline (process_event: coif3 + VisuShrink + quantization) is kept as the
    reference - it loses most isolated photons and part of the slow light.
"""
from helix.optical.config import OpticalConfig
from helix.optical.io import (config_from_file, list_events, read_event_chunks,
                              chunk_noise_sigma)
from helix.optical.pipeline import process_event, OpticalResult
from helix.optical.viz import decompose, plot_decomposition

__all__ = [
    "OpticalConfig", "config_from_file", "list_events", "read_event_chunks",
    "chunk_noise_sigma", "process_event", "OpticalResult",
    "decompose", "plot_decomposition",
]
