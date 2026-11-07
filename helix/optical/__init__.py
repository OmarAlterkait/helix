"""helix.optical — optical (PMT / X-ARAPUCA) waveforms, data and simulation alike.

One codec for every light system (helix.optical.codec): SBND PMT and X-ARAPUCA and goop's
simulated PMTs (codec.GOOP), built by helix.build (sources: helix.tpc.io_sbnd.SBNDSource,
helix.optical.goop.GoopSource). A system is its measured single-photon response
(helix.optical.calibrate); the codec deconvolves the window with it and keeps the integer-Haar differences where the light, or the departure from it, reaches
threshold, coarse sums always. Optical shards (helix.optical.shard: one modality per file next to
the TPC coefficient shards) and 128-slot tokens (helix.optical.tokens).
"""
from helix.optical.codec import GOOP, SBND, OpticalChain, OpticalSpec

__all__ = ["OpticalChain", "OpticalSpec", "SBND", "GOOP"]
