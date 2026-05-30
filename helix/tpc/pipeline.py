"""End-to-end TPC pipeline: coherent removal → wavelet sparsification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from helix.core.wavelet import sparsify, reconstruct, SparseResult
from helix.tpc.config import DetectorConfig
from helix.tpc.coherent import remove_coherent


@dataclass
class ProcessedPlane:
    cleaned: Any
    sparse: SparseResult
    reconstructed: Any
    config: DetectorConfig


def process_plane(image: Any, config: DetectorConfig, sigma_per_wire: Any | None = None) -> ProcessedPlane:
    """Full pipeline for one plane: (n_wires, n_ticks) → cleaned + sparse + reconstruction."""
    cleaned = remove_coherent(image, config, sigma_per_wire)
    sparse = sparsify(cleaned, wavelet=config.wavelet, level=config.dwt_level,
                      mode=config.dwt_mode, threshold=config.threshold_spec())
    recon = reconstruct(sparse, image.shape[1])
    return ProcessedPlane(cleaned=cleaned, sparse=sparse, reconstructed=recon, config=config)


def process_event(planes: dict[str, Any], config: DetectorConfig,
                  sigma_per_wire: dict[str, Any] | None = None) -> dict[str, ProcessedPlane]:
    """Process all planes of one event."""
    results = {}
    for label, image in planes.items():
        sw = sigma_per_wire.get(label) if sigma_per_wire else None
        results[label] = process_plane(image, config, sw)
    return results
