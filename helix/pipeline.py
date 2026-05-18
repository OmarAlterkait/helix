"""End-to-end pipeline: coherent removal → wavelet sparsification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from helix.config import DetectorConfig
from helix.coherent import remove_coherent
from helix.wavelet import sparsify, reconstruct, SparseResult


@dataclass
class ProcessedPlane:
    """Output of the full pipeline for one plane."""
    cleaned: Any
    sparse: SparseResult
    reconstructed: Any
    config: DetectorConfig


def process_plane(
    image: Any,
    config: DetectorConfig,
    sigma_per_wire: Any | None = None,
) -> ProcessedPlane:
    """Full pipeline for a single plane.

    Parameters
    ----------
    image : array (n_wires, n_ticks), float32
        Pedestal-subtracted digitized readout.
    config : DetectorConfig
    sigma_per_wire : array (n_wires,), optional

    Returns
    -------
    ProcessedPlane
    """
    cleaned = remove_coherent(image, config, sigma_per_wire)
    sparse = sparsify(cleaned, config)
    recon = reconstruct(sparse, config, image.shape[1])

    return ProcessedPlane(
        cleaned=cleaned,
        sparse=sparse,
        reconstructed=recon,
        config=config,
    )


def process_event(
    planes: dict[str, Any],
    config: DetectorConfig,
    sigma_per_wire: dict[str, Any] | None = None,
) -> dict[str, ProcessedPlane]:
    """Process all planes of one event.

    Parameters
    ----------
    planes : dict mapping plane label → (n_wires, n_ticks) array
    config : DetectorConfig
    sigma_per_wire : dict mapping plane label → (n_wires,) array, optional
    """
    results = {}
    for label, image in planes.items():
        sw = sigma_per_wire.get(label) if sigma_per_wire else None
        results[label] = process_plane(image, config, sw)
    return results
