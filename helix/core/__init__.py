"""helix.core — detector-agnostic signal-processing shared by TPC and optical.

Public surface: backend selection and the wavelet sparsification API. Heavy
frameworks (jax/torch) are imported lazily by the backend dispatcher, never at
``import helix`` time.
"""
from helix.core.backend import get_backend, set_backend
from helix.core.wavelet import SparseResult, ThresholdSpec, sparsify, reconstruct
from helix.core.provenance import BasisDescriptor, derive_band_lengths, descriptor_digest
from helix.core.coeff_event import CoeffEvent
from helix.core.coeff_io import (
    write_coeff_shard, read_coeff_event, read_coeff_shard, n_events,
    coeff_event_to_arrays, arrays_to_coeff_event,
)

__all__ = [
    "get_backend", "set_backend",
    "SparseResult", "ThresholdSpec", "sparsify", "reconstruct",
    "BasisDescriptor", "derive_band_lengths", "descriptor_digest",
    "CoeffEvent",
    "write_coeff_shard", "read_coeff_event", "read_coeff_shard", "n_events",
    "coeff_event_to_arrays", "arrays_to_coeff_event",
]
