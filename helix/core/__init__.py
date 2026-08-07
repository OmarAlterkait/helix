"""helix.core — detector-agnostic signal-processing shared by TPC and optical.

Public surface: backend selection and the wavelet sparsification API. Heavy
frameworks (jax/torch) are imported lazily by the backend dispatcher, never at
``import helix`` time.

**On the vocabulary.** ``CoeffEvent`` and the shard codec speak TPC:
``(band, plane_gid, wire, tau)``. The STRUCTURE is detector-agnostic — ``wire``
is "row index of the plane image" and ``tau`` "column index within the band", so
an optical chunk is a row exactly as a wire is — but the NAMES are TPC-derived,
deliberately. They are the on-disk column names of every corpus shard, they are
what the tokenizer and the trained checkpoints assume, and renaming them would
change the format and every golden for no functional gain. Read ``wire``/``tau``
as ``row``/``col`` when working on another detector.
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
