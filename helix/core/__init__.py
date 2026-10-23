"""helix.core — detector-agnostic signal-processing shared by TPC and optical.

Public surface: the coefficient event and its shard codec, the basis descriptor,
the wavelet transform. Signal processing is torch (any device); torch is imported
by the functions that compute, never at ``import helix`` time, so readers of the
corpus run without it.

**On the vocabulary.** ``CoeffEvent`` and the shard codec speak TPC:
``(band, plane_gid, wire, tau)``. The STRUCTURE is detector-agnostic — ``wire``
is "row index of the plane image" and ``tau`` "column index within the band", so
an optical chunk is a row exactly as a wire is — but the NAMES are TPC-derived,
deliberately. They are the on-disk column names of every corpus shard, they are
what the tokenizer and the trained checkpoints assume, and renaming them would
change the format and every golden for no functional gain. Read ``wire``/``tau``
as ``row``/``col`` when working on another detector.
"""
from helix.core.wavelet import SparseResult, wavedec, reconstruct
from helix.core.provenance import BasisDescriptor, derive_band_lengths, descriptor_digest
from helix.core.coeff_event import CoeffEvent
from helix.core.coeff_io import (
    write_coeff_shard, read_coeff_event, read_coeff_shard, n_events,
    arrays_to_coeff_event,
)

__all__ = [
    "SparseResult", "wavedec", "reconstruct",
    "BasisDescriptor", "derive_band_lengths", "descriptor_digest",
    "CoeffEvent",
    "write_coeff_shard", "read_coeff_event", "read_coeff_shard", "n_events",
    "arrays_to_coeff_event",
]
