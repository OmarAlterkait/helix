"""helix.core — detector-agnostic signal processing shared by TPC and optical: the coefficient
event and its shard codec (``coeff_event``, ``coeff_io``), the wavelet transform (``wavelet``),
robust statistics (``robust``). Torch is imported by the functions that compute, never at import.

**On the vocabulary.** ``CoeffEvent`` and the codec speak TPC, ``(band, plane_gid, wire, tau)``,
but the STRUCTURE is detector-agnostic: ``wire`` is the row of the plane image, ``tau`` the column
within the band, so an optical chunk is a row as a wire is. The names are the on-disk columns of
every shard and what the tokenizer and checkpoints assume; read them as ``row``/``col`` elsewhere.
"""
