"""helix.data — the coeff corpus as a pimm-data family.

The layer that turns helix's own on-disk format into training batches. It lives
here, not in pimm-data, because the format is helix's: the writer
(``helix.core.coeff_io``), the reader and the verifier are one artifact, and
splitting them across repos is what let the codec drift into two copies held
together by a cross-repo golden test.

INVARIANT: ``helix.core`` and ``helix.tpc`` never import ``pimm_data`` — that is
what keeps the DSP path installable with numpy alone (pimm-data requires
``torch>=2.5``). ``helix.data`` and ``helix.integrations`` may, and are the only
places that do.

The framework comes from pimm-data (``ShardEventDataset``, ``ShardReaderBase``,
``DATASETS``, ``read_shard_meta``); the family is ours and registers into
pimm-data's registry.
"""
from helix.data.coeff_reader import CoeffTPCReader
from helix.data.coeff_dataset import CoeffTPCDataset

__all__ = ["CoeffTPCReader", "CoeffTPCDataset"]
