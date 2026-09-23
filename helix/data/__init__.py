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

WHY THE TWO NAMES BELOW ARE LAZY
--------------------------------
This package MAY import pimm_data. Not every module in it NEEDS to, and two do
not: ``helix.data.bins`` (numpy) and ``helix.data.identity`` (h5py) are corpus
metadata, readable anywhere the corpus is readable. Importing either used to
pull ``torch`` and ``pimm_data`` anyway, because Python runs a package's
``__init__`` before any submodule of it and this one imported the reader and the
dataset eagerly.

That is not academic. pimm documents a launcher-only environment -- "Login nodes
and remote submission hosts may need only YAML parsing, Tyro, and Submitit...
It cannot import the full model stack" -- and `pimm submit` loads the training
config on the submitting host to check batch-size divisibility. helix's config
reads the bin table and the corpus run list at module scope, both from this
package, so submission demanded the full training stack on a login node and
failed with `No module named 'pimm_data'`. Same shape as the smoke-path failure
that `load_bins needed nothing from pimm` fixed: a cheap thing made expensive by
the package it is spelled inside.

Deferring costs nothing. Both names resolve on first attribute access, so
``from helix.data import CoeffTPCDataset`` still works unchanged, and
registration into pimm-data's registry is unaffected -- it happens when
``helix.integrations.pimm`` is imported by ``custom_imports``, which reaches
``coeff_dataset`` by its own module path, never through this one.
``tests/test_boundary.py`` pins both halves of this.
"""
import importlib
from typing import Any

#: name -> the module that defines it. Deliberately a table rather than a chain
#: of ``if``s: ``__dir__`` and the error message below both read it, so adding a
#: name here is the whole change.
_LAZY = {
    "CoeffTPCReader": "helix.data.coeff_reader",
    "CoeffTPCDataset": "helix.data.coeff_dataset",
}

__all__ = ["CoeffTPCReader", "CoeffTPCDataset"]


def __getattr__(name: str) -> Any:
    """Resolve a heavy name on first use (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    # Cache on the module so the second access is a plain global lookup and
    # never re-enters this function.
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY))
