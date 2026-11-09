"""helix.data — the coeff corpus as a pimm-data family (``dataset``), its simulation truth
(``cells``), the denoiser's labels (``denoise``), the bin grid (``bins``) and the verifier
(``verify``).

This package MAY import pimm_data (``helix.core`` and ``helix.tpc`` never do), but only
``dataset`` does. The two names below therefore resolve lazily: a package's ``__init__`` runs
before any submodule of it, and an eager import here would put torch and pimm_data behind
``import helix.data.bins`` -- which the training configs do at module scope on a launcher host
that has neither (``tests/test_boundary.py``). Registration into pimm-data's registry is
unaffected: ``helix.integrations.pimm`` reaches ``helix.data.dataset`` by its own path.
"""
import importlib

_LAZY = {"CoeffTPCReader": "helix.data.dataset", "CoeffTPCDataset": "helix.data.dataset"}
__all__ = sorted(_LAZY)


def __getattr__(name):
    """Resolve a heavy name on first use (PEP 562)."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value
    return value
