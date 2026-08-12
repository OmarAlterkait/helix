"""HELIX — Hierarchical Encoding for Learned Inference on eXperimental data.

  helix.core    — detector-agnostic wavelet sparsification, the coefficient
                  shard codec, and lazy backend dispatch (numpy / jax / torch,
                  imported only when selected)
  helix.tpc     — wire-plane pipeline: coherent noise removal + wavelet, and
                  the corpus builder
  helix.optical — PMT optical-waveform pipeline (goop light files)
  helix.model   — the coefficient foundation model and its tokenizer

**Every name below is resolved lazily** (PEP 562). This is not a style choice.

The umbrella used to import ``helix.tpc.{config,pipeline,io}`` eagerly, so
``import helix.core.coeff_io`` — or ``helix.model.tokenize``, which pimm-data
runs inside every DataLoader worker — pulled in the whole wire-plane HDF5 stack.
Measured: importing the tokenizer cost 1.07s and dragged ``h5py``, for a module
that needs neither. It also made ``helix.core``'s "detector-agnostic" claim
false at import time, and left the tokenizer one heavy or broken import in
``helix.tpc`` away from failing in every worker.

Two invariants this must preserve, both already enforced by tests:
``import helix`` must pull in neither ``pimm`` nor ``torch``
(``tests/test_integration_pimm.py``), and ``helix.model.tokenize`` must stay
torch-free (``tests/test_tokenize.py``, checked in a subprocess). Laziness
strengthens both — the model names below resolve only when touched.

``__version__`` stays eager: ``helix.core.coeff_io`` stamps it into every corpus
shard, and that must not depend on attribute-access order.
"""
__version__ = "0.2.0"

#: Public name -> the module that defines it. Three tiers: the coefficient
#: codec (the cross-repo contract pimm-data reads), the DSP, and the model.
_LAZY = {
    # -- codec: what pimm-data and the corpus verifier depend on -------------
    "CoeffEvent": "helix.core.coeff_event",
    "BasisDescriptor": "helix.core.provenance",
    "write_coeff_shard": "helix.core.coeff_io",
    "read_coeff_event": "helix.core.coeff_io",
    "audit_shard": "helix.core.coeff_io",
    "coord_digest": "helix.core.coeff_io",
    # -- DSP ----------------------------------------------------------------
    "get_backend": "helix.core.backend",
    "set_backend": "helix.core.backend",
    "sparsify": "helix.core.wavelet",
    "DetectorConfig": "helix.tpc.config",
    "process_plane": "helix.tpc.pipeline",
    "process_event": "helix.tpc.pipeline",
    "config_from_file": "helix.tpc.io",
    "build_corpus": "helix.tpc.corpus",
    "build_corpus_stream": "helix.tpc.corpus",
    # -- model: torch is imported only if one of these is touched -----------
    "build_fm": "helix.model",
    "FMModel": "helix.model",
    "CoeffTokenize": "helix.model.tokenize",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    """Resolve a public name to its defining module on first access (PEP 562)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'helix' has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(target), name)
    globals()[name] = value          # cache: subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
