"""helix.model — the FM MAE and its tokenizer: the interface pimm trains against.

See RESEARCH_EXTRACTION_MAP section 5a. helix PROVIDES the model; pimm owns the
loop and the probes.

The tokenizer lives here because the patch geometry IS a property of the model —
pw/pt/n_bands, the RoPE time coordinate, and the arcsinh normalisation are all
dictated by what the FM consumes. It carries both directions: ``assemble`` (rows
-> tokens) and ``detokenize`` / ``decode_prediction`` (tokens -> rows).

**Model attributes are imported lazily, on purpose.** ``helix.model.tokenize`` is
pure numpy and must stay importable WITHOUT torch — pimm-data runs it inside
DataLoader workers, and the DSP half of helix installs without the ``[torch]``
extra. A plain ``from helix.model.fm import FMModel`` here would drag torch into
every tokenizer import. PEP 562 keeps ``import helix.model.tokenize`` free of it
while ``from helix.model import FMModel`` still works.
"""

_LAZY = {
    "FMModel": "helix.model.fm",
    "build_fm": "helix.model.fm",
    "SerialFMModel": "helix.model.serial",
    "losses": "helix.model.loss",
    "losses_fused": "helix.model.loss",
    "losses_cat": "helix.model.loss",
    "make_mask": "helix.model.mask",
}

__all__ = sorted(_LAZY) + ["tokenize"]


def __getattr__(name):                     # PEP 562: defer the torch import
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(globals()))
