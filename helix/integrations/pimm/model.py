"""The ``Coeff-FM`` MODELS entry: helix's FM, plus checkpoint/bin loading."""

from __future__ import annotations

import torch

from pimm.models.builder import MODELS


@MODELS.register_module("Coeff-FM")
class CoeffFM:
    """Factory registered as ``Coeff-FM``.

    A class rather than a function because pimm's registry requires one:
    ``_register_module`` raises ``TypeError: module must be a class``, and
    ``build_from_cfg`` ends in ``obj_cls(**args)``. ``__new__`` returns the
    ``FMModel`` itself, so a config gets a model rather than a wrapper, and
    nothing downstream has to unwrap it.

    Found only by running it: the registry's type check fires at DECORATION
    time, so a function here fails at import of this module — every test that
    read the source instead of importing it stayed green.
    """

    def __new__(cls, checkpoint=None, weights=True, bins=None, **cfg):
        return build_coeff_fm(checkpoint=checkpoint, weights=weights,
                              bins=bins, **cfg)


def build_coeff_fm(checkpoint=None, weights=True, bins=None, **cfg):
    """Build the coefficient FM, optionally restoring a converted checkpoint.

    ``FMModel.forward(batch) -> dict`` already satisfies pimm's Trainer contract
    (``output_dict["loss"]``), so nothing is wrapped.

    Args:
        checkpoint (str | None): a helix eval artifact or a ``pimm export``
            directory — self-describing, carrying the architecture, the operating
            point and (for a categorical head) the bin ``edges``. Its
            architecture is used; ``cfg`` overrides individual fields.
        weights (bool): restore the weights. ``False`` builds the same
            architecture freshly initialised.
        **cfg: architecture kwargs for ``helix.model.build_fm``.
    """
    from helix.model import build_fm
    from helix.model.artifact import inspect, load
    from helix.model.checkpoint import apply_bins, load_state_dict

    art = None
    if checkpoint is not None:
        # One reader, one error message. Every refusal a raw pimm checkpoint,
        # a DCP resume directory or a research blob deserves lives in
        # helix.model.artifact, so the four call sites cannot drift apart.
        art = (load if weights else inspect)(checkpoint)
        arch = dict(art.arch)
        arch.update(cfg)                 # the config overrides the checkpoint
        cfg = arch
        if bins is None:
            bins = art.op.bins

    if isinstance(bins, str):
        bins = _load_bins(bins)

    model = build_fm(cfg)
    if art is not None and weights:
        load_state_dict(model, art.state_dict, bins=art.op.bins)
    if getattr(model, "n_bins", 0) > 0:
        if bins is None:
            raise ValueError(
                f"n_bins={model.n_bins} (categorical head) but no bin edges were "
                f"supplied. They are TRAINING-SET STATISTICS, not learned "
                f"parameters, so the model cannot invent them: pass "
                f"bins='/path/to/bins.pt' in the model config, or a `checkpoint` "
                f"that carries them. Derive fresh edges for a new corpus with "
                f"scripts/derive_coeff_bins.py — the ones m113 shipped with came "
                f"from a different noise model. (This used to name `research "
                f"tier1_setup_bins.py`, a gitignored directory that exists on "
                f"one machine: an error message must name something the reader "
                f"has.)")
        apply_bins(model, bins)
    return model


def _load_bins(path):
    """Bin edges from either a bins sidecar or a checkpoint that carries them.

    The sidecar is its own shape — `scripts/derive_coeff_bins.py` writes a bare
    ``edges`` mapping, which is not a checkpoint and has no architecture — so it is read
    here rather than taught to `helix.model.artifact`. Everything else is a
    checkpoint and goes to the one reader.

    The directory test comes FIRST and is not cosmetic: an eval artifact is a
    directory, and `torch.load` on one raises ``IsADirectoryError`` before any
    of the code that would have coped. That is the same shape of bug as
    `load_probe_model` learning about export dirs while its sibling did not.
    """
    import os

    from helix.model.artifact import inspect

    if not os.path.isdir(path):
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "edges" in blob:      # the sidecar
            return blob
    bins = inspect(path).op.bins
    if bins is None:
        raise ValueError(
            f"{path}: no bin edges found. Expected a bins sidecar (an 'edges' "
            f"mapping from scripts/derive_coeff_bins.py), or a checkpoint that carries "
            f"them — an eval artifact does, and a `pimm export` keeps them in "
            f"the weights as the persistent `bin_edges` buffer, so it reports "
            f"none here and needs none.")
    return bins
