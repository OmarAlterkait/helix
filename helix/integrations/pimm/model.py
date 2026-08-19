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
        checkpoint (str | None): a checkpoint from helix's
            ``tools/convert_fm_ckpt.py`` — self-contained, carrying ``config``,
            ``state_dict`` and (for a categorical head) inlined bin ``edges``.
            Its ``config`` supplies the architecture; ``cfg`` overrides fields.
        weights (bool): restore the weights. ``False`` builds the same
            architecture freshly initialised.
        **cfg: architecture kwargs for ``helix.model.build_fm``.
    """
    from helix.model import build_fm

    blob = None      # NOT `blob = bins = None`: that clobbered the caller's bins
    if checkpoint is not None:
        import torch
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "config" not in blob or "state_dict" not in blob:
            raise ValueError(
                f"{checkpoint} is not a converted checkpoint (no config/state_dict). "
                f"Run helix's tools/convert_fm_ckpt.py on the raw research file — "
                f"it also inlines the categorical bin edges, which the raw "
                f"checkpoint does not carry at all.")
        arch = dict(blob["config"])
        if isinstance(arch.get("film"), list):     # torch round-trip makes it a list
            arch["film"] = tuple(arch["film"])
        arch.update(cfg)
        cfg = arch
        if bins is None:
            bins = blob.get("bins")

    if isinstance(bins, str):
        bins = _load_bins(bins)

    model = build_fm(cfg)
    if blob is not None and weights:
        from helix.model.checkpoint import load_converted
        load_converted(model, blob)
    if getattr(model, "n_bins", 0) > 0:
        if bins is None:
            raise ValueError(
                f"n_bins={model.n_bins} (categorical head) but no bin edges were "
                f"supplied. They are TRAINING-SET STATISTICS, not learned "
                f"parameters, so the model cannot invent them: pass "
                f"bins='/path/to/bins.pt' in the model config, or a `checkpoint` "
                f"whose converted blob carries them inline. Derive fresh edges "
                f"for a new corpus with research tier1_setup_bins.py — the ones "
                f"m113 shipped with came from a different noise model.")
        from helix.model.checkpoint import apply_bins
        apply_bins(model, bins)
    return model


def _load_bins(path):
    """Bin edges from either a bins sidecar or a converted checkpoint."""
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "edges" in blob:                       # tier1_setup_bins.py sidecar
        return blob
    if isinstance(blob.get("bins"), dict):    # converted checkpoint
        return blob["bins"]
    raise ValueError(
        f"{path}: no bin edges found (expected an 'edges' key, or a converted "
        f"checkpoint carrying 'bins')")
