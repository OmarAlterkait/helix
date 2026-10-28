"""The MODELS entries: ``Coeff-FM`` (helix's FM with its bin table installed) and
``Coeff-Denoise`` (an FM run plus the denoising head)."""

from __future__ import annotations

from pimm.models.builder import MODELS


@MODELS.register_module("Coeff-FM")
class CoeffFM:
    """Factory registered as ``Coeff-FM``.

    A class rather than a function because pimm's registry requires one: ``_register_module``
    raises ``TypeError: module must be a class``, and ``build_from_cfg`` ends in
    ``obj_cls(**args)``. ``__new__`` returns the ``FMModel`` itself, so a config gets a model
    rather than a wrapper.

    Initialising from a checkpoint is pimm's ``weight=`` (``configs/pimm/fm_cooldown.py``);
    reading a trained run is :func:`helix.model.checkpoint.load_run`.
    """

    def __new__(cls, bins=None, **cfg):
        return build_coeff_fm(bins=bins, **cfg)


def build_coeff_fm(bins=None, **cfg):
    """The coefficient FM (``helix.model.build_fm``) with the bin table ``bins`` (a path to a
    ``scripts/derive_coeff_bins.py`` table, or the mapping itself) installed.

    ``FMModel.forward(batch) -> dict`` already satisfies pimm's Trainer contract
    (``output_dict["loss"]``), so nothing is wrapped.
    """
    from helix.model import build_fm
    from helix.model.checkpoint import apply_bins

    model = build_fm(cfg)              # raises on a key it does not know
    if bins is None:
        raise ValueError(
            "no bin edges were supplied. They are TRAINING-SET STATISTICS, not learned "
            "parameters, so the model cannot invent them: pass bins='/path/to/bins.pt' in the "
            "model config. Derive fresh edges for a new corpus with scripts/derive_coeff_bins.py.")
    if isinstance(bins, str):
        from helix.data.bins import load
        bins = load(bins)
    apply_bins(model, bins)
    return model


@MODELS.register_module("Coeff-Denoise")
class CoeffDenoise:
    """Factory registered as ``Coeff-Denoise``: a pretrained FM run plus the denoising head
    (:func:`helix.model.denoise.build_denoise`; ``configs/pimm/denoise.py``).

    ``fm`` names the FM run (architecture, tokenizer and, with ``pretrained``, weights). Not
    pimm's ``weight=``: that loads a checkpoint of THIS model and is what ``scripts/train.sh``
    sets on every resumed link, so it cannot also carry the initialisation.
    """

    def __new__(cls, fm, **cfg):
        from helix.model.denoise import build_denoise
        return build_denoise(fm, **cfg)
