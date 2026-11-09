"""Make helix's coefficient FM resolvable from a pimm config.

Importing this module registers helix's names into pimm's registries, among them:

    CoeffTokenize     TRANSFORMS  coeff rows -> patch tokens (helix.model.tokenize)
    DenoiseRows       TRANSFORMS  the denoiser's query cells and labels (helix.data.denoise)
    CoeffTPCDataset   DATASETS    the corpus, read by pimm-data
    Coeff-FM          MODELS      helix.model.build_fm, with its bin table
    Coeff-Denoise     MODELS      an FM run plus the denoising head (helix.model.denoise)

**pimm is not modified.** A config pulls this in with mmcv's standard
out-of-tree hook, which ``pimm.utils.config.Config.fromfile`` already honours::

    custom_imports = dict(imports=["helix.integrations.pimm"],
                          allow_failed_imports=False)

That keeps the dependency pointing one way — helix knows how to plug into pimm,
pimm knows nothing about helix — so this survives pimm's own churn (its
``migration/warpconvnet`` branch is mid-flight) without a PR into a framework
that serves many detectors, and without adding helix to its CI.

When that migration lands, its registry resolves plain import paths
(``type: "helix.model.fm:FMModel"``), and the MODELS entry below becomes
unnecessary. The DATASETS entry and CoeffCollect stay, because they adapt rather
than merely name.

Importing this needs pimm AND pimm-data installed; helix's own DSP, tokenizer and
model do not.

The ``Coeff-FM`` and ``Coeff-Denoise`` MODELS entries are defined here. Submodules: ``data``
(dataset + terminal transform), ``trainer`` (``FMTrainer`` + the schedules),
``hooks`` (provenance, the finite guard, the step watchdog). Importing this package registers
all of them, because ``custom_imports`` names this path and nothing else.

There is no in-training evaluation: validation loss does not rank representations, so a run
is scored afterwards (``python -m helix.eval``, and a fine-tune) from its ``model/last``.
"""

from helix.data.denoise import DenoiseRows
from helix.integrations.pimm.data import CoeffCollect, CoeffTPCDataset
from helix.integrations.pimm.hooks import (FiniteGuard, HelixPathBootstrap, HelixProvenance,
                                           StepWatchdog)
from helix.integrations.pimm.trainer import (FMTrainer, WarmupCosineLR,
                                             WSDCooldownLR, WSDStableLR)
from helix.model.tokenize import CoeffTokenize
from pimm.models.builder import MODELS

__all__ = ["CoeffTokenize", "DenoiseRows", "CoeffCollect", "CoeffTPCDataset", "CoeffFM",
           "CoeffDenoise", "build_coeff_fm",
           "FMTrainer", "WarmupCosineLR",
           "WSDStableLR", "WSDCooldownLR", "FiniteGuard", "StepWatchdog", "HelixProvenance",
           "HelixPathBootstrap"]


@MODELS.register_module("Coeff-FM")
class CoeffFM:
    """Factory registered as ``Coeff-FM``.

    A class rather than a function because pimm's registry requires one: ``_register_module``
    raises ``TypeError: module must be a class``, and ``build_from_cfg`` ends in
    ``obj_cls(**args)``. ``__new__`` returns the ``FMModel`` itself, so a config gets a model
    rather than a wrapper.

    ``init`` names an FM run (or one checkpoint of it) whose weights -- its ``model/last`` by
    default -- are loaded into the freshly built model before training: a warm start
    (``configs/pimm/fm_cooldown.py``, a sweep's ``then:``). Not pimm's ``weight=``, for the reason
    on :class:`CoeffDenoise`: that is what pimm's ``scripts/train.sh`` sets on every resumed link,
    so it cannot also carry the initialisation. A resumed link builds from ``init`` and then
    loads its own checkpoint over it. Reading the run back ignores ``init``
    (:func:`helix.model.checkpoint.model_config`): its weights are the run's own.
    """

    def __new__(cls, bins=None, init=None, **cfg):
        model = build_coeff_fm(bins=bins, **cfg)
        if init:
            from helix.model.checkpoint import load_run
            model.load_state_dict(load_run(init, device="cpu").model.state_dict(), strict=True)
        return model


def build_coeff_fm(bins=None, **cfg):
    """The coefficient FM (``helix.model.build_fm``) with the bin table ``bins`` (a path to a
    ``python -m helix.data.bins`` table, or the mapping itself) installed.

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
            "model config. Derive fresh edges for a new corpus with python -m helix.data.bins.")
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
    pimm's ``weight=``: that loads a checkpoint of THIS model and is what pimm's ``scripts/train.sh``
    sets on every resumed link, so it cannot also carry the initialisation.
    """

    def __new__(cls, fm, **cfg):
        from helix.model.denoise import build_denoise
        return build_denoise(fm, **cfg)
