"""Make helix's coefficient FM resolvable from a pimm config.

Importing this module registers three names into pimm's registries:

    CoeffTokenize     TRANSFORMS  coeff rows -> patch tokens (helix.model.tokenize)
    CoeffTPCDataset   DATASETS    the corpus, read by pimm-data
    Coeff-FM          MODELS      helix.model.build_fm, with its bin table

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
unnecessary. The TRANSFORMS and DATASETS entries stay, because they adapt rather
than merely name.

Importing this needs pimm AND pimm-data installed; helix's own DSP, tokenizer and
model do not.

Submodules: ``data`` (dataset + terminal transform), ``model`` (the ``Coeff-FM``
entry), ``trainer`` (``FMTrainer`` + the WSD schedules), ``hooks`` (what a run
WRITES: resume bootstrap, provenance, weight EMA, finite guard) and ``eval``
(what a run MEASURES). Importing this package registers all of them, because
``custom_imports`` names this path and nothing else.
"""

from helix.integrations.pimm.data import CoeffCollect, CoeffTPCDataset
from helix.integrations.pimm.eval import CoeffFMEvaluator
from helix.integrations.pimm.hooks import HelixPathBootstrap, WeightEMA
from helix.integrations.pimm.model import CoeffFM, build_coeff_fm
from helix.integrations.pimm.trainer import (FMTrainer, WSDCooldownLR,
                                             WSDStableLR)
from helix.model.tokenize import CoeffTokenize

__all__ = ["CoeffTokenize", "CoeffCollect", "CoeffTPCDataset", "CoeffFM",
           "build_coeff_fm",
           "FMTrainer", "CoeffFMEvaluator", "WSDStableLR", "WSDCooldownLR",
           "WeightEMA", "HelixPathBootstrap"]
