"""Make helix's coefficient FM resolvable from a pimm config.

Importing this module registers three names into pimm's registries:

    CoeffTokenize     TRANSFORMS  coeff rows -> patch tokens (helix.model.tokenize)
    CoeffTPCDataset   DATASETS    the corpus, read by pimm-data
    Coeff-FM          MODELS      helix.model.build_fm, + converted checkpoint

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

Split into submodules — ``_compat`` (import-time pimm patches), ``data``
(dataset + terminal transform), ``model`` (the ``Coeff-FM`` entry), ``trainer``
(``FMTrainer`` + the WSD schedules), ``hooks`` (what a run WRITES: resume
bootstrap, provenance, weight EMA) and ``eval`` (what a run MEASURES) — but the
MODULE PATH is unchanged. It had grown to 1089 lines and nine registry entries,
which is a training framework, not an integration seam.

Importing this package still registers everything, because ``custom_imports``
names this path and nothing else: a config says
``imports=["helix.integrations.pimm"]`` exactly as before, and every symbol
that used to be importable from here still is. Order matters — ``_compat``
first, since it patches pimm before anything else touches it.
"""

from helix.integrations.pimm import _compat  # noqa: F401  (import-time patches; MUST be first)
from helix.integrations.pimm.data import CoeffCollect, CoeffTPCDataset
from helix.integrations.pimm.eval import CoeffFMEvaluator, _acc_grid_free, p_dev
from helix.integrations.pimm.hooks import HelixPathBootstrap, WeightEMA
from helix.integrations.pimm.model import CoeffFM, build_coeff_fm, _load_bins
from helix.integrations.pimm.trainer import (FMTrainer, WSDCooldownLR,
                                             WSDStableLR)
from helix.integrations.pimm._compat import _patch_rng_restore_to_cpu
from helix.model.tokenize import CoeffTokenize

__all__ = ["CoeffTokenize", "CoeffCollect", "CoeffTPCDataset", "CoeffFM",
           "build_coeff_fm",
           "FMTrainer", "CoeffFMEvaluator", "WSDStableLR", "WSDCooldownLR",
           "WeightEMA", "HelixPathBootstrap"]
