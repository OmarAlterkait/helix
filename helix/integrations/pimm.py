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
"""

from __future__ import annotations

from pimm.datasets.builder import DATASETS
from pimm.datasets.transform import Compose
from pimm.datasets.transform.common import TRANSFORMS
from pimm.models.builder import MODELS
from torch.utils.data import Dataset

from helix.model.tokenize import CoeffTokenize

__all__ = ["CoeffTokenize", "CoeffCollect", "CoeffTPCDataset", "build_coeff_fm"]

# The tokenizer needs no adapter — it is already a duck-typed transform
# (``scope`` + ``__call__(dict) -> dict``), which is why it can live in helix
# with no pimm import. Only the NAME has to reach pimm's registry.
TRANSFORMS.register_module(module=CoeffTokenize, name="CoeffTokenize")


@TRANSFORMS.register_module()
class CoeffCollect:
    """Tokenised part -> the flat tensor dict ``FMModel.forward`` consumes.

    The terminal per-event transform, and it exists for three specific reasons
    that only show up when you run pimm's actual collate over a real sample:

    1. **Flatten.** ``CoeffTokenize`` leaves tokens nested under its part, but
       the model reads ``plane_id``/``inp``/... at the top level.

    2. **Tensorise.** pimm's ``collate_fn`` CONCATENATES tensor leaves
       (``torch.cat``) but sends anything else to ``default_collate``, which
       STACKS. Handing it numpy therefore produced ``inp`` of shape
       ``(1, n_cells, n_slot)`` — a spurious batch dimension the model cannot
       consume. Converting here puts us on the concatenating path, which is also
       the one that stays correct if batching ever arrives.

    3. **Drop ``n_cells``.** It is an int, so collate turns it into
       ``tensor([30976])`` while ``make_mask`` does ``torch.rand(n)``.
       ``FMModel.forward`` derives it from ``plane_id.shape[0]`` anyway, which is
       also correct for a concatenated batch.

    Deliberately does NOT emit ``offset``. pimm's ``run_step`` does
    ``if "offset" in input_dict: input_dict["coord"].shape[0]`` — an offset
    without a ``coord`` raises KeyError *after* the forward. The FM has no
    ``coord`` and, having no event separation, requires ``batch_size=1`` anyway
    (see MULTI_EVENT_BATCHING.md).
    """

    scope = "sample"

    #: int/scalar sample fields that must not reach the model as 0-d tensors
    DROP = ("n_cells",)

    def __init__(self, part="coeff", keys=None, keep=("name",)):
        self.part = part
        self.keys = tuple(keys) if keys else None
        self.keep = tuple(keep)

    def __call__(self, data):
        import numpy as np
        import torch

        sub = data.get(self.part)
        if sub is None:
            raise KeyError(
                f"CoeffCollect: no part {self.part!r} in the sample (have "
                f"{sorted(data)}) — it must run AFTER CoeffTokenize")
        out = {}
        for k, v in sub.items():
            if k.startswith("_") or k in self.DROP:
                continue
            if self.keys is not None and k not in self.keys:
                continue
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(np.ascontiguousarray(v))
        for k in self.keep:                     # carry the event id for seeding/logging
            if k in data:
                out[k] = data[k]
        return out


@DATASETS.register_module()
class CoeffTPCDataset(Dataset):
    """The wavelet-coefficient corpus as a pimm dataset.

    Wraps ``pimm_data.CoeffTPCDataset`` as a pure reader (``transform=None``) and
    runs pimm's own transform pipeline on the raw nested sample — the same shape
    pimm's ``lucid_event_ssl.py`` already uses to consume pimm-data, so a config
    author sees one transform registry.

    Each sample is per-coefficient ROWS, not tokens::

        {'coeff': {'band','plane_gid','wire','tau','value','_meta'}, 'name', 'split'}

    ``_meta`` carries the shard tables the tokenizer needs (``gids``,
    ``n_wires``, ``band_lengths``, ``norm_sigma``), so a DataLoader worker is
    self-sufficient.

    **batch_size must be 1.** The FM has no event separation — attention runs
    over whatever tokens it receives — so a larger batch silently trains a model
    whose tokens attend across unrelated events. See ``MULTI_EVENT_BATCHING.md``.
    """

    def __init__(self, data_root, split="", dataset_name="coeff_tpc",
                 modalities=("coeff", "coeff_clean"), transform=None, loop=1,
                 max_len=-1, strict_lengths=True):
        super().__init__()
        try:
            from pimm_data import CoeffTPCDataset as _DS
        except ImportError:                       # older layout / partial install
            from pimm_data.coeff import CoeffTPCDataset as _DS
        self._inner = _DS(data_root=data_root, split=split,
                          dataset_name=dataset_name, modalities=tuple(modalities),
                          transform=None, loop=loop, max_len=max_len,
                          strict_lengths=strict_lengths)
        self.transform = Compose(transform)

    def __len__(self):
        return len(self._inner)

    def get_data(self, idx):
        """The raw nested sample, untransformed."""
        return self._inner.get_data(idx)

    def __getitem__(self, idx):
        return self.transform(self.get_data(idx))


@MODELS.register_module("Coeff-FM")
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

    blob = bins = None
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
        model.load_state_dict(blob["state_dict"], strict=True)
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
        model.set_bins(bins["edges"], bins.get("cent_asinh"), bins.get("cent_lin"))
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
