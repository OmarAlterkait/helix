"""The corpus and its terminal transform, as pimm registry entries.

``CoeffTokenize`` needs no adapter (it is already a duck-typed transform living
in helix with no pimm import); only its NAME has to reach pimm's registry.
``CoeffCollect`` and ``CoeffTPCDataset`` do adapt, so they are defined here.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from pimm.datasets.builder import DATASETS
from pimm.datasets.transform import Compose
from pimm.datasets.transform.common import TRANSFORMS

from helix.model.tokenize import CoeffTokenize


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

    # 'ident' is kept by DEFAULT. CoeffCollect rebuilds `out` from scratch and
    # copies only `keep` from the top level, so with keep=("name",) the source
    # identity pimm-data attaches — (run, source_file, event), the thing that
    # lets a probe reach simulation truth without rebuilding the corpus — was
    # silently dropped before it ever reached a batch. pimm's collate handles it:
    # str lists stay lists, the int becomes a tensor.
    def __init__(self, part="coeff", keys=None, keep=("name", "ident")):
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
                 max_len=-1, strict_lengths=True, event_range=None,
                 exclude_range=None, holdout=None, split_role=None):
        super().__init__()
        try:
            from pimm_data import CoeffTPCDataset as _DS
        except ImportError:                       # older layout / partial install
            from pimm_data.coeff import CoeffTPCDataset as _DS
        # Split parameters must be FORWARDED. This wrapper re-declares the inner
        # dataset's signature, so anything added there is invisible here until
        # it is listed — and configs resolve THIS class, not the inner one.
        #
        # That has now bitten twice: event_range/exclude_range (fixed in
        # df14602) and then holdout/split_role, which failed the first real
        # 2-GPU launch with "unexpected keyword argument 'holdout'". The unit
        # tests construct the inner dataset directly and cannot see it. If a
        # third split parameter appears, add it here in the same commit.
        self._inner = _DS(data_root=data_root, split=split,
                          dataset_name=dataset_name, modalities=tuple(modalities),
                          transform=None, loop=loop, max_len=max_len,
                          strict_lengths=strict_lengths,
                          event_range=event_range, exclude_range=exclude_range,
                          holdout=holdout, split_role=split_role)
        self.transform = Compose(transform)

    def __len__(self):
        return len(self._inner)

    def get_data(self, idx):
        """The raw nested sample, untransformed."""
        return self._inner.get_data(idx)

    def __getitem__(self, idx):
        return self.transform(self.get_data(idx))
