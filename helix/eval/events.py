"""The evaluated events: a run's held-out split, tokenized once, with its cells truth.

Every metric reads the same :class:`Event` objects, so an event is read, tokenized and moved to
the device once, and a model's frozen features and the event's map rows and windows are computed
once per event however many metrics use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Event:
    index: int                 # position in the evaluated sequence, 0..n-1
    cells: dict                # helix.data.cells record
    coeff_cells: np.ndarray    # fine cells of the kept coefficients (helix.data.cells.coeff_cells)
    band_lengths: np.ndarray
    cell_key: np.ndarray       # per token, the tokenizer's cell key (numpy)
    batch: dict                # the FM's batch (torch, on the device)
    _feats: dict = field(default_factory=dict, repr=False)
    _truth: tuple = field(default=None, repr=False)         # (seed, truth)

    def features(self, name, model):
        """``model.encode(batch)`` (all tokens visible), float32, cached under ``name``."""
        if name not in self._feats:
            import torch
            from helix.eval.recon import autocast
            with torch.no_grad(), autocast(self.batch["plane_id"].device):
                self._feats[name] = model.encode(self.batch).float()
        return self._feats[name]

    def truth(self, seed, windows):
        """:func:`helix.eval.windows.event_truth` of the event (its windows only if ``windows``),
        drawn with ``[seed, index]``, cached."""
        s, t = self._truth or (None, None)
        if s != seed or (windows and "wkey" not in t):
            from helix.eval.windows import event_truth
            t = event_truth(self.cells, self.coeff_cells, np.random.default_rng([seed, self.index]), windows)
            self._truth = (seed, t)
        return t


def dataset(run):
    """The run's corpus (``data.train`` of its config) restricted to the ``probe`` role of its
    holdout, with the coefficients and their cells truth."""
    from helix.data import CoeffTPCDataset
    d = (run.config.get("data") or {}).get("train") or {}
    return CoeffTPCDataset(data_root=d["data_root"], split=d.get("split", ""),
                           dataset_name=d.get("dataset_name", "sim_wire"), modalities=("coeff", "cells"),
                           holdout=d.get("holdout"), split_role="probe")


def events(ds, tokenizer, device, max_events=None):
    """:class:`Event` for each of the first ``max_events`` events of ``ds``, in its order."""
    import torch
    from helix.data.cells import coeff_cells
    from helix.model.tokenize import CoeffTokenize

    tokenize = CoeffTokenize(part="coeff", cfg=tokenizer)          # the training transform itself
    n = len(ds) if not max_events else min(int(max_events), len(ds))
    for i in range(n):
        s = ds.get_data(i)
        c = dict(s["coeff"])
        meta = c["_meta"]
        fm = tokenize(dict(s))["coeff"]
        B = {k: torch.as_tensor(v).to(device) for k, v in fm.items() if isinstance(v, np.ndarray)}
        B["n_cells"] = B["plane_id"].shape[0]
        yield Event(index=i, cells=s["cells"],
                    coeff_cells=coeff_cells(c["band"], c["plane_gid"], c["wire"], c["tau"], tokenizer),
                    band_lengths=np.asarray(meta["band_lengths"]), cell_key=np.asarray(fm["cell_key"]), batch=B)
