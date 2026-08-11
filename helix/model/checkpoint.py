"""Reading a converted FM checkpoint's recorded operating point.

Separate from :mod:`helix.integrations.pimm` because none of this needs pimm —
it reads a blob and returns a :class:`~helix.model.tokenize.PatchConfig`. Living
in the pimm adapter made it unimportable without pimm installed, which is the
opposite of what a config recipe needs. Separate from
:mod:`helix.model.tokenize` because that module is deliberately torch-free and a
subprocess test enforces it.
"""

from __future__ import annotations

def patch_config_from_checkpoint(checkpoint):
    """The ``PatchConfig`` a converted checkpoint was TRAINED with.

    The blob records ``tokenizer`` (pw, pt, cell_t) because none of it is
    recoverable from the weights. Nothing read it: ``build_coeff_fm`` took only
    config/state_dict/bins, and every recipe built ``CoeffTokenize`` with no
    ``cfg=``, falling through to ``PatchConfig()`` — whose ``cell_t`` default is
    ``centroid`` while m113 trained on ``grid_center``. Measured on corpus event
    0, ``t_phys`` differs on 94.06% of 30976 cells, mean |delta| 19.5. So the
    encode recipe restored real trained weights and then fed them a time
    coordinate the model had never seen.

    Returns ``None`` when the checkpoint records no tokenizer, so a caller can
    tell "not recorded" from "recorded as the default".
    """
    import torch
    from helix.model.tokenize import PatchConfig

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tok = blob.get("tokenizer")
    if not tok:
        return None
    kw = {k: tok[k] for k in ("pw", "pt", "cell_t") if k in tok}
    if tok.get("n_bands"):
        kw["n_bands"] = int(tok["n_bands"])
    return PatchConfig(**kw)

