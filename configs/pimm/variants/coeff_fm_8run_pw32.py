# Patch-size variant of helix's 8-run training config: 32 wires per token
# instead of 16 (pt = 8 band-ticks unchanged), so n_slot = 32 * 8. The tokenizer
# is interpretation, not compression: no corpus rebuild, only a retrain.
# Lists are replaced wholesale by the _base_ merge, so the transform is restated
# for train, val and test with only pw changed.
import os as _os

_base_ = [_os.path.join(_os.environ["HELIX_ROOT"], "configs", "pimm", "coeff_fm_train_8run.py")]
_tf = [
    dict(type="CoeffTokenize", part="coeff", clean_part="coeff_clean",
         cfg=dict(cell_t="grid_center", pw=32), fm_names=True),
    dict(type="CoeffCollect", part="coeff"),
]
model = dict(n_slot=256)
transform = _tf  # the top-level copy too, so nothing reading it sees the base pw
data = dict(train=dict(transform=_tf), val=dict(transform=_tf), test=dict(transform=_tf))
del _os
