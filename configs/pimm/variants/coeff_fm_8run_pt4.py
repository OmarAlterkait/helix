# Patch-size variant of helix's 8-run training config: 4 band-ticks per token
# instead of 8 (pw = 16 wires unchanged), so n_slot = 16 * 4. The tokenizer
# is interpretation, not compression: no corpus rebuild, only a retrain.
# Lists are replaced wholesale by the _base_ merge, so the transform is restated
# for train, val and test with only pt changed.
import os as _os

_base_ = [_os.path.join(_os.environ["HELIX_ROOT"], "configs", "pimm", "coeff_fm_train_8run.py")]
_tf = [
    dict(type="CoeffTokenize", part="coeff", clean_part="coeff_clean",
         cfg=dict(cell_t="grid_center", pt=4), fm_names=True),
    dict(type="CoeffCollect", part="coeff"),
]
model = dict(n_slot=64)
transform = _tf  # the top-level copy too, so nothing reading it sees the base pt
data = dict(train=dict(transform=_tf), val=dict(transform=_tf), test=dict(transform=_tf))
del _os
