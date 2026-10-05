# Patch-shape variant of helix's 8-run training config: 8 wires x 16 band-ticks
# (n_slot = 128, as the default 16 x 8): finer in wire, coarser in time, about the same tokens. The tokenizer
# is interpretation, not compression: no corpus rebuild, only a retrain.
# Lists are replaced wholesale by the _base_ merge, so the transform is restated
# for train, val and test with only pw and pt changed.
import os as _os

_base_ = [_os.path.join(_os.environ["HELIX_ROOT"], "configs", "pimm", "coeff_fm_train_8run.py")]
_tf = [
    dict(type="CoeffTokenize", part="coeff", clean_part="coeff_clean",
         cfg=dict(cell_t="grid_center", pw=8, pt=16), fm_names=True),
    dict(type="CoeffCollect", part="coeff"),
]
model = dict(n_slot=128)
transform = _tf  # the top-level copy too, so nothing reading it sees the base pw and pt
data = dict(train=dict(transform=_tf), val=dict(transform=_tf), test=dict(transform=_tf))
del _os
