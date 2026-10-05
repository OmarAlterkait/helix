# Band variant of helix's 8-run training config: D2 dropped (A4, D4, D3 only)
# -- 26% fewer tokens; n_slot unchanged (16 x 8 per band). The tokenizer
# is interpretation, not compression: no corpus rebuild, only a retrain.
# Lists are replaced wholesale by the _base_ merge, so the transform is restated
# for train, val and test with only n_bands changed.
import os as _os

_base_ = [_os.path.join(_os.environ["HELIX_ROOT"], "configs", "pimm", "coeff_fm_train_8run.py")]
_tf = [
    dict(type="CoeffTokenize", part="coeff", clean_part="coeff_clean",
         cfg=dict(cell_t="grid_center", n_bands=3), fm_names=True),
    dict(type="CoeffCollect", part="coeff"),
]
model = dict(n_slot=128)
transform = _tf  # the top-level copy too, so nothing reading it sees the base n_bands
data = dict(train=dict(transform=_tf), val=dict(transform=_tf), test=dict(transform=_tf))
del _os
