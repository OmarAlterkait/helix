# Real-data control for helix's 8-run training config: the value target is the
# NOISY input (normalised), not the simulated clean coefficient -- what pretraining
# on detector data would have. Masked prediction must infer it from context. The
# bin edges are the base config's (derived from clean values); noisy values all sit
# beyond the threshold, inside the range those edges cover.
# Lists are replaced wholesale by the _base_ merge, so the transform is restated.
import os as _os

_base_ = [_os.path.join(_os.environ["HELIX_ROOT"], "configs", "pimm", "coeff_fm_train_8run.py")]
_tf = [
    dict(type="CoeffTokenize", part="coeff",
         cfg=dict(cell_t="grid_center"), fm_names=True, noisy_target=True),
    dict(type="CoeffCollect", part="coeff"),
]
transform = _tf
data = dict(train=dict(transform=_tf), val=dict(transform=_tf), test=dict(transform=_tf))
del _os
