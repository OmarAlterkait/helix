# Encoder-pooling variant at pw8 (docs/ENCODER_POOLING.md): pw8 band tokens for two
# blocks, then every band of a (plane, 8-wire, 128-tick) location pooled into one
# token for the trunk; masking in the same location units. A config file because
# mask_cell's tuple cannot cross sbatch --export.
import os as _os

_base_ = [_os.path.join(_os.environ["HELIX_ROOT"], "configs", "pimm", "variants", "coeff_fm_8run_pw8.py")]
model = dict(varlen=True, fused_qk=True, mask_mode="location", band_pool=2, mask_cell=(8, 128))
del _os
