# Control for coeff_fm_8run_pw8bp2.py: pw8 band tokens and the same location mask
# units, no pooling -- so pooling is judged against its own masking task.
import os as _os

_base_ = [_os.path.join(_os.environ["HELIX_ROOT"], "configs", "pimm", "variants", "coeff_fm_8run_pw8.py")]
model = dict(varlen=True, fused_qk=True, mask_mode="location", mask_cell=(8, 128))
del _os
