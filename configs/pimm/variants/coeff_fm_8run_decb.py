# Decoder-budget variant of helix's 8-run training config: every masked A4/D4
# token decoded, a random third of D3/D2, each decoded row weighted 1/p in the
# loss (helix.model.fm dec_frac). Training only: evaluation decodes everything.
# A config file because the per-band tuple cannot cross sbatch --export.
import os as _os

_base_ = [_os.path.join(_os.environ["HELIX_ROOT"], "configs", "pimm", "coeff_fm_train_8run.py")]
model = dict(dec_frac=(1.0, 1.0, 1 / 3, 1 / 3))
del _os
