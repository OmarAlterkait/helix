"""helix.model — the FM MAE, exposing the interface pimm trains against.

See RESEARCH_EXTRACTION_MAP section 5a. helix PROVIDES the model; pimm owns the
loop and the probes. Importing this module needs torch (``helix[torch]``); the
DSP and the tokenizer do not.
"""
from helix.model.fm import FMModel, build_fm
from helix.model.serial import SerialFMModel
from helix.model.loss import losses, losses_fused, losses_cat
from helix.model.mask import make_mask

__all__ = ["FMModel", "SerialFMModel", "build_fm", "losses", "losses_fused",
           "losses_cat", "make_mask"]
