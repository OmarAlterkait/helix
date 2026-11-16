"""pimm config: the cooldown of warmup-stable-decay, from the stable phase's last checkpoint.

The stable phase (``fm.py``) commits to no step count, so it can be extended; this short run
decays the LR as ``max(floor, 1 - sqrt(p))`` over its own steps (``WSDCooldownLR``; pimm
injects ``total_steps``). Cadence and hooks are the base's. Its weight is ``model/last``
(``helix.eval``'s default, a denoiser's ``model.fm=<run>``).

Its length is the one every evaluated encoder was cooled for (``lr*_cool``, whose config.py
records it): one pass over the first 57,600 training events in file order, 3,600 steps at 16
GPUs, a tenth of the 35.6k-step stable phase. Uncapped, one epoch of j000000 is 11.9k steps.
"""
from helix.paths import root as _root                            # noqa: E402

_base_ = ["./fm.py"]

data = dict(train=dict(max_len=57600))

# The warm start: the stable run's model/last loaded into the fresh model (Coeff-FM's ``init``).
# Not pimm's weight=, which train.sh sets on every resumed link to the run's OWN checkpoint.
model = dict(init=str(_root("HELIX_EXP") / "fm"))
epoch = 1
scheduler = dict(type="WSDCooldownLR", warmup=0, floor=1e-3)
save_path = str(_root("HELIX_EXP") / "fm_cooldown")

del _root
