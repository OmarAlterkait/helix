"""pimm config: the cooldown of warmup-stable-decay, from the stable phase's last checkpoint.

The stable phase (``fm.py``) commits to no step count, so it can be extended; this short run
decays the LR as ``max(floor, 1 - sqrt(p))`` over its own steps (``WSDCooldownLR``; pimm
injects ``total_steps``). It anneals on the first ``COOLDOWN_RUNS`` runs of the corpus, one
epoch; cadence and hooks are the base's. Its weight is ``model/last`` (``helix.eval``'s
default, a denoiser's ``model.fm=<run>``).
"""
from helix.paths import root as _root                            # noqa: E402
from helix.data.verify import corpus_runs as _corpus_runs     # noqa: E402

_base_ = ["./fm.py"]

COOLDOWN_RUNS = 3
data = dict(train=dict(split=_corpus_runs(str(_root("HELIX_CORPUS").parent))[:COOLDOWN_RUNS]))

# The warm start: the stable run's model/last loaded into the fresh model (Coeff-FM's ``init``).
# Not pimm's weight=, which train.sh sets on every resumed link to the run's OWN checkpoint.
model = dict(init=str(_root("HELIX_EXP") / "fm"))
epoch = 1
scheduler = dict(type="WSDCooldownLR", warmup=0, floor=1e-3)
save_path = str(_root("HELIX_EXP") / "fm_cooldown")

del _root, _corpus_runs
