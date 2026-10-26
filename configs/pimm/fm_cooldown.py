"""pimm config: the cooldown of warmup-stable-decay, from the stable phase's last checkpoint.

The stable phase (``fm.py``) commits to no step count, so it can be extended; this short run
decays the LR as ``max(floor, 1 - sqrt(p))`` over its own steps (``WSDCooldownLR``; pimm
injects ``total_steps``). It anneals on the first ``COOLDOWN_RUNS`` runs of the corpus, one
epoch; cadence is ``"auto"`` as in the base.
"""
# helix is importable here because the launcher exports PYTHONPATH; the sys.path block and
# its HelixPathBootstrap hook belong to the base.
from helix.paths import root as _root                            # noqa: E402
from helix.data.identity import corpus_runs as _corpus_runs     # noqa: E402

_base_ = ["./fm.py"]

COOLDOWN_RUNS = 3
_over = dict(split=_corpus_runs(str(_root("HELIX_CORPUS").parent))[:COOLDOWN_RUNS])
data = dict(train=dict(**_over), val=dict(**_over), test=dict(**_over))

weight = str(_root("HELIX_EXP") / "fm" / "model" / "last")
resume = False
epoch = 1
scheduler = dict(type="WSDCooldownLR", warmup=0, floor=1e-3)
save_path = str(_root("HELIX_EXP") / "fm_cooldown")

del _root, _corpus_runs, _over
