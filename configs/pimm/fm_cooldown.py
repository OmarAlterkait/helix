"""pimm config: the cooldown of warmup-stable-decay, from the stable phase's last checkpoint.

The stable phase (``fm.py``) commits to no step count, so it can be extended; this short run
decays the LR as ``max(floor, 1 - sqrt(p))`` over its own steps (``WSDCooldownLR``; pimm
injects ``total_steps``). It anneals on the first ``COOLDOWN_RUNS`` runs of the corpus, one
epoch; cadence is ``"auto"`` as in the base.

No ``WeightEMA``: it would start from a clone of the stable phase's weights, and at decay 0.9999
(a ~10k-step time constant) a cooldown's EMA is mostly those un-annealed weights (~39% after
9.4k steps). A cooldown's weight is ``model/last`` -- ``helix.eval --weights last``, a denoiser's
``model.fm=<run>/model/last``; ``weights="ema"`` on it is refused (no EMA file), not substituted.
"""
# helix is importable here because the launcher exports PYTHONPATH; the sys.path block and
# its HelixPathBootstrap hook belong to the base.
from helix.paths import root as _root                            # noqa: E402
from helix.data.verify import corpus_runs as _corpus_runs     # noqa: E402

_base_ = ["./fm.py"]

COOLDOWN_RUNS = 3
_over = dict(split=_corpus_runs(str(_root("HELIX_CORPUS").parent))[:COOLDOWN_RUNS])
data = dict(train=dict(**_over), val=dict(**_over), test=dict(**_over))

weight = str(_root("HELIX_EXP") / "fm" / "model" / "last")
resume = False
epoch = 1
scheduler = dict(type="WSDCooldownLR", warmup=0, floor=1e-3)
# fm.py's hooks without WeightEMA (a list is replaced, not merged; tests/test_pimm_config.py
# checks the two stay the same otherwise)
hooks = [
    dict(type="HelixPathBootstrap"),
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=1),
    dict(type="InformationWriter", log_frequency=200),
    dict(type="FiniteGuard", check_every=100, max_skips=20),
    dict(type="CoeffFMEvaluator", every_n_steps="auto", max_batches=200),
    dict(type="CheckpointSaver", save_freq="auto"),
]
save_path = str(_root("HELIX_EXP") / "fm_cooldown")

del _root, _corpus_runs, _over
