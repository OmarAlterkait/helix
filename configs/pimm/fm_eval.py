"""pimm config: score a checkpoint on the corpus's probe split, no training step.

    scripts/helix_run.sh python -m pimm.train --config-file configs/pimm/fm_eval.py \\
        --options weight=<checkpoint>

(``scripts/eval_checkpoint.py`` does the same for one checkpoint, without pimm's loop.)
"""
from helix.paths import root as _root                            # noqa: E402

_base_ = ["./fm.py"]

weight = None                           # --options weight=...
resume = False
evaluate = True
save_path = str(_root("HELIX_EXP") / "fm_eval")
batch_size = 1
batch_size_val = 1
batch_size_test = 1
data = dict(val=dict(split_role="probe"), test=dict(split_role="probe"))
hooks = [
    dict(type="HelixPathBootstrap"),
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="InformationWriter", log_frequency=200),
    dict(type="CoeffFMEvaluator", max_batches=None),
]

del _root
