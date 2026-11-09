"""pimm config: a few real FMTrainer steps of a small model — the wiring check.

pimm's Trainer, hooks, optimizer, scheduler and checkpointing against helix's model
and the corpus, on one 11 GB card: the base's model at d=256 (d/d_base = 2, so muP scales),
two blocks, a 32-bin head, 32 training events.

    scripts/helix_run.sh python -m pimm.train --config-file configs/pimm/fm_smoke.py
"""
from helix.paths import root as _root                            # noqa: E402

_base_ = ["./fm.py"]

save_path = str(_root("HELIX_EXP") / "fm_smoke")
num_worker = 2
epoch = 1
model = dict(d=256, blocks=2, dec_blocks=1, heads=4, n_bins=32, compile_blocks=False,
             bins=str(_root("HELIX_ARCHIVE") / "coeff_bins_K32_smoke.pt"))
data = dict(train=dict(max_len=32))

del _root
