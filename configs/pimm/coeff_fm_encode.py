"""pimm config: the coefficient FM over the corpus.

Run from a checkout that has pimm installed — the config need NOT live inside
pimm, because ``Config.fromfile`` takes a path::

    python -m pimm.cli.train --config-file /path/to/helix/configs/pimm/coeff_fm_encode.py

``custom_imports`` is mmcv's standard out-of-tree hook, which pimm's
``Config.fromfile`` already honours. It imports helix's adapter, which registers
``CoeffTokenize`` / ``CoeffCollect`` / ``CoeffTPCDataset`` / ``Coeff-FM`` into
pimm's registries. pimm itself is unmodified.

**Self-contained on purpose — no ``_base_``.** ``_base_`` paths resolve relative
to the config file, so inheriting pimm's ``configs/_base_/default_runtime.py``
from a config that lives in helix would need a brittle relative path across two
repositories. Every field pimm's Trainer reads is therefore set here, and
``tests/test_pimm_config_contract.py`` checks that against the Trainer's actual
source so a missing field fails in CI rather than ten minutes into a launch.
"""

custom_imports = dict(
    imports=["helix.integrations.pimm"],
    allow_failed_imports=False,          # fail loudly: a silent miss = "type not found"
)

CORPUS = "/sdf/data/neutrino/omara/coeff_tpc/run_0027575715"
CKPT = "/sdf/data/neutrino/omara/archive/fm_m113_converted.pt"

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
weight = None
resume = False
evaluate = False          # no coeff evaluator hook yet; see TODO.md 5
test_only = False
seed = 0
save_path = "exp/coeff_fm_encode"

# batch_size MUST stay 1. The FM has no event separation: attention runs over
# whatever tokens it is given, so batch_size=2 does not fail — it silently trains
# a model whose tokens attend across unrelated events. One event is already
# ~31-40k tokens and saturates the GPU roughly 8x over, so there is nothing to
# gain either. See MULTI_EVENT_BATCHING.md.
batch_size = 1
batch_size_val = 1
batch_size_test = 1
num_worker = 4

epoch = 1
eval_epoch = 1
clip_grad = 1.0

sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
matmul_precision = "high"
prefetch_factor = None
detect_anomaly = False
mix_prob = 0
deterministic = False

# muP cannot be expressed as pimm's keyword-matching param_dicts: the hidden
# group's scaling comes from the model's own width multiplier. An FMTrainer
# overriding build_optimizer to call model.param_groups() is the way in
# (TODO.md 2); until then this stays None rather than approximating it wrongly.
param_dicts = None

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
# Architecture comes from the checkpoint (convert_fm_ckpt.py infers it from
# tensor shapes and cross-checks the recorded metadata), so nothing is restated
# here — a config that restated it could silently disagree.
model = dict(type="Coeff-FM", checkpoint=CKPT, weights=True)

optimizer = dict(type="AdamW", lr=3e-4, weight_decay=0.05)
scheduler = dict(type="OneCycleLR", max_lr=3e-4, pct_start=0.05,
                 anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0)

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
# The corpus stores RAW coefficients; the tokenizer normalises with the shard's
# frozen norm_sigma table, which rides along in sample['coeff']['_meta'].
transform = [
    dict(type="CoeffTokenize", part="coeff", clean_part="coeff_clean",
         fm_names=True),
    # Terminal per-event step. Not optional and not cosmetic: it flattens the
    # part to the top level, and converts numpy -> torch so pimm's collate takes
    # its CONCATENATE path. Left as numpy, collate sends the arrays to
    # default_collate, which STACKS them and hands the model an
    # (1, n_cells, n_slot) input it cannot consume. It also drops the int
    # `n_cells` (collate would make it tensor([N]) while make_mask wants an int)
    # and emits no `offset` (pimm's run_step reads input_dict["coord"] whenever
    # an offset is present, and the FM has no coord).
    dict(type="CoeffCollect", part="coeff"),
]

_data_common = dict(
    type="CoeffTPCDataset",
    data_root=CORPUS,
    dataset_name="sim_wire",
    modalities=("coeff", "coeff_clean"),
    transform=transform,
)

data = dict(
    train=dict(**_data_common),
    val=dict(**_data_common),
    test=dict(**_data_common),
)

# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------
# Deliberately NOT pimm's default list: that carries SemSegEvaluator, which reads
# segmentation outputs this model does not produce. MAEEvaluator is not here
# either — it passes return_pred= (which FMModel.forward rejects) and reads
# coord_loss / feat_loss / mask_ratio_actual, which are a point-cloud MAE's
# quantities, not occupancy BCE and coefficient value. Both are opt-in, so the
# fix is to add a coeff evaluator, not to bend those (TODO.md 5).
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="DefaultTrainer")
