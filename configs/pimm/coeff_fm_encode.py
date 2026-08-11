"""pimm config: run the coefficient FM over the corpus (encode / eval, no training).

Run it from a checkout that has pimm installed — the config file does NOT need to
live inside pimm::

    python -m pimm.cli.train --config-file /path/to/helix/configs/pimm/coeff_fm_encode.py

``custom_imports`` is mmcv's standard out-of-tree hook, which
``pimm.utils.config.Config.fromfile`` already honours. It imports helix's adapter,
which registers ``CoeffTokenize`` / ``CoeffTPCDataset`` / ``Coeff-FM`` into pimm's
registries. pimm itself is unmodified.
"""

custom_imports = dict(
    imports=["helix.integrations.pimm"],
    allow_failed_imports=False,          # fail loudly: a silent miss = "type not found"
)

# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------
# MUST stay 1. The FM has no event separation: attention runs over whatever
# tokens it is given, so batch_size=2 does not fail — it silently trains a model
# whose tokens attend across unrelated events. One event is already ~31-40k
# tokens and saturates the GPU ~8x over, so there is nothing to gain either.
# See MULTI_EVENT_BATCHING.md.
batch_size = 1
batch_size_val = 1
num_worker = 4
mix_prob = 0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
seed = 0

CORPUS = "/sdf/data/neutrino/omara/coeff_tpc/run_0027575715"
CKPT = "/sdf/data/neutrino/omara/archive/fm_m113_converted.pt"

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
# The architecture comes from the checkpoint itself (convert_fm_ckpt.py infers it
# from tensor shapes and cross-checks the recorded metadata), so nothing is
# restated here — a config that restated it could silently disagree.
model = dict(type="Coeff-FM", checkpoint=CKPT, weights=True)

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
)
