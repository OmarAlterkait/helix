"""Score the m113 reference on a probe split, through ITS OWN bins.

m113 is the strongest prior model we have: 1,010,000 steps over 161,000 events
(`provenance.train`), against ~113-119k steps here -- roughly a 9x training
budget, at IDENTICAL architecture (d=512, blocks=12, dec_blocks=4, heads=8,
n_bins=128, dec_mode=cross, serial, muP d_base=128) and an identical tokenizer
(pw=16 pt=8 cell_t=grid_center lev/delta/toff/sigma_norm all equal to our
default -- checked field by field, not assumed).

**Weights and bins both come from the checkpoint, not from `weight=`.** The
converted blob carries `config`, `state_dict` and inlined `bins`, so
`build_coeff_fm(checkpoint=...)` restores the architecture and the head's own bin
partition together. Passing a corpus-derived table instead would decode m113's
logits against edges it never trained on -- the same defect that made the first
F0 comparison here invalid.

Its inlined bins carry `cent_asinh`/`cent_lin` but NO `cent_ratio`, so the ratio
centroids are DERIVED. That under-reads sum|centroid| by ~2.7-3.0% per band and
depresses charge closure and var_expl slightly; there is no measured table for
this partition to substitute, so it is a floor on m113, not a correction to make.

Two runs are meaningful and they answer different questions:

  CORPUS=<r1>   the common ground every other row here is on, but a DOMAIN SHIFT
                for m113 -- it never saw R1's coherent-noise gate.
  CORPUS=<pre-R1>  m113's home turf, and the number that says what it is worth
                when the data matches what it was built for.

Set both with COEFF_EVAL_CORPUS; neither is "the" answer alone.
"""

_base_ = ["./coeff_fm_eval_probe.py"]

import os as _os
CKPT = _os.environ.get("COEFF_EVAL_CKPT",
                       "/sdf/data/neutrino/omara/archive/fm_m113_artifact")
_CORPUS = (_os.environ.get("COEFF_EVAL_CORPUS")
           or "/sdf/data/neutrino/omara/coeff_tpc_r1/run_0027575715")
del _os

# bins=None is load-bearing: build_coeff_fm falls back to the artifact's own
# inlined table only when nothing else is supplied.
model = dict(type="Coeff-FM", checkpoint=CKPT, weights=True, bins=None)

# No cfg.weight -- CheckpointLoader has nothing to load and says so; the weights
# arrive through model.checkpoint.
weight = None

_eval_data = dict(split_role="probe", data_root=_CORPUS)
data = dict(val=_eval_data, test=_eval_data)
