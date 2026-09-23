"""The smoke config with `fast_path` on — the integration test for it.

`helix/model/fastpath.py` and `helix/model/head.py` were validated against the
shipped path by `tests/test_fast_path.py` (bit-exact trunk) and measured by
`tools/profile/`, but neither exercise runs pimm's Trainer: no hooks, no
evaluator, no checkpoint save/load, no DDP. This config is the cheapest thing
that does all four, so that "it is faster in a benchmark" and "it trains" are
separate claims with separate evidence.

    scripts/helix_run.sh python3 -m pimm.train --config-file \
        configs/pimm/coeff_fm_smoke_fast.py

Compare its loss trace against `coeff_fm_smoke.py`. They will not be identical —
the head re-orders an fp32 sum — but they must agree to the ~1e-5 relative that
`tests/test_fast_path.py` pins, and the gap must not grow with step count.
"""

_base_ = ["./coeff_fm_smoke.py"]

# Merged into the base's `model` dict; everything else is inherited, so the two
# runs differ in exactly one key.
model = dict(fast_path=True)
