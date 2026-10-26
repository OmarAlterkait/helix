"""Warmup and cadence follow the run's step count (helix.core.cadence)."""
from helix.core import cadence


def test_warmup_scales_with_width_and_is_at_most_a_fifth_of_the_run():
    assert cadence.warmup_steps(1_000_000, 128, 128) == 500
    assert cadence.warmup_steps(1_000_000, 512, 128) == 1000
    assert cadence.warmup_steps(25, 512, 128) == 5
    assert cadence.warmup_steps(3, 128, 128) == 1


def test_a_short_run_still_evaluates_and_saves():
    assert 25 // cadence.eval_every(25) >= cadence.MIN_EVALS
    assert 25 // cadence.save_every(25) >= cadence.MIN_SAVES
