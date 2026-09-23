"""Length bucketing must speed a step up without quietly changing the run.

The throughput case is measured on hardware (helix/integrations/pimm/sampler.py
carries the numbers). What is checked here is everything that could make that win
worthless: an event dropped, an event seen twice, a rank left idle, or a
correlation between event size and training step that amounts to an unchosen
curriculum.

These run without pimm's sampler: the class under test is built from whatever
base is handed in, so a stub base with the same contract exercises the ordering
logic directly. That is deliberate -- the ordering is helix's, the epoch and
checkpoint semantics are pimm's, and only the first is this file's business.
"""
from __future__ import annotations

import numpy as np
import pytest

# The ordering logic needs no pimm, but it lives in helix.integrations.pimm, whose
# __init__ imports pimm -- so without pimm this module cannot be imported at all,
# and an unguarded import fails collection for the WHOLE run. Skip instead, like
# the other pimm-gated modules; HELIX_REQUIRE_PIMM=1 makes a missing pimm a
# startup error (tests/conftest.py), so the skip is never silent where it matters.
pytest.importorskip("pimm")
from helix.integrations.pimm.sampler import bucketed_sampler_class  # noqa: E402


class _Base:
    """Minimal stand-in for pimm's StatefulRandomSampler."""

    def __init__(self, data_source, *, num_replicas, rank, seed=0, epoch=0,
                 shuffle=True, drop_last=False):
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = epoch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = int(np.ceil(len(data_source) / num_replicas))
        self.total_size = self.num_samples * num_replicas

    def _build_order(self):           # pimm's plain strided order
        idx = np.arange(len(self.data_source))
        pad = self.total_size - len(idx)
        if pad > 0:
            idx = np.concatenate([idx, idx[:pad]])
        return idx[self.rank:self.total_size:self.num_replicas].tolist()


class _Events:
    """A dataset whose events have a realistic spread of sizes."""

    def __init__(self, n=512, seed=0):
        rng = np.random.default_rng(seed)
        # Lognormal, scaled to the corpus's measured 39k..760k range.
        s = rng.lognormal(mean=12.5, sigma=0.35, size=n)
        self._sizes = np.clip(s, 39_000, 760_000).astype(np.int64)

    def __len__(self):
        return len(self._sizes)

    def event_sizes(self):
        return self._sizes


def _order_all_ranks(ds, R, mega=8, seed=0, epoch=0):
    cls = bucketed_sampler_class(_Base, mega=mega)
    return [cls(ds, num_replicas=R, rank=r, seed=seed, epoch=epoch)._build_order()
            for r in range(R)]


@pytest.mark.parametrize("R", [4, 8, 16])
def test_every_event_is_still_seen_exactly_once(R):
    """Reordering must not lose or duplicate work.

    A sampler that drops events trains on less data than the run claims, and one
    that duplicates them silently reweights the corpus. Either is invisible in
    the loss curve.
    """
    ds = _Events(n=R * 40)
    orders = _order_all_ranks(ds, R)
    flat = sorted(i for o in orders for i in o)
    assert flat == list(range(len(ds))), "bucketing changed the multiset of events"


@pytest.mark.parametrize("R", [4, 8, 16])
def test_ranks_are_balanced_and_disjoint(R):
    """Every rank does the same number of steps, and no two share an event."""
    ds = _Events(n=R * 40)
    orders = _order_all_ranks(ds, R)
    assert len({len(o) for o in orders}) == 1, "ranks got different step counts"
    seen = set()
    for o in orders:
        assert not (seen & set(o)), "two ranks were handed the same event"
        seen |= set(o)


def test_a_step_is_much_more_homogeneous_than_a_random_one():
    """The point of the exercise, stated as a number.

    A DDP step costs its largest event, so what matters is max/mean WITHIN a
    step. Bucketing should collapse it towards 1.
    """
    R, ds = 16, _Events(n=16 * 60)
    sizes = ds.event_sizes()
    orders = _order_all_ranks(ds, R)
    steps = np.array(orders).T                      # (n_steps, R)

    bucketed = np.mean([sizes[s].max() / sizes[s].mean() for s in steps])
    rng = np.random.default_rng(1)
    shuffled = np.array([rng.permutation(len(sizes)) for _ in range(40)])
    random = np.mean([sizes[p[:R]].max() / sizes[p[:R]].mean() for p in shuffled])

    assert bucketed < random, f"bucketed {bucketed:.3f} not below random {random:.3f}"
    # Loose bound: the gain must be real, not a rounding artefact. Measured on
    # this distribution it is ~1.05 against ~1.5.
    assert bucketed < 1.15, f"within-step max/mean still {bucketed:.3f}"


def test_size_is_not_correlated_with_training_step():
    """No accidental curriculum.

    Sorting globally would walk the model from the smallest events to the
    largest -- a curriculum nobody chose, and one that would confound every
    comparison against a non-bucketed run. The megabatch plus the block shuffle
    exist to prevent exactly that, so the correlation between step index and
    batch size must stay near zero.
    """
    R, ds = 8, _Events(n=8 * 200)
    sizes = ds.event_sizes()
    steps = np.array(_order_all_ranks(ds, R, mega=8)).T
    means = np.array([sizes[s].mean() for s in steps])
    r = np.corrcoef(np.arange(len(means)), means)[0, 1]
    assert abs(r) < 0.2, f"batch size correlates with step index (r={r:.3f})"


def test_falls_back_when_the_dataset_cannot_size_itself():
    """A dataset without sizes must train, not crash.

    Bucketing is an optimisation. A sampler is a bad place to fail a run, so an
    unsized dataset gets pimm's ordinary order.
    """
    class _NoSizes(list):
        pass

    ds = _NoSizes(range(64))
    cls = bucketed_sampler_class(_Base, mega=4)
    got = cls(ds, num_replicas=4, rank=1)._build_order()
    assert got == _Base(ds, num_replicas=4, rank=1)._build_order()


def test_a_different_epoch_gives_a_different_order():
    """Epochs must not repeat the same batches in the same order."""
    ds = _Events(n=256)
    a = _order_all_ranks(ds, 8, epoch=0)[0]
    b = _order_all_ranks(ds, 8, epoch=1)[0]
    assert a != b, "bucketing froze the order across epochs"


def test_the_registered_wrapper_forwards_event_sizes():
    """The class pimm's sampler actually sees must expose sizes.

    ``helix.integrations.pimm.data.CoeffTPCDataset`` re-declares the inner
    dataset's interface, and configs -- and the sampler -- resolve THAT class.
    Its own docstring records two earlier cases of something being added to the
    inner dataset and staying invisible here; ``event_sizes`` was the third, and
    it was the worst kind, because the sampler DEGRADES rather than raising:
    bucketing silently did not happen and the A/B compared two identical runs.

    Checked by signature, not by construction, so it costs no corpus: the point
    is that the name exists on the wrapper at all.
    """
    pytest.importorskip("pimm_data")
    from helix.integrations.pimm.data import CoeffTPCDataset as Wrapper
    from helix.data.coeff_dataset import CoeffTPCDataset as Inner

    assert hasattr(Wrapper, "event_sizes"), (
        "the registered wrapper does not forward event_sizes(), so length "
        "bucketing will silently fall back to pimm's ordinary order")
    # Anything else the sampler may come to rely on should fail here too.
    for name in ("event_sizes", "__len__"):
        assert hasattr(Inner, name) and hasattr(Wrapper, name), name
