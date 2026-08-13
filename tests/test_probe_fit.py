"""The probe head: event-grouped folds, epoch budget, early stopping, and the metric."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helix.probe.fit import fit_probe, _event_folds
from helix.probe.metrics import fisher_r


def _signal(n_ev=20, per_ev=400, d=8, noise=0.4, seed=0):
    """y is a linear function of X plus per-event offset — recoverable, but only
    if the head is actually trained and the split does not leak."""
    rng = np.random.default_rng(seed)
    X, y, ev, pl = [], [], [], []
    w = rng.normal(size=d)
    for e in range(n_ev):
        x = rng.normal(size=(per_ev, d))
        X.append(x)
        y.append(x @ w + rng.normal(0, noise, per_ev) + 3.0 * e)
        ev.append(np.full(per_ev, e))
        pl.append(rng.integers(0, 6, per_ev))
    return (np.concatenate(X), np.concatenate(y), np.concatenate(ev),
            np.concatenate(pl))


def test_folds_group_whole_events():
    """A row-wise split would leak: patches of one event would sit on both sides."""
    ev = np.repeat(np.arange(10), 50)
    folds = _event_folds(ev, n_folds=5, seed=1)
    for e in np.unique(ev):
        assert len(np.unique(folds[ev == e])) == 1, f"event {e} spans folds"
    assert len(np.unique(folds)) == 5


def test_every_row_gets_an_out_of_fold_prediction():
    X, y, ev, _ = _signal()
    oof, info = fit_probe(X, y, ev, n_folds=4, epochs=6, seeds=(0,), device="cpu")
    assert oof.shape == y.shape
    assert np.isfinite(oof).all()
    assert info["n_events"] == 20 and info["n_folds"] == 4


def test_it_actually_learns_the_signal():
    X, y, ev, pl = _signal(noise=0.3)
    oof, _ = fit_probe(X, y, ev, n_folds=4, epochs=25, seeds=(0,), device="cpu")
    r, rs, info = fisher_r(y, oof, ev, pl, min_rows=20)
    assert r > 0.5, f"probe failed to recover a linear signal: r={r}"
    assert info["n_groups"] > 0


def test_early_stopping_reports_where_it_stopped():
    """Without early stopping an arm that memorises faster scores higher; the
    stopping epoch is reported so a run that never stopped is visible."""
    X, y, ev, _ = _signal(noise=1.5)
    _, info = fit_probe(X, y, ev, n_folds=3, epochs=40, seeds=(0,),
                        patience=2, device="cpu")
    assert 0 < info["mean_stop_epoch"] <= 40
    assert "hit_epoch_cap" in info


def test_refuses_fewer_events_than_folds():
    X, y, ev, _ = _signal(n_ev=3)
    with pytest.raises(ValueError, match="cannot make"):
        fit_probe(X, y, ev, n_folds=5, epochs=2, seeds=(0,), device="cpu")


def test_fisher_r_scores_within_group_not_globally():
    """A per-event offset must not create correlation on its own — that is the
    between-event structure a geometry-only baseline already reproduces."""
    rng = np.random.default_rng(0)
    ev = np.repeat(np.arange(8), 200)
    pl = np.zeros(1600, int)
    y = rng.normal(size=1600) + 10.0 * ev          # big between-event spread
    pred = 10.0 * ev                                # knows ONLY the event
    r, _, _ = fisher_r(y, pred, ev, pl, min_rows=50)
    assert np.isnan(r), "constant-within-group prediction must not score"

    r2, _, _ = fisher_r(y, y, ev, pl, min_rows=50)
    assert r2 > 0.99


def test_fisher_r_reports_what_it_skipped():
    ev = np.repeat(np.arange(4), 10)
    pl = np.zeros(40, int)
    y = np.arange(40.0)
    r, rs, info = fisher_r(y, y, ev, pl, min_rows=100)
    assert np.isnan(r) and info["n_skipped_small"] == 4 and info["n_groups"] == 0
