"""DenoiseEvents.rows: the support threshold decides which cells count as 'charged' for
sampling; the regression target stays each cell's full charge."""
from types import SimpleNamespace

import numpy as np

from helix.data.denoise import DenoiseEvents
from helix.probe.resolution import fkey


def _rows(c):
    ds = DenoiseEvents("", [], "", items=[], near_per_sig=0, neg_per_sig=0, cov_per_sig=0, support_min_q=c)
    ce = SimpleNamespace(band=np.array([0]), plane_gid=np.array([0]), wire=np.array([0]), tau=np.array([0]))
    tkeys = fkey(np.zeros(3, np.int64), np.array([10, 20, 30]), np.array([5, 5, 5]))
    tq = np.array([0.5, 30.0, 4000.0])                          # a tail, a faint cell, a bright cell
    rows, y = ds.rows(ce, tkeys, tq, np.random.default_rng(0))
    return set(rows.tolist()), dict(zip(rows.tolist(), y.tolist())), tkeys, tq, ds.q0


def test_tails_are_not_sampled_as_charged_above_the_threshold():
    r0, _, tkeys, _, _ = _rows(0.0)
    r25, _, _, _, _ = _rows(25.0)
    assert set(tkeys.tolist()) <= r0                            # c = 0: every cell with any charge
    assert tkeys[0] not in r25 and {tkeys[1], tkeys[2]} <= r25  # c = 25: the 0.5 e- tail is not 'charged'


def test_regression_target_is_the_full_charge():
    _, y, tkeys, tq, q0 = _rows(25.0)
    for k, q in zip(tkeys.tolist(), tq.tolist()):
        if k in y:
            assert np.isclose(y[k], np.log1p(q / q0), rtol=1e-6)
