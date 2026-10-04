"""helix.probe.resolution: the fixed-grid resolution metrics.

Synthetic events, CPU, numpy only. Each test pins one property the metrics must
have for a patch-size comparison to mean anything: keys round-trip, a perfect
prediction localises exactly and separates every pair, a blurred one does not,
and windows are not centred on the particle they score.
"""

import numpy as np
import pytest

from helix.probe import resolution as R


def test_keys_round_trip_including_negative_ticks():
    g, fw, ft = np.array([0, 5, 3]), np.array([0, 17, 984]), np.array([-30, 0, 270])
    assert all((a == b).all() for a, b in zip(R.unkey(R.fkey(g, fw, ft)), (g, fw, ft)))


def _event(rng, n_part=40):
    """Compact particles (a few wires x a few ticks) scattered over one plane."""
    g, w, t, q, trk, E = [], [], [], [], [], {}
    for i in range(n_part):
        cw, ct = rng.integers(40, 900), rng.integers(200, 3000)
        ww, tt = np.meshgrid(np.arange(cw, cw + 3), np.arange(ct, ct + 20), indexing="ij")
        g.append(np.zeros(ww.size, int)); w.append(ww.ravel()); t.append(tt.ravel())
        q.append(np.full(ww.size, 100.0)); trk.append(np.full(ww.size, i)); E[i] = 0.05 * (i + 1)
    pix = dict(g=np.concatenate(g), w=np.concatenate(w), t=np.concatenate(t), q=np.concatenate(q),
               trk=np.concatenate(trk), E=E)
    cells = R.fkey(pix["g"], pix["w"] // R.FW, pix["t"] // R.FT)
    noise = R.fkey(np.zeros(500, int), rng.integers(0, 500, 500), rng.integers(0, 250, 500))
    return pix, np.concatenate([cells, noise])


def _stats(pred_fn, tr, pw=16, pt=8):
    meta = tr["meta"]
    for m in meta:
        m["ev"] = 0
    starts = np.searchsorted(tr["wid"], np.arange(len(meta)))
    ends = np.append(starts[1:], len(tr["wid"]))
    return R.window_stats(pred_fn(tr["wq"]), tr["wq"], meta, starts, ends, pw, pt, -2.38, {0: -17.4, 1: 2.6, 2: 5.5})


def test_truth_has_all_window_kinds_and_valid_keys():
    tr = R.event_truth(*_event(np.random.default_rng(0)), np.random.default_rng(1))
    kinds = {m["kind"] for m in tr["meta"]}
    assert {"iso", "bg"} <= kinds
    assert len(tr["wkey"]) == len(tr["wq"]) == len(tr["wid"]) == len(tr["meta"]) * (R.WW // R.FW) * (R.WT // R.FT)
    for m in tr["meta"]:
        if m["kind"] == "bg":
            assert tr["wq"][tr["wid"] == tr["meta"].index(m)].sum() == 0


def test_windows_are_not_centred_on_the_particle():
    tr = R.event_truth(*_event(np.random.default_rng(0)), np.random.default_rng(1))
    off = [m["cen"][0][0] - R.WW / 2 for m in tr["meta"] if m["kind"] == "iso"]
    assert np.std(off) > 1.0


def test_perfect_prediction_localises_exactly_and_blur_does_not():
    tr = R.event_truth(*_event(np.random.default_rng(0)), np.random.default_rng(1))
    perfect = _stats(lambda q: q, tr)
    blur = _stats(lambda q: np.full_like(q, q.mean()) + 0.01 * (q > 0), tr)
    loc_p = [w["loc"][0] for w in perfect if "loc" in w]
    loc_b = [w["loc"][0] for w in blur if "loc" in w]
    assert max(loc_p) == pytest.approx(0.0, abs=1e-9)
    assert np.median(loc_b) > np.median(loc_p)


def test_auc_and_floor_efficiency():
    assert R.auc([2, 3], [0, 1]) == 1.0 and R.auc([0, 1], [2, 3]) == 0.0
    tr = R.event_truth(*_event(np.random.default_rng(0)), np.random.default_rng(1))
    W = _stats(lambda q: q, tr)
    s = R.scalars(np.array([1.0, 0.0]), np.array([1.0, 0.0]), [0.5], W, 0.5)
    effs = [v for k, v in s.items() if k.startswith("floor_eff1pct_") and not np.isnan(v)]
    assert effs and all(v == 1.0 for v in effs)        # every isolated particle beats empty windows
