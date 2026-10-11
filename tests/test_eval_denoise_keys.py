"""The keys scripts/eval_denoise.py added for a trustworthy evaluation: per-plane 1%-FPR
thresholds, the draw aggregate, and the complete-truth lookup (no GPU, no data)."""
import importlib.util
import json
import os
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("eval_denoise", os.path.join(HERE, "..", "scripts", "eval_denoise.py"))
EV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EV)


def _windows(rng, scale=(1.0, 1.0, 1.0), n_noise=2000, n_iso=600):
    W = []
    for p in range(3):
        for kind in ("bg", "bgn"):
            W += [dict(kind=kind, p=p, score=float(scale[p] * x), eb=-1) for x in rng.normal(0, 1, n_noise)]
        W += [dict(kind="iso", p=p, score=float(scale[p] * x), eb=1, snr=6.0, clean=True) for x in rng.normal(2.5, 1, n_iso)]
    return W


def test_pp_matches_pooled_when_scales_agree_and_differs_when_not():
    rng = np.random.default_rng(0)
    W = _windows(rng)
    out = EV.pp_scalars(W)
    # equal scales: per-plane thresholds ~ the pooled one
    allbg = np.quantile([w["score"] for w in W if w["kind"] == "bg"], 0.99)
    pooled = np.mean([w["score"] > allbg for w in W if w["kind"] == "iso"])
    assert abs(out["floor_pp_eff1pct_0.1-0.2"] - pooled) < 0.03
    assert set(f"near_pp_eff1pct_0.1-0.2_{p}" for p in "UVY") <= set(out)
    assert "snr_pp_eff1pct_near_5-7" in out
    # a plane on a 10x scale: per-plane thresholds are scale-free, the pooled one is not
    W2 = _windows(np.random.default_rng(0), scale=(1.0, 1.0, 10.0))
    out2 = EV.pp_scalars(W2)
    assert abs(out2["floor_pp_eff1pct_0.1-0.2"] - out["floor_pp_eff1pct_0.1-0.2"]) < 1e-9


def test_aggregate_combines_bootstrap_and_draw_spread(tmp_path):
    a = types.SimpleNamespace(out=str(tmp_path / "agg.jsonl"))
    rng = np.random.default_rng(1)
    per = []
    for k in range(5):
        v = 0.6 + 0.01 * k
        boots = [{"x": v + e} for e in rng.normal(0, 0.005, 200)]
        per.append(({"x": v, "val_mse": 0.1}, boots))
    res = EV.aggregate(a, "T@draws5", per, dict(checkpoint="c", step=0, train_events=1, init=None, Q0=1.0, n_windows=1))
    m, lo, hi = res["trained"]["x"]
    assert abs(m - 0.62) < 1e-9
    sd = np.std([0.6 + 0.01 * k for k in range(5)], ddof=1)
    half = (hi - lo) / 2
    assert half > sd / np.sqrt(5) - 1e-4                      # at least the draw part
    assert abs(res["draw_sd"]["x"] - round(sd, 5)) < 1e-9
    row = json.loads(open(a.out).read().strip())
    assert row["source"] == "draws5" and row["n_draws"] == 5


def test_complete_truth_lookup_absent_is_zero():
    K = np.array([3, 7, 11], np.int64); Q = np.array([1.0, 2.0, 3.0])
    got = EV.CompleteTruth.lookup(K, Q, np.array([7, 8, 11, 0, 99]))
    assert np.array_equal(got, [2.0, 0.0, 3.0, 0.0, 0.0])
    assert np.array_equal(EV.CompleteTruth.lookup(np.zeros(0, np.int64), np.zeros(0), np.array([1, 2])), [0.0, 0.0])
