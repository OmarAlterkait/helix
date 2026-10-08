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


def test_cell_inputs_matches_the_probe_gather_arithmetic():
    """cell_inputs is the probe's gather, moved: the same covering tokens and the
    same offset features as the inline version eval_resolution used to carry."""
    from helix.model.tokenize import PatchConfig, pixel_cells
    cfg = PatchConfig(cell_t="grid_center")
    rng = np.random.default_rng(0)
    bl = np.array([280, 280, 560, 1120])
    keys = R.fkey(rng.integers(0, 6, 400), rng.integers(0, 900, 400), rng.integers(0, 260, 400))
    g, fw, ft = R.unkey(keys)
    w, t = fw * R.FW, ft * R.FT + R.FT // 2
    pc = pixel_cells(g, w, t, bl, cfg)
    tok_keys = np.unique(pc[rng.random(pc.shape) < 0.5])        # a token set covering about half
    idx, aux = R.cell_inputs(keys, tok_keys, bl, cfg)
    # reference: the inline gather as it stood in scripts/eval_resolution.py
    order = np.argsort(tok_keys); cks = tok_keys[order]
    pos = np.clip(np.searchsorted(cks, pc), 0, len(cks) - 1)
    ref_idx = np.where(cks[pos] == pc, order[pos], -1)
    dec = (1 << np.asarray(cfg.lev)).astype(np.float64); toff = np.asarray(cfg.toff)[g % 3]
    offs = []
    for b in range(cfg.n_bands):
        tau = (t + toff) / dec[b] - cfg.delta[b]
        offs += [(w % cfg.pw + 0.5) / cfg.pw, (tau / cfg.pt) % 1.0]
    x = np.stack(offs, 1).astype(np.float32)
    ang = 2 * np.pi * x[..., None] * np.array([1, 2, 3], np.float32)
    ref_aux = np.concatenate([(ref_idx >= 0).astype(np.float32),
                              np.concatenate([np.sin(ang), np.cos(ang)], -1).reshape(len(keys), -1)], 1)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(aux, ref_aux)
    assert aux.shape == (len(keys), 13 * cfg.n_bands)


def test_near_windows_are_empty_but_beside_charge():
    rng = np.random.default_rng(3)
    keys, q = [], []
    for g in (0, 4):                                               # two planes, a few compact blobs each
        for _ in range(6):
            cw, ct = rng.integers(20, 400), rng.integers(20, 200)
            ww, tt = np.meshgrid(np.arange(cw, cw + 3), np.arange(ct, ct + 2), indexing="ij")
            keys.append(R.fkey(np.full(ww.size, g), ww.ravel(), tt.ravel())); q.append(np.full(ww.size, 500.0))
    keys, q = np.concatenate(keys), np.concatenate(q)
    order = np.argsort(keys); keys, q = keys[order], q[order]
    wins = R.near_windows(keys, q, np.random.default_rng(0), n=20)
    assert 0 < len(wins) <= 20 and {g for g, _, _ in wins} == {0, 4}       # shared over the charged planes
    G, CW, CT = R.unkey(keys)
    for g, w0, t0 in wins:
        assert w0 % R.FW == 0 and t0 % R.FT == 0
        fw, ft = w0 // R.FW, t0 // R.FT
        on = G == g
        inside = (CW >= fw - 1) & (CW < fw + R.WW // R.FW + 1) & (CT >= ft - 1) & (CT < ft + R.WT // R.FT + 1)
        near = (CW >= fw - 8) & (CW < fw + R.WW // R.FW + 8) & (CT >= ft - 8) & (CT < ft + R.WT // R.FT + 8)
        assert not (on & inside).any() and (on & near).any()       # empty with a margin, charge within the pad
    assert wins == R.near_windows(keys, q, np.random.default_rng(0), n=20)  # reproducible


def test_near_scalars_threshold_near_activity():
    W = ([dict(kind="bg", score=s) for s in np.linspace(0, 1, 200)]
         + [dict(kind="bgn", score=s) for s in np.linspace(0, 3, 200)]          # a haze: near windows score higher
         + [dict(kind="iso", eb=1, score=2.0, snr=6.0, clean=True)] * 10)
    s = R.near_scalars(W)
    assert s["near_fpr_at_far1pct"] > 0.5                                       # the far threshold flags the haze
    assert s["snr_eff1pct_far_5-7"] == 1.0 and s["snr_eff1pct_near_5-7"] == 0.0  # credited far, not near
    assert s["near_eff1pct_0.1-0.2"] == 0.0
    assert R.near_scalars([w for w in W if w["kind"] != "bgn"]) == {}
