"""helix.eval.windows and the fine-grid join in helix.data.cells.

Synthetic events, CPU, numpy only. Each test pins one property the metrics must have for a
patch-size comparison to mean anything: keys round-trip, a perfect prediction localises exactly
and separates every pair, a blurred one does not, windows are not centred on the particle they
score, and a cell finds the tokens covering it with the tokenizer's own arithmetic.
"""

import numpy as np
import pytest

from helix.data import cells as C
from helix.data.denoise import near_windows
from helix.eval import windows as W


def test_keys_round_trip_including_negative_ticks():
    g, fw, ft = np.array([0, 5, 3]), np.array([0, 17, 984]), np.array([-30, 0, 270])
    assert all((a == b).all() for a, b in zip(C.unkey(C.fkey(g, fw, ft)), (g, fw, ft)))


def _event(rng, n_part=40):
    """Compact particles (a few wires x a few ticks) scattered over one plane, as a cells record."""
    g, w, t, q, trk = [], [], [], [], []
    for i in range(n_part):
        cw, ct = rng.integers(40, 900), rng.integers(200, 3000)
        ww, tt = np.meshgrid(np.arange(cw, cw + 3), np.arange(ct, ct + 20), indexing="ij")
        g.append(np.zeros(ww.size, int)); w.append(ww.ravel()); t.append(tt.ravel())
        q.append(np.full(ww.size, 100.0)); trk.append(np.full(ww.size, i))
    g, w, t, q, trk = map(np.concatenate, (g, w, t, q, trk))
    rec = C.cell_truth(g, w, t, q, trk, trk, np.zeros((len(q), 2)))
    rec["trk_energy"] = 0.05 * (np.arange(n_part) + 1)
    cells = C.fkey(g, w // C.FW, t // C.FT)
    noise = C.fkey(np.zeros(500, int), rng.integers(0, 500, 500), rng.integers(0, 250, 500))
    return rec, np.concatenate([cells, noise])


N_CELL = (W.WW // C.FW) * (W.WT // C.FT)


def _meta(tr):
    return [dict(m, ev=0) for m in tr["meta"]]


def _stats(pred_fn, tr):
    return W.window_stats(pred_fn(tr["wq"]), tr["wq"], _meta(tr))


def _truth(seed=0):
    return W.event_truth(*_event(np.random.default_rng(seed)), np.random.default_rng(1))


def test_truth_has_all_window_kinds_and_valid_keys():
    tr = _truth()
    kinds = {m["kind"] for m in tr["meta"]}
    assert {"iso", "bg"} <= kinds
    assert len(tr["wkey"]) == len(tr["wq"]) == len(tr["meta"]) * N_CELL
    for i, m in enumerate(tr["meta"]):
        cells = slice(i * N_CELL, (i + 1) * N_CELL)
        assert np.array_equal(tr["wkey"][cells], W.window_cells(m["g"], m["w0"], m["t0"]))
        if m["kind"] == "bg":
            assert tr["wq"][cells].sum() == 0
        if m["kind"] == "iso":
            assert m["E"][0] > 0


def test_windows_are_not_centred_on_the_particle():
    tr = _truth()
    off = [m["cen"][0][0] - W.WW / 2 for m in tr["meta"] if m["kind"] == "iso"]
    assert np.std(off) > 1.0


def test_perfect_prediction_localises_exactly_and_blur_does_not():
    tr = _truth()
    perfect = _stats(lambda q: q, tr)
    blur = _stats(lambda q: np.full_like(q, q.mean()) + 0.01 * (q > 0), tr)
    loc_p = perfect["loc_w"][np.isfinite(perfect["loc_w"])]
    loc_b = blur["loc_w"][np.isfinite(blur["loc_w"])]
    assert len(loc_p) and max(loc_p) == pytest.approx(0.0, abs=1e-9)
    assert np.median(loc_b) > np.median(loc_p)


def test_auc_counts_a_tie_as_a_half():
    auc = lambda pos, neg: W.auc(W.ranks(np.r_[pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))],  # noqa: E731
                                 np.r_[np.zeros(len(pos)), np.ones(len(neg))])
    assert auc([2, 3], [0, 1]) == 1.0 and auc([0, 1], [2, 3]) == 0.0
    assert auc(np.zeros(50), np.zeros(60)) == 0.5
    assert auc(np.r_[np.zeros(40), np.ones(10)], np.zeros(60)) == pytest.approx(0.6)
    r = W.ranks([0.0, 1.0, 1.0, 2.0])                     # weights: a row counted twice
    assert W.auc(r, np.array([0, 2, 0, 1.0]), np.array([1, 0, 1, 0.0])) == pytest.approx(5 / 6)   # (2 x (1 + 1/2) + 2) / (3 x 2)


def test_a_perfect_prediction_scores_perfectly():
    tr = _truth()
    y = np.log1p(tr["mq"] / C.Q0)
    out = W.score_cells(y, y, np.zeros(len(y), int), np.zeros(len(y), int), tr["wq"], tr["wq"], _meta(tr), n_boot=5)
    assert out["presence_auc"][0] == 1.0 and out["map_r"][0] > 0.99
    effs = [v[0] for k, v in out.items() if k.startswith("floor_eff1pct_") and v[0] == v[0]]
    assert effs and all(v == 1.0 for v in effs)        # every isolated particle beats empty windows


def test_fisher_z_is_within_group():
    rng = np.random.default_rng(0)
    grp = np.repeat(np.arange(10), 50)
    y = rng.standard_normal(500) + 10 * grp                 # between-group variance dominates
    p = y + rng.standard_normal(500)
    z = W.fisher_z(p, y, grp)
    assert len(z) == 10 and np.tanh(np.mean(list(z.values()))) == pytest.approx(0.7, abs=0.1)
    # the per-group definition: a group too small, or constant in either, has no z
    grp2 = np.concatenate([grp, np.full(5, 10), np.full(30, 11), np.full(30, 12), np.full(30, 13)])
    y2 = np.concatenate([y, rng.standard_normal(5), np.ones(30), rng.standard_normal(60)])
    p2 = np.concatenate([p, rng.standard_normal(65), np.full(30, 2.0)])
    want = {k: float(np.arctanh(np.clip(np.corrcoef(p2[grp2 == k], y2[grp2 == k])[0, 1], -0.999, 0.999)))
            for k in range(14) if (grp2 == k).sum() >= 21 and y2[grp2 == k].std() > 0 and p2[grp2 == k].std() > 0}
    got = W.fisher_z(p2, y2, grp2)
    assert set(got) == set(want) == set(range(10)) | {12}
    assert all(got[k] == pytest.approx(want[k], abs=1e-9) for k in want)



def test_bootstrap_weighs_an_event_by_how_often_it_was_drawn():
    v, seen = np.arange(20.0), []

    def stat(m):
        seen.append(m)
        return {"m": float((m * v).sum() / m.sum())}

    pt, lo, hi = W.bootstrap(stat, 20, n_boot=200)["m"]
    assert pt == pytest.approx(9.5) and lo < pt < hi
    assert (seen[0] == 1).all() and all(m.sum() == 20 for m in seen) and any(m.max() > 1 for m in seen)


def test_cell_inputs_matches_the_gather_arithmetic():
    """cell_inputs: the covering tokens and offset features of the tokenizer's own pixel_cells."""
    from helix.model.tokenize import PatchConfig, pixel_cells
    cfg = PatchConfig(cell_t="grid_center")
    rng = np.random.default_rng(0)
    bl = np.array([280, 280, 560, 1120])
    keys = C.fkey(rng.integers(0, 6, 400), rng.integers(0, 900, 400), rng.integers(0, 260, 400))
    g, fw, ft = C.unkey(keys)
    w, t = fw * C.FW, ft * C.FT + C.FT // 2
    pc = pixel_cells(g, w, t, bl, cfg)
    tok_keys = np.unique(pc[rng.random(pc.shape) < 0.5])        # a token set covering about half
    idx, aux = C.cell_inputs(keys, tok_keys, bl, cfg)
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


def test_coeff_cells_are_the_cells_under_the_kept_coefficients():
    from helix.model.tokenize import PatchConfig, tick_of_tau
    cfg = PatchConfig(cell_t="grid_center")
    band, g, wire, tau = np.array([0, 3, 4, 0]), np.array([1, 2, 2, 1]), np.array([10, 11, 12, 10]), np.array([5, 9, 9, 5])
    got = C.coeff_cells(band, g, wire, tau, cfg)
    k = band < cfg.n_bands                                              # D1 is not tokenized
    want = np.unique(C.fkey(g[k], wire[k] // C.FW, np.floor(tick_of_tau(tau[k], g[k], band[k], cfg) / C.FT)))
    np.testing.assert_array_equal(got, want)
    assert len(got) == 2


def test_near_windows_are_empty_but_beside_charge():
    rng = np.random.default_rng(3)
    keys, q = [], []
    for g in (0, 4):                                               # two planes, a few compact blobs each
        for _ in range(6):
            cw, ct = rng.integers(20, 400), rng.integers(20, 200)
            ww, tt = np.meshgrid(np.arange(cw, cw + 3), np.arange(ct, ct + 2), indexing="ij")
            keys.append(C.fkey(np.full(ww.size, g), ww.ravel(), tt.ravel())); q.append(np.full(ww.size, 500.0))
    keys, q = np.concatenate(keys), np.concatenate(q)
    order = np.argsort(keys); keys, q = keys[order], q[order]
    wins = near_windows(keys, q, np.random.default_rng(0), n=20)
    assert 0 < len(wins) <= 20 and {g for g, _, _ in wins} == {0, 4}
    G, CW, CT = C.unkey(keys)
    for g, w0, t0 in wins:
        assert w0 % C.FW == 0 and t0 % C.FT == 0
        fw, ft = w0 // C.FW, t0 // C.FT
        on = G == g
        inside = (CW >= fw - 1) & (CW < fw + W.WW // C.FW + 1) & (CT >= ft - 1) & (CT < ft + W.WT // C.FT + 1)
        near = (CW >= fw - 8) & (CW < fw + W.WW // C.FW + 8) & (CT >= ft - 8) & (CT < ft + W.WT // C.FT + 8)
        assert not (on & inside).any() and (on & near).any()
    assert wins == near_windows(keys, q, np.random.default_rng(0), n=20)


def test_cell_neighbors_centre_is_the_covering_token_and_shifts_by_one_patch():
    from helix.model.tokenize import PatchConfig, cell_key, pixel_cells, unpack_cell_key
    cfg = PatchConfig(cell_t="grid_center")
    bl = np.array([271, 271, 542, 1084, 2168])
    rng = np.random.default_rng(0)
    toks = []
    for b in range(cfg.n_bands):
        for wb in range(2, 8):
            for tb in range(2, 12):
                toks.append(cell_key(1, b, wb * cfg.pw, tb * cfg.pt, cfg))
    ck = np.array(rng.permutation(toks), np.int64)
    keys = C.fkey(np.full(40, 1), rng.integers(40, 100, 40), rng.integers(20, 60, 40))
    nbr = C.cell_neighbors(keys, ck, bl, cfg)
    idx, _ = C.cell_inputs(keys, ck, bl, cfg)
    assert nbr.shape == (40, cfg.n_bands, 9) and np.array_equal(nbr[:, :, 4], idx)
    g, fw, ft = C.unkey(keys)
    pc = pixel_cells(g, fw * C.FW, ft * C.FT + C.FT // 2, bl, cfg)
    for j, (dw, dt) in enumerate([(a, c) for a in (-1, 0, 1) for c in (-1, 0, 1)]):
        hit = nbr[:, :, j] >= 0
        _, _, wb, tb = unpack_cell_key(ck[nbr[:, :, j][hit]])
        _, _, wb0, tb0 = unpack_cell_key(pc[hit])
        assert np.array_equal(wb, wb0 + dw) and np.array_equal(tb, tb0 + dt)
