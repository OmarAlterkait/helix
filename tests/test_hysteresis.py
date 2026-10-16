"""Hysteresis selection (helix.core.hysteresis): definition, and torch == numpy."""
import numpy as np
import pytest

from helix.core import hysteresis as H


def test_definition_8_connected_and_1d():
    cand = np.zeros((5, 8), bool)
    cand[1, 1:4] = True                       # a run with a seed at its end
    cand[2, 4] = True                         # diagonal continuation (8-connected, not 1d)
    cand[4, 6] = True                         # isolated, no seed
    seed = np.zeros_like(cand); seed[1, 1] = True
    k8 = H.grow(seed, cand, "8")
    assert k8[1, 1:4].all() and k8[2, 4] and not k8[4, 6]
    k1 = H.grow(seed, cand, "1d")
    assert k1[1, 1:4].all() and not k1[2, 4]
    s2 = np.zeros_like(cand); s2[0, 0] = True  # a seed outside cand is kept
    assert H.grow(s2, cand)[0, 0]


def test_window_any():
    m = np.zeros((2, 20), bool); m[0, 10] = True
    w = H.window_any(m, 3, 5)                  # out[i] = any m[i-3 : i+6]
    assert np.where(w[0])[0].tolist() == list(range(5, 14)) and not w[1].any()


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    for trial in range(20):
        R, C = rng.integers(1, 100), rng.integers(1, 200)
        a = np.abs(rng.normal(size=(R, C)))
        seed, cand = a > 2.8, a > rng.uniform(0.5, 1.5)
        for conn in ("8", "1d"):
            ref = H.grow(seed, cand, conn)
            got = H.grow(torch.as_tensor(seed), torch.as_tensor(cand), conn).numpy()
            assert (ref == got).all()
        m = rng.random((R, C)) < 0.05
        for d in (0, 1):
            assert (H.window_any(m, 3, 7, d) == H.window_any(torch.as_tensor(m), 3, 7, d).numpy()).all()
    seeds = [rng.random((30, n)) < 0.02 for n in (5, 40, 1)]
    cands = [rng.random((30, n)) < 0.5 for n in (5, 40, 1)]
    ref = H.grow_many(seeds, cands)
    got = H.grow_many([torch.as_tensor(x) for x in seeds], [torch.as_tensor(x) for x in cands])
    assert all((r == g.numpy()).all() for r, g in zip(ref, got))
    snake = np.zeros((400, 400), bool); snake[np.arange(400), np.arange(400)] = True
    seed = np.zeros_like(snake); seed[-1, -1] = True
    assert int(H.grow(torch.as_tensor(seed), torch.as_tensor(snake)).sum()) == 400
