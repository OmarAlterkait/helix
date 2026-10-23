"""Hysteresis selection (helix.core.hysteresis): definition, and the flood fill == labelling
the connected components (scipy.ndimage.label)."""
import numpy as np
import pytest
from scipy.ndimage import label

torch = pytest.importorskip("torch")
from helix.core import hysteresis as H  # noqa: E402

_STRUCT = {"8": np.ones((3, 3), bool), "1d": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], bool)}


def _grow_by_labels(seed, cand, conn):
    lab, n = label(cand | seed, structure=_STRUCT[conn])
    good = np.zeros(n + 1, bool)
    good[lab[seed]] = True
    good[0] = False
    return good[lab]


def test_definition_8_connected_and_1d():
    cand = np.zeros((5, 8), bool)
    cand[1, 1:4] = True                       # a run with a seed at its end
    cand[2, 4] = True                         # diagonal continuation (8-connected, not 1d)
    cand[4, 6] = True                         # isolated, no seed
    seed = np.zeros_like(cand); seed[1, 1] = True
    k8 = H.grow(seed, cand, "8").numpy()
    assert k8[1, 1:4].all() and k8[2, 4] and not k8[4, 6]
    k1 = H.grow(seed, cand, "1d").numpy()
    assert k1[1, 1:4].all() and not k1[2, 4]
    s2 = np.zeros_like(cand); s2[0, 0] = True  # a seed outside cand is kept
    assert H.grow(s2, cand)[0, 0]


def test_window_any():
    m = np.zeros((2, 20), bool); m[0, 10] = True
    w = H.window_any(m, 3, 5).numpy()          # out[i] = any m[i-3 : i+6]
    assert np.where(w[0])[0].tolist() == list(range(5, 14)) and not w[1].any()
    rng = np.random.default_rng(1)
    m = rng.random((7, 50)) < 0.05
    for d in (0, 1):
        ref = np.stack([np.convolve(r, np.ones(11), "full")[7:7 + r.size] > 0 for r in (m if d else m.T)])
        assert (H.window_any(m, 3, 7, d).numpy() == (ref if d else ref.T)).all()


def test_flood_fill_matches_component_labelling():
    rng = np.random.default_rng(0)
    for trial in range(20):
        R, C = rng.integers(1, 100), rng.integers(1, 200)
        a = np.abs(rng.normal(size=(R, C)))
        seed, cand = a > 2.8, a > rng.uniform(0.5, 1.5)
        for conn in ("8", "1d"):
            assert (_grow_by_labels(seed, cand, conn) == H.grow(seed, cand, conn).numpy()).all()
    seeds = [rng.random((30, n)) < 0.02 for n in (5, 40, 1)]
    cands = [rng.random((30, n)) < 0.5 for n in (5, 40, 1)]
    got = H.grow_many([torch.as_tensor(x) for x in seeds], [torch.as_tensor(x) for x in cands])
    assert all((_grow_by_labels(s, c, "8") == g.numpy()).all() for s, c, g in zip(seeds, cands, got))
    snake = np.zeros((400, 400), bool); snake[np.arange(400), np.arange(400)] = True
    seed = np.zeros_like(snake); seed[-1, -1] = True
    assert int(H.grow(torch.as_tensor(seed), torch.as_tensor(snake)).sum()) == 400
