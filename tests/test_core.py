"""helix.core: the wavelet transform, robust statistics and hysteresis selection."""
import pytest

torch = pytest.importorskip("torch")

import numpy as np
import pywt
from scipy.ndimage import label

from helix.core import wavelet as W  # noqa: E402
from helix.core import robust  # noqa: E402


# ---- wavelet (helix.core.wavelet) --------------------------------------------------------

# The DWT (stride-2 conv1d, periodization) and its inverse are coefficient-identical to pywt.

@pytest.mark.parametrize("name", ["haar", "db2", "db4", "coif3", "sym4", "bior2.2"])
@pytest.mark.parametrize("n,level", [(64, 3), (1024, 6), (4352, 4), (6144, 10)])
def test_wavedec_matches_pywt(name, n, level):
    x = np.random.default_rng(n).normal(size=(3, n))
    ref = pywt.wavedec(x, name, level=min(level, pywt.dwt_max_level(n, pywt.Wavelet(name).dec_len)),
                       mode="periodization", axis=-1)
    got, lev = W.wavedec(torch.as_tensor(x), name, level, "periodization")
    assert lev == len(ref) - 1
    for r, g in zip(ref, got):
        np.testing.assert_allclose(g.numpy(), r, rtol=0, atol=1e-10)
    if pywt.Wavelet(name).orthogonal:                 # the torch inverse is the adjoint: orthonormal only
        np.testing.assert_allclose(W.reconstruct(got, name, n).numpy(), x, rtol=0, atol=1e-10)


# ---- robust statistics (helix.core.robust) -----------------------------------------------

# Robust statistics (helix.core.robust): the np.median convention, masked medians, MAD sigma,
# clipped pedestal / noise of integer samples.

def test_q50_is_np_median_not_torch_median():
    """`torch.median` returns the LOWER of two middles; `np.median` averages them. Every sigma
    feeds a threshold, so on even-length inputs the tie-break shifts it: torch.median kept 1-2
    extra coefficients per event, always more, never fewer."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert float(robust.q50(x, 0)) == 2.5            # average of the two middles
    assert float(torch.median(x)) == 2.0             # the lower one
    odd = torch.tensor([1.0, 5.0, 3.0])
    assert float(robust.q50(odd, 0)) == float(np.median(odd.numpy()))
    for n in (64, 5000, 5001):                       # selection and sort branches
        m = torch.randn(7, n)
        np.testing.assert_allclose(robust.q50(m, -1).numpy(), np.median(m.numpy(), axis=-1), rtol=1e-6, atol=1e-6)
    nan = torch.tensor([[1.0, float("nan"), 3.0]])
    assert torch.isnan(robust.q50(nan, 1)).all()     # NaN propagates, as in np.median


def test_masked_median_and_mad():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 40)).astype(np.float32)
    v = rng.random((6, 40)) < 0.6
    v[5] = False                                     # no valid entry: NaN
    got = robust.masked_median(torch.as_tensor(x), torch.as_tensor(v), 1).numpy()
    ref = [np.median(r[m]) if m.any() else np.nan for r, m in zip(x, v)]
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)
    s = robust.mad(torch.as_tensor(x), 1).numpy()
    np.testing.assert_allclose(s, np.median(np.abs(x), 1) / 0.6745, rtol=1e-6)
    assert float(robust.mad(torch.zeros(10), 0, 1e-6)) == pytest.approx(1e-6)


def test_clipped_stats_on_integer_codes():
    """Pedestal and noise of rounded Gaussian samples, with a few large pulses: unbiased to a
    few hundredths of a count where the median / MAD of codes are quantized."""
    rng = np.random.default_rng(2)
    mu, sd = 900.3, 1.7
    x = np.round(rng.normal(mu, sd, size=(50, 4000)))
    x[:, 100:130] += 200                              # light / signal, clipped away
    m, s, n = (t.numpy() for t in robust.clipped_stats(x))
    assert abs(np.mean(m) - mu) < 0.02 and abs(np.mean(s) - np.sqrt(sd ** 2 + 1 / 12)) < 0.02
    assert (n > 3900).all()
    m, s, n = robust.clipped_stats(x[:, :5])          # too few samples: no estimate
    assert torch.isnan(m).all() and torch.isnan(s).all()


# ---- hysteresis (helix.core.robust) ------------------------------------------------------

# Hysteresis selection (helix.core.robust): definition, and the flood fill == labelling
# the connected components (scipy.ndimage.label).

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
    k8 = robust.grow(seed, cand, "8").numpy()
    assert k8[1, 1:4].all() and k8[2, 4] and not k8[4, 6]
    k1 = robust.grow(seed, cand, "1d").numpy()
    assert k1[1, 1:4].all() and not k1[2, 4]
    s2 = np.zeros_like(cand); s2[0, 0] = True  # a seed outside cand is kept
    assert robust.grow(s2, cand)[0, 0]


def test_window_any():
    m = np.zeros((2, 20), bool); m[0, 10] = True
    w = robust.window_any(m, 3, 5).numpy()          # out[i] = any m[i-3 : i+6]
    assert np.where(w[0])[0].tolist() == list(range(5, 14)) and not w[1].any()
    rng = np.random.default_rng(1)
    m = rng.random((7, 50)) < 0.05
    for d in (0, 1):
        ref = np.stack([np.convolve(r, np.ones(11), "full")[7:7 + r.size] > 0 for r in (m if d else m.T)])
        assert (robust.window_any(m, 3, 7, d).numpy() == (ref if d else ref.T)).all()


def test_flood_fill_matches_component_labelling():
    rng = np.random.default_rng(0)
    for trial in range(20):
        R, C = rng.integers(1, 100), rng.integers(1, 200)
        a = np.abs(rng.normal(size=(R, C)))
        seed, cand = a > 2.8, a > rng.uniform(0.5, 1.5)
        for conn in ("8", "1d"):
            assert (_grow_by_labels(seed, cand, conn) == robust.grow(seed, cand, conn).numpy()).all()
    seeds = [rng.random((30, n)) < 0.02 for n in (5, 40, 1)]
    cands = [rng.random((30, n)) < 0.5 for n in (5, 40, 1)]
    got = robust.grow_many([torch.as_tensor(x) for x in seeds], [torch.as_tensor(x) for x in cands])
    assert all((_grow_by_labels(s, c, "8") == g.numpy()).all() for s, c, g in zip(seeds, cands, got))
    snake = np.zeros((400, 400), bool); snake[np.arange(400), np.arange(400)] = True
    seed = np.zeros_like(snake); seed[-1, -1] = True
    assert int(robust.grow(torch.as_tensor(seed), torch.as_tensor(snake)).sum()) == 400
