"""Robust statistics (helix.core.robust): the np.median convention, masked medians, MAD sigma,
clipped pedestal / noise of integer samples."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from helix.core import robust  # noqa: E402


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
