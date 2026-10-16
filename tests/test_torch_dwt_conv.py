"""The torch DWT (stride-2 conv1d, periodization) is coefficient-identical to pywt."""
import numpy as np
import pytest
import pywt

torch = pytest.importorskip("torch")
from helix.core import wavelet_ops_torch as W  # noqa: E402


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
        np.testing.assert_allclose(W._waverec(got, name).numpy(), x, rtol=0, atol=1e-10)
