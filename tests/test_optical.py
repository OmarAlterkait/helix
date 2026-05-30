"""Tests for the optical (PMT) chunk pipeline."""
import numpy as np
import pytest

from helix.core import backend
from helix.core.wavelet import sparsify, reconstruct, ThresholdSpec
from helix.optical.io import pad_batch
from helix.optical.metrics import chunk_metrics


@pytest.fixture(autouse=True)
def _numpy_backend():
    backend.set_backend("numpy")
    yield
    backend.set_backend("numpy")


def _chunks():
    """Synthetic stored chunks: variable length, each a bipolar pulse + noise."""
    rng = np.random.default_rng(1)
    chunks = []
    for n in (4000, 4096, 3500, 5000):
        t = np.arange(n)
        wf = -200 * np.exp(-t / 6.0) + 30 * np.exp(-t / 620.0)
        wf = wf + rng.standard_normal(n).astype(np.float32) * 2.6
        chunks.append(wf.astype(np.float32))
    return chunks


def test_pad_batch_divisible_by_level():
    chunks = _chunks()
    level = 8
    batch, lengths = pad_batch(chunks, level)
    assert batch.shape[0] == len(chunks)
    assert batch.shape[1] % (1 << level) == 0
    assert batch.shape[1] >= max(lengths)
    assert list(lengths) == [len(c) for c in chunks]


def test_chunk_sparsify_roundtrip():
    chunks = _chunks()
    batch, lengths = pad_batch(chunks, 8)
    r = sparsify(batch, wavelet="coif3", level=8, mode="periodization",
                 threshold=ThresholdSpec("topk", keep=0.02))
    recon = np.asarray(reconstruct(r, batch.shape[1]))
    m = chunk_metrics(batch, recon, lengths)
    assert r.compression > 5.0       # top-k 2% -> ~high compression
    assert m["peak_err"] < 0.05      # prompt amplitude preserved
    assert m["rel_rms"] < 0.4        # noise dropped; spike retained


def test_chunk_metrics_masks_padding():
    # padded tail must not affect metrics: identical recon over valid region -> ~0 error
    x = np.zeros((2, 16), np.float32); x[:, :10] = np.arange(10)
    r = x.copy(); r[:, 10:] = 999.0       # garbage in padded region
    m = chunk_metrics(x, r, np.array([10, 10]))
    assert m["rel_rms"] < 1e-6


def test_default_method_is_visushrink():
    from helix.optical import OpticalConfig
    th = OpticalConfig().threshold
    assert th.method == "universal" and th.func == "hard"
    assert th.scale == 1.2        # 1× noise-RMS operating point (locked default)


def test_visushrink_supplied_sigma():
    # VisuShrink with a caller-supplied (padding-independent) sigma reconstructs
    # the pulse and removes noise; padding the batch must not change the result.
    chunks = _chunks()
    batch, lengths = pad_batch(chunks, 8)
    sigma = np.array([np.std(c[-500:]) for c in chunks], np.float32)   # noise from quiet tail
    r = sparsify(batch, wavelet="coif3", level=8, mode="periodization",
                 threshold=ThresholdSpec("universal", func="hard", scale=1.0), sigma=sigma)
    recon = np.asarray(reconstruct(r, batch.shape[1]))
    m = chunk_metrics(batch, recon, lengths, signal_peak_min=20.0)
    assert r.compression > 1.0
    assert m["peak_err"] < 0.05


def test_decomposition_viz_runs():
    import matplotlib; matplotlib.use("Agg")
    from helix.optical import decompose, plot_decomposition
    sig = _chunks()[0]
    bands, scales, names = decompose(sig, wavelet="coif3", level=6)
    assert len(bands) == 7 and names[0].startswith("A")
    for style in ("bars", "map"):
        fig = plot_decomposition(sig, wavelet="coif3", level=6, style=style)
        assert fig is not None
        import matplotlib.pyplot as plt; plt.close(fig)
