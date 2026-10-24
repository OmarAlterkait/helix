"""Fast paths agree with their general forms, on CPU and (when available) CUDA (run by hand on
turing: CI has no GPU): all planes' hysteresis grown at once vs one plane at a time."""
import numpy as np
import pytest
import pywt

torch = pytest.importorskip("torch")

from helix.tpc import threshold_rows as tr  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _t(x, dev):
    return torch.as_tensor(np.asarray(x), device=dev)


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _planes(seed=0, nt=512, rows=(96, 96, 64)):
    """Three planes with board common modes shared across planes, dead rows, pulses."""
    rng = np.random.default_rng(seed)
    labels = [np.arange(r) // 32 for r in rows]                        # 3 boards per plane span all planes
    cm = rng.normal(0, 3, (4, nt)).astype(np.float32) * rng.uniform(0.5, 3, (4, 1)).astype(np.float32)
    imgs = []
    for p, r in enumerate(rows):
        x = rng.normal(0, 2, (r, nt)).astype(np.float32) + cm[labels[p]]
        x[10:14, 200:210] += 40.0                                       # a track segment
        x[50, 300:306] += 12.0                                          # a small deposit
        dead = [3, 70] if r > 70 else [3]
        x[dead] = 0.0                                                   # dead rows
        labels[p] = labels[p].copy(); labels[p][dead] = -1
        imgs.append(x)
    return imgs, labels


def _bands(img, wav="coif3", level=4):
    return [np.asarray(c, np.float32) for c in pywt.wavedec(img, wav, level=level, mode="periodization", axis=-1)]


@pytest.mark.parametrize("dev", DEVICES)
def test_threshold_planes_rows_matches_per_plane(dev):
    imgs, labels = _planes(seed=6)
    jobs = [([_t(c, dev) for c in _bands(im)], dict(kappa=1.1, valid=_t(lab >= 0, dev), floor=2.0))
            for im, lab in zip(imgs, labels)]
    together = tr.threshold_planes_rows(jobs)
    for (b, kw), (o, k, n, s) in zip(jobs, together):
        o1, k1, n1, s1 = tr.threshold_bands_rows(b, **kw)
        assert (k, n) == (k1, n1) and np.array_equal(s, s1)
        for x, y in zip(o, o1):
            assert torch.equal(x, y)
