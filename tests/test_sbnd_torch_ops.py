"""Torch twins of the SBND additions agree with the numpy references.

Covers the labeled coherent gate (masked groups, per-group scale, cross-plane
stacking) and the live-row threshold (plain and hysteresis). Runs on CPU torch and,
when available, CUDA (run by hand on turing: CI has no GPU).

Tolerances: float32 reduction order differs between the backends (as for HELIX's
contiguous gate, ~1e-4 ADC), so values are compared to 1e-3 and discrete outputs
(kept coefficients, seeds) may differ only where a value sits on its threshold.
"""
import numpy as np
import pytest
import pywt

torch = pytest.importorskip("torch")

from helix.tpc import coherent_gate_labels as cgl  # noqa: E402
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
@pytest.mark.parametrize("kw", [dict(sigc_scope="global", tau=0.05),
                                dict(sigc_scope="group", tau=None)])
def test_label_gate_torch_matches_numpy(dev, kw):
    imgs, labels = _planes()
    bp = [_bands(im) for im in imgs]
    ref = cgl.coherent_gate_planes(bp, labels, **kw)
    got = cgl.coherent_gate_planes([[_t(b, dev) for b in bl] for bl in bp], labels, **kw)
    for rp, gp in zip(ref, got):
        for r, g in zip(rp, gp):
            assert isinstance(g, torch.Tensor) and g.device.type == dev
            np.testing.assert_allclose(_np(g), r, rtol=0, atol=1e-3)


@pytest.mark.parametrize("dev", DEVICES)
def test_label_gate_full_groups_fast_path(dev):
    rng = np.random.default_rng(3)
    b = rng.normal(0, 1, (256, 300)).astype(np.float32)
    g = cgl.ChannelGroups(np.arange(256) // 64)
    ref = cgl.gate_band_labeled(b, g, npass=2, tau=0.05)
    got = cgl.gate_band_labeled(_t(b, dev), g, npass=2, tau=0.05)
    np.testing.assert_allclose(_np(got), ref, rtol=0, atol=1e-3)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("kw", [dict(floor=None), dict(floor=None, sigma_scope="channel"),
                                dict(floor=None, sigma_scope="group"), dict(), dict(floor=1.5, sigma_scope="group")])
def test_threshold_rows_torch_matches_numpy(dev, kw):
    imgs, labels = _planes(seed=5)
    im, lab = imgs[0], labels[0]
    valid = lab >= 0
    b = _bands(im)
    kw = dict(kw)
    if kw.get("sigma_scope") == "group":
        kw["groups"] = lab
    ref, kref, nref, sref = tr.threshold_bands_rows(b, kappa=0.9, valid=valid, **kw)
    got, kgot, ngot, sgot = tr.threshold_bands_rows([_t(c, dev) for c in b], kappa=0.9, valid=_t(valid, dev), **kw)
    assert ngot == nref
    assert abs(kgot - kref) <= max(2, 1e-3 * kref)           # only values on the threshold may flip
    np.testing.assert_allclose(sgot, sref, rtol=1e-4)
    for r, g in zip(ref, got):
        g = _np(g)
        both = (r != 0) & (g != 0)
        np.testing.assert_allclose(g[both], r[both], rtol=0, atol=1e-5)


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
    ref = tr.threshold_planes_rows([([_np(c) for c in b], dict(kw, valid=_np(kw["valid"]))) for b, kw in jobs])
    for (o, k, _, _), (r, kr, _, _) in zip(together, ref):
        assert abs(k - kr) <= max(2, 1e-3 * kr)
