"""Live-row threshold: parity with the universal threshold and robustness to missing channels."""
import numpy as np
import pytest
import pywt

torch = pytest.importorskip("torch")
from helix.tpc import threshold_rows as tr  # noqa: E402


def threshold_bands_rows(coeffs, **kw):
    out, k, n, s = tr.threshold_bands_rows(coeffs, **kw)
    return [o.numpy() for o in out], k, n, s


def _universal(bands):
    """HELIX's universal rule in numpy: per-band MAD sigma over all rows, every band
    (approximation too) hard-thresholded at sigma sqrt(2 ln N_band)."""
    out = []
    for c in bands:
        s = np.float32(np.median(np.abs(c)) / 0.6745)
        out.append(np.where(np.abs(c) >= s * np.float32(np.sqrt(2 * np.log(c.shape[-1]))), c, 0).astype(np.float32))
    return out, int(sum(np.count_nonzero(o) for o in out))


def _bands(seed=0, rows=400):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 2, (rows, 1024)).astype(np.float32)
    x[50:60, 400:420] += 30
    return [np.asarray(c, np.float32) for c in pywt.wavedec(x, "coif3", level=4, mode="periodization", axis=-1)]


def test_parity_with_threshold_bands_all_rows_valid():
    b = _bands()
    ref, kref = _universal(b)
    got, kgot, _, _ = threshold_bands_rows(b, kappa=1.0, floor=None)          # seeds only = universal
    assert abs(kgot - kref) <= 2                              # float32 sigma: only values on the threshold flip
    for r, g in zip(ref, got):
        both = (r != 0) & (g != 0)
        assert both.sum() >= (r != 0).sum() - 2
        np.testing.assert_array_equal(r[both], g[both])


def test_missing_rows_do_not_collapse_sigma():
    b = _bands(seed=1)
    valid = np.ones(400, bool); valid[:250] = False            # most of the plane missing
    zeroed = [np.where(valid[:, None], c, 0.0).astype(np.float32) for c in b]
    _, k_all_rows = _universal(zeroed)
    _, k_live, _, _ = threshold_bands_rows(zeroed, kappa=1.0, valid=valid)
    _, k_ref, _, _ = threshold_bands_rows([c[valid] for c in b], kappa=1.0)
    assert k_all_rows > 20 * k_live                           # the bug: sigma from mostly-zero rows
    assert k_live == k_ref                                    # the fix: same as the live rows alone


def test_group_and_channel_scope():
    b = _bands(seed=2)
    groups = np.arange(400) // 100
    for scope, g in (("group", groups), ("channel", None)):
        out, k, _, s = threshold_bands_rows(b, kappa=1.0, sigma_scope=scope, groups=g)
        assert 0 < k < sum(c.size for c in b) and np.all(np.isfinite(s[1:]))


def test_hysteresis_is_a_superset_grown_through_the_floor():
    b = _bands(seed=3)
    plain, kp, _, _ = threshold_bands_rows(b, kappa=1.1, floor=None)
    hyst, kh, _, _ = threshold_bands_rows(b, kappa=1.1, floor=2.0)
    assert kh > kp
    for p, h in zip(plain, hyst):
        assert np.array_equal(h[p != 0], p[p != 0])        # every plain survivor is kept
    # a floor above the seed threshold adds nothing
    same, ks, _, _ = threshold_bands_rows(b, kappa=1.1, floor=50.0)
    assert ks == kp


def test_sim_chain_hysteresis_default():
    """The simulation chain thresholds with hysteresis by default: a superset of the
    seeds-only (old universal) survivors, and recorded in the chain digest."""
    from helix.tpc.chain import TPCChain, sim_config
    rng = np.random.default_rng(4)
    img = rng.normal(0, 2, (128, 2701)).astype(np.float32)
    img[40:60, 1000:1012] += 25
    nw = [[128, 128, 128]]
    ch = TPCChain(sim_config(nw, {"coherent": {"method": "none"}}))
    cu = TPCChain(sim_config(nw, {"coherent": {"method": "none"}, "threshold": {"floor": None}}))
    assert ch.cfg["threshold"]["floor"] == 2.0
    ph = ch.process_images([img] * 3, tpc=0)[0]
    pu = cu.process_images([img] * 3, tpc=0)[0]
    assert ph["n_kept"] > pu["n_kept"]
    for a, b in zip(pu["bands"], ph["bands"]):
        a, b = a.numpy(), b.numpy()
        assert np.array_equal(b[a != 0], a[a != 0])
    assert ch.digest != cu.digest
