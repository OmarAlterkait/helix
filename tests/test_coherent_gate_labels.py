"""Label-grouped coherent gate: parity with the contiguous-block gate and with a
per-group reference on groups with excluded channels."""
import numpy as np
import pytest

from helix.tpc.coherent_gate_ops_numpy import gate_band
from helix.tpc.coherent_gate_labels import (ChannelGroups, gate_band_labeled, coherent_gate_labeled,
                                            coherent_gate_planes)


def _band(W=256, L=300, seed=0, gs=64):
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 1, (W, L)).astype(np.float32)
    b += np.repeat(rng.normal(0, 3, (W // gs + 1, L)), gs, axis=0)[:W].astype(np.float32)   # block common mode
    b[40:50, 100:110] += 25.0                                                               # "signal"
    return b


@pytest.mark.parametrize("tau", [None, 0.05])
@pytest.mark.parametrize("npass", [1, 2])
def test_contiguous_labels_match_contiguous_gate(tau, npass):
    b = _band()
    ref = gate_band(b, group_size=64, kgate=3.0, ksig=3.0, npass=npass, tau=tau)
    got = gate_band_labeled(b, ChannelGroups(np.arange(256) // 64), kgate=3.0, ksig=3.0, npass=npass, tau=tau)
    np.testing.assert_array_equal(got, ref)


def test_permuted_groups_equal_contiguous_on_permuted_rows():
    b = _band(seed=1)
    perm = np.random.default_rng(2).permutation(256)
    labels = np.empty(256, int); labels[perm] = np.arange(256) // 64     # rows perm[0:64] form group 0, ...
    bp = np.empty_like(b); bp[perm] = b                                    # b's row i goes to perm[i]
    got = gate_band_labeled(bp, ChannelGroups(labels), npass=2, tau=0.05)
    ref = gate_band(b, group_size=64, kgate=3.0, ksig=3.0, npass=2, tau=0.05)
    np.testing.assert_allclose(got[perm], ref, rtol=0, atol=1e-5)     # float32 summation order only


def _reference_masked(b, labels, kgate=3.0, ksig=3.0, npass=2, tau=0.05, scope="global"):
    """Slow per-group reference (loop over groups, np.quantile on valid rows)."""
    b = b.astype(np.float32)
    gids = [g for g in np.unique(labels) if g >= 0]
    rows = [np.where(labels == g)[0] for g in gids]
    mad = lambda x: max(np.quantile(np.abs(x), 0.5) / 0.6745, 1e-6)  # noqa: E731

    def cm(sm):
        M, O = [], []
        for r in rows:
            x = b[r]; med = np.quantile(x, 0.5, axis=0); res = x - med
            sg = mad(res.ravel()); uf = (np.abs(res) <= ksig * sg) & ~sm[r]; nuf = uf.sum(0)
            M.append(np.where(nuf > 0, (x * uf).sum(0) / np.maximum(nuf, 1), med)); O.append(1 - nuf / len(r))
        return np.array(M), np.array(O)

    sm = np.zeros(b.shape, bool); out = b.copy(); occ1 = None
    for p in range(npass):
        M, O = cm(sm)
        if p == 0:
            occ1 = O
        if scope == "group":
            sc = np.array([max(float(np.quantile(np.abs(m), 0.5)) / 0.6745, 1e-6) for m in M])[:, None]
        else:
            sc = max(float(np.quantile(np.abs(M), 0.5)) / 0.6745, 1e-6)
        refuse = (np.abs(M) >= kgate * sc) & ((occ1 > tau) if tau is not None else True)
        Mc = np.where(refuse, 0.0, M)
        out = b.copy()
        for k, r in enumerate(rows):
            out[r] = b[r] - Mc[k]
        if p + 1 < npass:
            sm = np.zeros(b.shape, bool)
            for r in rows:
                c = np.abs(out[r]); sm[r] = c > ksig * mad(c.ravel())
    return out


def test_groups_with_excluded_channels_match_reference():
    b = _band(seed=3)
    labels = np.arange(256) // 64
    labels[[3, 70, 71, 200]] = -1            # dead channels: groups of 63, 62, 64, 63
    got = gate_band_labeled(b, ChannelGroups(labels), npass=2, tau=0.05)
    ref = _reference_masked(b, labels)
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-5)
    np.testing.assert_array_equal(got[[3, 70, 71, 200]], b[[3, 70, 71, 200]])     # excluded rows untouched


def test_cross_plane_split():
    rng = np.random.default_rng(4)
    planes = [[rng.normal(size=(n, 50)).astype(np.float32) for _ in range(3)] for n in (64, 64, 128)]
    labels = [np.repeat([0, 1], 32), np.repeat([0, 1], 32), np.repeat([0, 1], 64)]   # 2 boards across 3 planes
    out = coherent_gate_planes(planes, labels, npass=2, tau=0.05)
    stacked = [np.concatenate([p[i] for p in planes]) for i in range(3)]
    ref = coherent_gate_labeled(stacked, np.concatenate(labels), npass=2, tau=0.05)
    for i in range(3):
        np.testing.assert_array_equal(np.concatenate([o[i] for o in out]), ref[i])


@pytest.mark.parametrize("tau", [None, 0.05])
def test_per_group_scale_matches_reference(tau):
    rng = np.random.default_rng(5)
    b = rng.normal(0, 1, (256, 300)).astype(np.float32)
    amp = np.array([0.5, 1.0, 2.0, 4.0])                      # groups with different coherent levels
    b += np.repeat(rng.normal(0, 1, (4, 300)) * amp[:, None], 64, axis=0).astype(np.float32)
    labels = np.arange(256) // 64
    labels[[3, 200]] = -1
    for lab in (labels, np.arange(256) // 64):                 # with and without excluded rows
        got = gate_band_labeled(b, ChannelGroups(lab), npass=2, tau=tau, sigc_scope="group")
        ref = _reference_masked(b, lab, tau=tau, scope="group")
        np.testing.assert_allclose(got, ref, rtol=0, atol=2e-5)


def test_uniform_signal_kept_with_group_scale_and_no_tau():
    """A signal identical on every row of one group (e.g. a track at constant drift time)
    is subtracted by the occupancy rule but kept by the per-group magnitude test."""
    rng = np.random.default_rng(6)
    b = rng.normal(0, 1, (256, 300)).astype(np.float32)
    b += np.repeat(rng.normal(0, 1.5, (4, 300)), 64, axis=0).astype(np.float32)
    b[64:128, 150] += 40.0                                     # uniform over group 1
    G = ChannelGroups(np.arange(256) // 64)
    kept_tau = gate_band_labeled(b, G, npass=2, tau=0.05)[64:128, 150].mean()
    kept_pb = gate_band_labeled(b, G, npass=2, tau=None, sigc_scope="group")[64:128, 150].mean()
    assert kept_tau < 10.0 < 35.0 < kept_pb
