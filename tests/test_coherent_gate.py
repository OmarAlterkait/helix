"""Tests for the qualified coherent gate (R2).

- parity: the 1-pass core reproduces the R2 algorithm transcribed inline (numpy
  self-consistency). NOTE both sides use np.median, so this is NOT a cross-backend
  check against the old torch.median reference (they differ by the even-n tie-break
  ~1e-4); the shipped default is 'quantile' (A-parity) by deliberate choice.
- behavioral: the gate removes block common-mode coherent noise while preserving
  signal, and the 2-pass estimate removes at least as much as 1-pass.
- guards: gate_approx=False leaves the approx band untouched; non-finite fails open.
"""
from __future__ import annotations

import numpy as np
import pytest

from helix.tpc.coherent_gate import coherent_gate

GS = 64


def _ref_smart_gate_1pass(bands, kgate, ksig, gs):
    """Inline numpy transcription of measure_coeffs.smart_gate_bands (median sigc)."""
    out = []
    for b in bands:
        W, Lb = b.shape
        ngf = W // gs
        Ms = []
        if ngf > 0:
            bf = b[:ngf * gs].reshape(ngf, gs, Lb)
            med = np.quantile(bf, 0.5, axis=1)
            resid = bf - med[:, None]
            sg = np.maximum(np.quantile(np.abs(resid).reshape(ngf, -1), 0.5, axis=1) / 0.6745, 1e-6)
            uf = np.abs(resid) <= ksig * sg[:, None, None]
            nuf = uf.sum(1)
            mean = (bf * uf).sum(1) / np.maximum(nuf, 1)
            Ms.append(np.where(nuf > 0, mean, med))
        rem = W - ngf * gs
        if rem > 0:
            blk = b[ngf * gs:]
            med = np.quantile(blk, 0.5, axis=0)
            resid = blk - med
            sg = np.maximum(np.quantile(np.abs(resid).reshape(-1), 0.5) / 0.6745, 1e-6)
            uf = np.abs(resid) <= ksig * sg
            nuf = uf.sum(0)
            mean = (blk * uf).sum(0) / np.maximum(nuf, 1)
            Ms.append(np.where(nuf > 0, mean, med)[None, :])
        M = np.concatenate(Ms, 0)
        sigc = max(np.median(np.abs(M)) / 0.6745, 1e-6)
        Mc = np.where(np.abs(M) < kgate * sigc, M, 0.0)
        n_blocks = (W + gs - 1) // gs
        idx = np.minimum(np.arange(W) // gs, n_blocks - 1)
        out.append(b - Mc[idx])
    return out


def _synthetic_bands(rng, W=128, Lbs=(20, 20, 40)):
    """Coherent (rank-1 per block) + sparse signal + incoherent, per band."""
    bands, coh_parts = [], []
    for Lb in Lbs:
        nb = (W + GS - 1) // GS
        block_cm = rng.standard_normal((nb, Lb)).astype(np.float32) * 2.5   # coherent
        idx = np.minimum(np.arange(W) // GS, nb - 1)
        coherent = block_cm[idx]
        incoh = rng.standard_normal((W, Lb)).astype(np.float32) * 0.3
        signal = np.zeros((W, Lb), np.float32)
        for _ in range(5):                                                  # a few big signal coeffs
            signal[rng.integers(W), rng.integers(Lb)] = rng.uniform(20, 40)
        bands.append((coherent + incoh + signal).astype(np.float32))
        coh_parts.append(coherent)
    return bands, coh_parts


def test_parity_1pass_median():
    rng = np.random.default_rng(0)
    bands, _ = _synthetic_bands(rng)
    got = coherent_gate(bands, group_size=GS, kgate=4.0, ksig=3.0, npass=1, sigc_mode="median")
    ref = _ref_smart_gate_1pass(bands, 4.0, 3.0, GS)
    for g, r in zip(got, ref):                       # float32 result vs float64 reference
        np.testing.assert_allclose(g, np.asarray(r, np.float32), rtol=1e-5, atol=1e-5)


def test_removes_coherent_and_2pass_not_worse():
    rng = np.random.default_rng(1)
    bands, coh = _synthetic_bands(rng)
    one = coherent_gate(bands, kgate=3.0, ksig=3.0, npass=1)
    two = coherent_gate(bands, kgate=3.0, ksig=3.0, npass=2)

    # signal-independent metric: residual coherent = cleaned - ideal, where the
    # ideal cleaned band is (band - true_coherent) = signal + incoherent.
    def coh_residual(cleaned):
        return sum(float(np.mean((c - (b - ch)) ** 2))
                   for c, b, ch in zip(cleaned, bands, coh))

    e_raw = sum(float(np.mean(ch ** 2)) for ch in coh)     # coherent energy present
    e_one, e_two = coh_residual(one), coh_residual(two)
    assert e_one < 0.2 * e_raw, f"1-pass left too much coherent: {e_one} vs {e_raw}"
    assert e_two <= e_one * 1.05, f"2-pass worse than 1-pass: {e_two} vs {e_one}"


def test_gate_approx_false_leaves_approx():
    rng = np.random.default_rng(2)
    bands, _ = _synthetic_bands(rng)
    out = coherent_gate(bands, gate_approx=False)
    np.testing.assert_array_equal(out[0], bands[0].astype(np.float32))
    assert not np.array_equal(out[1], bands[1])          # detail bands still gated


def test_nonfinite_fails_open():
    rng = np.random.default_rng(3)
    bands, _ = _synthetic_bands(rng)
    bands[1][0, 0] = np.nan
    with pytest.warns(UserWarning, match="non-finite"):
        out = coherent_gate(bands, npass=2)
    np.testing.assert_array_equal(out[1], bands[1].astype(np.float32))    # unchanged


def test_kgate_sequence_longer_than_npass_raises():
    """A per-pass kgate longer than npass must NOT be silently truncated.

    `pipeline.py` stamps `removal_json` with the kgate the caller passed, so
    truncation ships coefficients that the recorded provenance does not describe:
    `--kgate 2.5,3.5` at npass=1 built a k=2.5 corpus labelled [2.5, 3.5]. Found
    by an audit of the per-pass option, which had exactly this hole.
    """
    import numpy as np
    import pytest

    from helix.tpc.coherent_gate_ops_numpy import gate_band

    b = np.random.default_rng(0).normal(size=(200, 64)).astype(np.float32)
    # the legitimate uses still work
    gate_band(b, group_size=64, kgate=[2.5, 3.5], ksig=3.0, npass=2)
    gate_band(b, group_size=64, kgate=3.0, ksig=3.0, npass=2)
    gate_band(b, group_size=64, kgate=[2.5], ksig=3.0, npass=2)      # short: padded
    with pytest.raises(ValueError, match="per-pass entries but npass"):
        gate_band(b, group_size=64, kgate=[2.5, 3.5], ksig=3.0, npass=1)
