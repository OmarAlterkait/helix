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


# ---- R1: the occupancy condition on the refusal -----------------------------

def _rng_band(seed=0, W=1969, Lb=271):
    import numpy as np
    rng = np.random.default_rng(seed)
    b = rng.normal(scale=3.0, size=(W, Lb)).astype(np.float32)
    # inject block-wide common mode (coherent: IDENTICAL on every wire) ...
    for g in range(0, W, 64):
        b[g:g + 64] += rng.normal(scale=7.0, size=(1, Lb)).astype(np.float32)
    # ... and sparse signal on a few wires (contamination: only SOME wires)
    for w in rng.choice(W, 40, replace=False):
        b[w, rng.choice(Lb, 5, replace=False)] += 120.0
    return b


def test_tau_none_reproduces_the_legacy_gate_bitwise():
    """`tau=None` must be bit-identical to the magnitude-only rule.

    Corpora built before 2026-08-16 used it, so this is what makes them
    reproducible after the occupancy condition landed.
    """
    import numpy as np

    from helix.tpc.coherent_gate_ops_numpy import _block_common_mode, _sigc, gate_band

    b = _rng_band()
    for npass in (1, 2):
        got = gate_band(b, group_size=64, kgate=3.0, ksig=3.0, npass=npass, tau=None)
        # recompute the legacy rule inline, independent of the implementation
        W = b.shape[0]
        nblk = (W + 63) // 64
        idx = np.minimum(np.arange(W) // 64, nblk - 1)
        sm = np.zeros_like(b, dtype=bool)
        for p in range(npass):
            M = _block_common_mode(b, 3.0, sm, 64)
            Mc = np.where(np.abs(M) < 3.0 * _sigc(M, "quantile"), M, 0.0)
            ref = b - Mc[idx]
            if p + 1 < npass:
                from helix.tpc.coherent_gate_ops_numpy import _detect_signal
                sm = _detect_signal(ref, 3.0, 64)
        assert np.array_equal(got, ref.astype(b.dtype, copy=False)), \
            f"tau=None diverged from the legacy rule at npass={npass}"


def test_tau_only_ever_subtracts_more_single_pass():
    """At npass=1 the occupancy condition can only RE-ADMIT refusals.

    It ANDs an extra term onto the refusal, so a cell the legacy rule already
    subtracted must come out identical; tau can only turn a refusal INTO a
    subtraction. This pins that tau changes the DECISION and never the estimator.

    Stated for npass=1 deliberately. At npass=2 it does NOT hold, and that is
    correct rather than a bug: pass 1's output feeds `_detect_signal`, so
    re-admitting a cell changes pass 2's signal mask and therefore pass 2's
    common mode everywhere. R1's effect is consequently not confined to the
    cells it re-admits -- see test_tau_npass2_propagates_through_detection.
    """
    import numpy as np

    from helix.tpc.coherent_gate_ops_numpy import gate_band

    b = _rng_band(seed=1)
    legacy = gate_band(b, group_size=64, kgate=3.0, ksig=3.0, npass=1, tau=None)
    r1 = gate_band(b, group_size=64, kgate=3.0, ksig=3.0, npass=1, tau=0.05)
    diff = legacy != r1
    assert diff.any(), "tau=0.05 changed nothing on a band with coherent noise"
    assert np.array_equal(legacy[~diff], r1[~diff])
    assert np.allclose(legacy[diff], b[diff]), \
        "tau altered a cell the legacy rule had already subtracted"


def test_tau_npass2_propagates_through_detection():
    """At npass=2 the change is NOT confined to the re-admitted cells.

    Documented because it is surprising and it bounds what the parity tests can
    claim: pass 1's cleaned band seeds `_detect_signal`, so a re-admitted cell
    shifts pass 2's mask and hence pass 2's estimate for its whole block.
    """
    import numpy as np

    from helix.tpc.coherent_gate_ops_numpy import gate_band

    b = _rng_band(seed=1)
    legacy = gate_band(b, group_size=64, kgate=3.0, ksig=3.0, npass=2, tau=None)
    r1 = gate_band(b, group_size=64, kgate=3.0, ksig=3.0, npass=2, tau=0.05)
    diff = legacy != r1
    touched = ~np.isclose(legacy[diff], b[diff])
    assert touched.any(), (
        "expected npass=2 to alter cells the legacy rule had already subtracted, "
        "via the pass-1 -> _detect_signal -> pass-2 feedback")


def test_occupancy_uses_the_real_block_width():
    """The trailing partial block must be judged on its own width, not 64.

    Planes are 1969 and 1443 wires, so the last block holds 49 or 35. Scoring its
    occupancy against 64 would make the same number of flagged wires look like
    less contamination there than in a full block.
    """
    import numpy as np

    from helix.tpc.coherent_gate_ops_numpy import _block_common_mode

    W = 1969                                   # 30 full blocks + 49
    b = np.zeros((W, 8), np.float32)
    sm = np.zeros_like(b, dtype=bool)
    b[-49:, :] = 1.0
    b[-1, :] = 500.0                           # one clear outlier in the partial block
    _, occ = _block_common_mode(b, 3.0, sm, 64, return_occ=True)
    assert occ.shape[0] == 31
    assert occ[-1, 0] == pytest.approx(1.0 / 49.0, rel=1e-6), \
        f"partial block occupancy scored against the wrong width: {occ[-1, 0]}"


def test_backends_agree_to_a_documented_tolerance_not_exactly():
    """numpy and torch agree to ~1e-4 ADC, and the docs must not claim more.

    float32 reduction order differs between the backends, so the masked mean and
    the MAD differ slightly; discrete threshold comparisons then turn some of
    those into different decisions, and the block broadcast applies each decision
    to 64 wires. Measured max |numpy - torch| on a real event is 1.2e-4 ADC,
    unchanged by npass or tau. This pins a bound rather than asserting equality,
    so a genuine regression is caught without the test failing on arithmetic that
    was never going to match.
    """
    import numpy as np
    import torch

    from helix.tpc.coherent_gate_ops_numpy import gate_band as gb_np
    from helix.tpc.coherent_gate_ops_torch import gate_band as gb_t

    b = _rng_band(seed=3)
    for npass in (1, 2):
        for tau in (None, 0.05):
            a = gb_np(b, group_size=64, kgate=3.0, ksig=3.0, npass=npass, tau=tau)
            c = gb_t(torch.as_tensor(b), group_size=64, kgate=3.0, ksig=3.0,
                     npass=npass, tau=tau).numpy()
            worst = float(np.abs(a - c).max())
            assert worst < 1e-2, (
                f"backends diverged by {worst:.3e} ADC at npass={npass}, tau={tau} "
                f"- far beyond the ~1e-4 reduction-order floor, so something "
                f"structural differs rather than the arithmetic")
