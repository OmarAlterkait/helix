"""Round-trip identity for the coeff codec — the acceptance gate for the corpus.

Two levels (COEFF_CORPUS_DESIGN.md §7):
  1. codec bit-identity:  read_coeff_event(write_coeff_shard([ce])) == ce  (float32)
  2. decode completeness: reconstruct(read(write(ce))) == reconstruct(ce)  (image-level)
plus provenance: band_lengths drift fails loudly.
"""
from __future__ import annotations

import numpy as np
import pytest

from helix.core.backend import set_backend
from helix.core.provenance import BasisDescriptor, derive_band_lengths
from helix.core.wavelet import SparseResult, ThresholdSpec, sparsify, reconstruct
from helix.core.coeff_event import CoeffEvent
from helix.core.coeff_io import write_coeff_shard, read_coeff_event, read_coeff_shard, n_events

WAVELET, LEVEL, MODE, NT = "db2", 2, "periodization", 64


def _basis():
    bl = derive_band_lengths(WAVELET, LEVEL, MODE, NT)
    return BasisDescriptor(
        wavelet=WAVELET, level=LEVEL, mode=MODE, n_ticks_raw=NT, pad=0,
        band_lengths=bl, removal=dict(kind="gate", kgate=3.0, npass=2),
        threshold=dict(method="universal", func="hard", scale=1.0), sigma_norm=2.6)


def _rand_results(basis, gids, n_wires, rng, *, density=0.3, empty_band=None, empty=False):
    """{gid: SparseResult} with random sparse band arrays consistent with the basis."""
    results = {}
    for gid, nw in zip(gids, n_wires):
        coeffs = []
        for b, L in enumerate(basis.band_lengths):
            c = rng.standard_normal((nw, L)).astype(np.float32)
            if empty or empty_band == b:
                c[:] = 0.0
            else:
                c[rng.random((nw, L)) > density] = 0.0     # sparsify
            coeffs.append(c)
        spb = np.array([np.median(np.abs(c)) / 0.6745 for c in coeffs], np.float32)
        results[gid] = SparseResult(coeffs=coeffs, n_kept=0, n_total=0,
                                    sigma_per_band=spb, wavelet=WAVELET, level=LEVEL, mode=MODE)
    return results


def _event(basis, gids, n_wires, rng, *, event=0, **kw):
    return CoeffEvent.from_sparse_results(
        _rand_results(basis, gids, n_wires, rng, **kw), basis=basis,
        run="run_000", source_file="shard_0000.h5", event=event)


def _assert_event_equal(a: CoeffEvent, b: CoeffEvent):
    for name in ("band", "plane_gid", "wire", "tau", "value", "gids", "n_wires", "sigma_threshold"):
        av, bv = getattr(a, name), getattr(b, name)
        assert av.dtype == bv.dtype, f"{name}: dtype {av.dtype} != {bv.dtype}"
        np.testing.assert_array_equal(av, bv, err_msg=f"{name} differs")
    assert a.run == b.run and a.source_file == b.source_file and a.event == b.event
    assert a.basis.digest() == b.basis.digest()


@pytest.fixture(autouse=True)
def _numpy_backend():
    set_backend("numpy")


# ---- level 1: codec bit-identity -----------------------------------------

@pytest.mark.parametrize("kw", [
    dict(),                       # dense-ish
    dict(density=0.02),           # very sparse
    dict(empty=True),             # empty event (0 coeffs)
    dict(empty_band=0),           # all-zero approx band
    dict(empty_band=2),           # all-zero finest band
])
def test_roundtrip_single_event(tmp_path, kw):
    basis = _basis()
    rng = np.random.default_rng(0)
    gids = np.array([0, 1, 2], np.int32)
    n_wires = np.array([8, 8, 5], np.int32)
    ce = _event(basis, gids, n_wires, rng, **kw)
    path = tmp_path / "s_coeff_0000.h5"
    write_coeff_shard(path, [ce], dataset_name="s")
    _assert_event_equal(read_coeff_event(path, 0), ce)


def test_single_coeff(tmp_path):
    basis = _basis()
    gids, n_wires = np.array([0], np.int32), np.array([4], np.int32)
    coeffs = [np.zeros((4, L), np.float32) for L in basis.band_lengths]
    coeffs[1][2, 3] = 1.2345
    res = {0: SparseResult(coeffs=coeffs, n_kept=1, n_total=0,
                           sigma_per_band=np.ones(len(basis.band_lengths), np.float32),
                           wavelet=WAVELET, level=LEVEL, mode=MODE)}
    ce = CoeffEvent.from_sparse_results(res, basis=basis, event=7)
    assert ce.n_coeff == 1
    path = tmp_path / "s_coeff_0000.h5"
    write_coeff_shard(path, [ce])
    _assert_event_equal(read_coeff_event(path, 0), ce)


def test_multi_event_shard(tmp_path):
    basis = _basis()
    rng = np.random.default_rng(1)
    gids, n_wires = np.array([0, 1, 2], np.int32), np.array([8, 8, 5], np.int32)
    events = [
        _event(basis, gids, n_wires, rng, event=0),
        _event(basis, gids, n_wires, rng, event=1, empty=True),   # empty middle event
        _event(basis, gids, n_wires, rng, event=2, density=0.05),
    ]
    path = tmp_path / "s_coeff_0000.h5"
    write_coeff_shard(path, events, dataset_name="s")
    assert n_events(path) == 3
    for i, e in enumerate(events):
        _assert_event_equal(read_coeff_event(path, i), e)
    for a, b in zip(read_coeff_shard(path), events):
        _assert_event_equal(a, b)


# ---- level 2: decode completeness (reconstruct equality) ------------------

def test_reconstruct_matches(tmp_path):
    basis = _basis()
    rng = np.random.default_rng(2)
    # real DSP: sparsify a synthetic multi-plane event, then round-trip the coeffs
    results = {}
    images = {}
    for gid, nw in ((0, 8), (1, 6)):
        img = rng.standard_normal((nw, NT)).astype(np.float32)
        images[gid] = img
        results[gid] = sparsify(img, wavelet=WAVELET, level=LEVEL, mode=MODE,
                                threshold=ThresholdSpec(method="universal", scale=1.0,
                                                        per_band_sigma=True))
    ce = CoeffEvent.from_sparse_results(results, basis=basis)
    path = tmp_path / "s_coeff_0000.h5"
    write_coeff_shard(path, [ce])
    ce2 = read_coeff_event(path, 0)

    recon_direct = {g: reconstruct(results[g], NT) for g in results}
    recon_roundtrip = ce2.reconstruct_images(NT)
    for g in results:
        np.testing.assert_array_equal(recon_roundtrip[g], recon_direct[g],
                                      err_msg=f"gid {g} reconstruct differs")


# ---- provenance: band_lengths drift fails loudly --------------------------

def test_band_length_drift_raises():
    basis = _basis()
    bad = BasisDescriptor(wavelet=WAVELET, level=LEVEL, mode=MODE, n_ticks_raw=NT, pad=0,
                          band_lengths=tuple(x + 1 for x in basis.band_lengths))
    with pytest.raises(ValueError, match="band_lengths"):
        bad.validate()
