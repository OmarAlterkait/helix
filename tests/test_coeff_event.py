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


# ---- regression: audit fixes ----------------------------------------------

def _one_coeff_result(basis, gid, nw=8):
    coeffs = [np.zeros((nw, L), np.float32) for L in basis.band_lengths]
    coeffs[1][2, 3] = 1.5
    return {gid: SparseResult(coeffs=coeffs, n_kept=1, n_total=0,
                              sigma_per_band=np.ones(len(basis.band_lengths), np.float32),
                              wavelet=WAVELET, level=LEVEL, mode=MODE)}


def test_plane_gid_beyond_255_no_wrap(tmp_path):
    basis = _basis()
    ce = CoeffEvent.from_sparse_results(_one_coeff_result(basis, 300), basis=basis, event=0)
    assert ce.plane_gid.dtype == np.int32 and int(ce.plane_gid[0]) == 300
    path = tmp_path / "s_coeff_0000.h5"
    write_coeff_shard(path, [ce])
    ce2 = read_coeff_event(path, 0)
    assert int(ce2.plane_gid[0]) == 300                 # was uint8-wrapped to 44
    assert 300 in ce2.reconstruct_images(NT)            # to_band_lists keyed by 300, no KeyError


def test_sigma_none_and_wrong_length_raise():
    basis = _basis()
    coeffs = [np.zeros((4, L), np.float32) for L in basis.band_lengths]
    none_res = {0: SparseResult(coeffs=coeffs, n_kept=0, n_total=0, sigma_per_band=None,
                                wavelet=WAVELET, level=LEVEL, mode=MODE)}
    with pytest.raises(ValueError, match="sigma_per_band is None"):
        CoeffEvent.from_sparse_results(none_res, basis=basis)
    short_res = {0: SparseResult(coeffs=coeffs, n_kept=0, n_total=0,
                                 sigma_per_band=np.ones(1, np.float32),
                                 wavelet=WAVELET, level=LEVEL, mode=MODE)}
    with pytest.raises(ValueError, match="sigma_per_band has"):
        CoeffEvent.from_sparse_results(short_res, basis=basis)


def test_cross_band_wire_mismatch_raises():
    basis = _basis()
    coeffs = [np.zeros((4, L), np.float32) for L in basis.band_lengths]
    coeffs[1] = np.zeros((6, basis.band_lengths[1]), np.float32)   # different wire count
    coeffs[1][5, 0] = 1.0
    res = {0: SparseResult(coeffs=coeffs, n_kept=1, n_total=0,
                           sigma_per_band=np.ones(len(basis.band_lengths), np.float32),
                           wavelet=WAVELET, level=LEVEL, mode=MODE)}
    with pytest.raises(ValueError, match="wires"):
        CoeffEvent.from_sparse_results(res, basis=basis)


def test_basis_digest_mismatch_raises(tmp_path):
    import h5py
    basis = _basis()
    ce = CoeffEvent.from_sparse_results(_one_coeff_result(basis, 0), basis=basis, event=0)
    path = tmp_path / "s_coeff_0000.h5"
    write_coeff_shard(path, [ce])
    with h5py.File(path, "a") as f:
        f["config"].attrs["basis_digest"] = "wrongdigest"
    with pytest.raises(ValueError, match="basis_digest mismatch"):
        read_coeff_event(path, 0)


def test_flatbands_lens_must_match_the_basis():
    """`_flat_rows` derives band and tau from basis.band_lengths and ignores
    FlatBands.lens, so a mismatch silently re-tags every coefficient of the plane
    rather than raising. The list path always checked this; the FlatBands branch
    `continue`d past it — the same `continue` that once dropped sigma_threshold.

    Reachable in production: `event_coeff_event` derives ONE event-wide basis
    from an arbitrary plane (`next(iter(results.values()))`) and applies it to
    every plane, so planes that disagree on padded length hit exactly this.
    """
    import importlib.util
    import numpy as np
    import pytest
    from helix.core.wavelet import FlatBands, SparseResult
    from helix.core.coeff_event import CoeffEvent
    from helix.core.provenance import BasisDescriptor

    lens = (4, 4, 8)
    basis = BasisDescriptor(wavelet="db2", level=2, mode="periodization",
                            n_ticks_raw=16, pad=0, band_lengths=lens,
                            removal={}, threshold={}, sigma_norm=2.6)
    nw = 3

    # The ACCEPT half decodes through `_flat_rows`, a jax-only kernel. The REJECT
    # half below is pure validation and must run everywhere — it is the actual
    # regression, and gating the whole test on jax would hide it in exactly the
    # bare install where nothing else covers this path either.
    try:                       # find_spec itself raises on a broken install
        _has_jax = importlib.util.find_spec("jax") is not None
    except Exception:
        _has_jax = False
    if _has_jax:
        good = FlatBands(np.zeros((nw, sum(lens)), np.float32), list(lens))
        good.flat[0, 0] = 1.0
        res_ok = SparseResult(coeffs=good, n_kept=1, n_total=good.flat.size,
                              sigma_per_band=np.ones(len(lens), np.float32),
                              wavelet="db2", level=2, mode="periodization")
        ce = CoeffEvent.from_sparse_results({0: res_ok}, basis=basis, run="r",
                                            source_file="s.h5", event=0)
        assert ce.n_coeff == 1

    bad_lens = (4, 8, 4)                      # same total width, different split
    bad = FlatBands(np.zeros((nw, sum(bad_lens)), np.float32), list(bad_lens))
    bad.flat[0, 0] = 1.0
    res_bad = SparseResult(coeffs=bad, n_kept=1, n_total=bad.flat.size,
                           sigma_per_band=np.ones(len(bad_lens), np.float32),
                           wavelet="db2", level=2, mode="periodization")
    with pytest.raises(ValueError, match="FlatBands lens"):
        CoeffEvent.from_sparse_results({0: res_bad}, basis=basis, run="r",
                                       source_file="s.h5", event=0)
