"""Acceptance checks for a BUILT corpus — semantic, not just structural.

The lesson these encode: a shard whose ``sigma_threshold`` (and hence
``norm_sigma``) was entirely zero passed round-trip, identity, digest and
co-support checks. Nothing asserted that a value must be *populated*, must
*vary*, or must *reconstruct the physics*. These do.

  1. degenerate-field audit  — catches all-zero / constant / out-of-range content
  2. backend equivalence     — numpy vs jax on the SAME noisy input
  3. physics acceptance (F0) — reconstruct a built shard vs the clean truth
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import h5py

# find_spec, NOT pytest.importorskip: importorskip raises Skipped at DECORATION
# time, which aborts collection of the WHOLE module. Used in a decorator it
# silently dropped all 8 non-jax tests here — the audit_shard corruption sweep,
# the real-data F0 acceptance, and both normalisation tests — in any environment
# without jax, which includes a bare `pip install -e .` (jax is an extra).
def _jax_available():
    # find_spec, not import: importing jax here would initialise CUDA at
    # COLLECTION time, which turns a bad driver into a collection error instead
    # of a skip. try/except because find_spec itself raises on a broken install.
    try:
        return importlib.util.find_spec("jax") is not None
    except Exception:
        return False


_HAS_JAX = _jax_available()

from helix.core.backend import set_backend
from helix.core.coeff_io import audit_shard, read_coeff_event
from helix.tpc.config import DetectorConfig
from helix.tpc.corpus import build_corpus

GIDS = [0, 1]
NW = [8, 6]
NT = 102


def _config():
    return DetectorConfig(num_time_steps=NT, wavelet="db2", dwt_level=2,
                          group_size=64, threshold_kappa=1.0)


def _plane_fn(ev):
    rng = np.random.default_rng(1000 + ev)
    clean, noisy = {}, {}
    for gid, w in zip(GIDS, NW):
        c = np.zeros((w, NT), np.float32)
        for _ in range(3):
            c[rng.integers(w), rng.integers(NT)] = rng.uniform(20, 40)
        clean[gid] = c
        nb = (w + 64 - 1) // 64
        coh = (rng.standard_normal((nb, NT)).astype(np.float32) * 2.0)[
            np.minimum(np.arange(w) // 64, nb - 1)]
        noisy[gid] = (c + coh + rng.standard_normal((w, NT)).astype(np.float32) * .5)
    return noisy, clean


# ---- 1. degenerate-field audit --------------------------------------------

def test_audit_passes_a_good_shard(tmp_path):
    set_backend("numpy")
    build_corpus(range(4), _plane_fn, _config(), tmp_path, dataset_name="cx")
    assert audit_shard(tmp_path / "cx_coeff_0000.h5", strict=False) == []


@pytest.mark.parametrize("corrupt,expect", [
    ("sigma_zero", "sigma_threshold"),
    ("sigma_constant", "bit-identical"),
    ("norm_zero", "norm_sigma"),
    ("tau_oob", "tau"),
    ("offset_break", "event_offset"),
    ("one_plane_sigma_zero", "plane gid 1 is ALL ZERO"),
    ("gids_unsorted", "strictly increasing"),
    ("gids_duplicate", "strictly increasing"),
    ("digest_break", "coord_digest"),
])
def test_audit_catches_corruption(tmp_path, corrupt, expect):
    """Each of these is a shape the codec would happily write and every
    structural test would happily accept."""
    set_backend("numpy")
    build_corpus(range(4), _plane_fn, _config(), tmp_path, dataset_name="cx")
    p = tmp_path / "cx_coeff_0000.h5"
    with h5py.File(p, "a") as f:
        if corrupt == "sigma_zero":
            f["coord/sigma_threshold"][...] = 0.0
        elif corrupt == "sigma_constant":
            s = f["coord/sigma_threshold"][:]
            f["coord/sigma_threshold"][...] = np.repeat(s[:1], s.shape[0], axis=0)
        elif corrupt == "norm_zero":
            f["config/norm_sigma"][...] = 0.0
        elif corrupt == "tau_oob":
            t = f["coord/tau"][:]; t[0] = 10 ** 6
            f["coord/tau"][...] = t
        elif corrupt == "offset_break":
            o = f["coord/event_offset"][:]; o[-1] += 7
            f["coord/event_offset"][...] = o
        elif corrupt == "one_plane_sigma_zero":
            # ONE dead plane in an otherwise healthy table: the old global
            # nanmax check passed this happily.
            s = f["coord/sigma_threshold"][:]; s[:, 1, :] = 0.0
            f["coord/sigma_threshold"][...] = s
        elif corrupt == "gids_unsorted":
            f["config/gids"][...] = np.array([1, 0], np.int32)
        elif corrupt == "gids_duplicate":
            f["config/gids"][...] = np.array([0, 0], np.int32)
        elif corrupt == "digest_break":
            d = f["coord/coord_digest"][:]; d[0] ^= np.uint64(0xDEADBEEF)
            f["coord/coord_digest"][...] = d
    probs = audit_shard(p, strict=False)
    assert any(expect in s for s in probs), f"audit missed {corrupt}: {probs}"
    with pytest.raises(ValueError, match="shard audit failed"):
        audit_shard(p)


def test_clean_shard_is_values_only_and_pairs_by_digest(tmp_path):
    """The clean target is co-supported, so it stores no coords (13 of every 34
    bytes in a pair). What makes that safe is the digest: a MISPAIRED clean shard
    must raise, because silent misalignment of every target row is precisely the
    failure this codebase has already shipped twice."""
    from helix.core.coeff_io import coord_digest, write_coeff_shard
    set_backend("numpy")
    build_corpus(range(4), _plane_fn, _config(), tmp_path, dataset_name="cx")
    p_noisy = tmp_path / "cx_coeff_0000.h5"
    p_clean = tmp_path / "cx_coeff_clean_0000.h5"

    # values-only really is smaller, and really has no coords
    with h5py.File(p_clean, "r") as f:
        assert not bool(f["config"].attrs["has_coords"])
        assert set(f["coord"]) == {"event_offset", "coord_digest", "sigma_threshold"}
    assert p_clean.stat().st_size < p_noisy.stat().st_size

    # correct pairing round-trips
    ce = read_coeff_event(p_clean, 0, coords_from=p_noisy)
    noisy0 = read_coeff_event(p_noisy, 0)
    np.testing.assert_array_equal(ce.band, noisy0.band)
    np.testing.assert_array_equal(ce.tau, noisy0.tau)

    # a shard from a DIFFERENT build has the same shapes but different coords:
    # the digest is what catches it. Positional length checks would not.
    other = tmp_path / "other"
    build_corpus(range(4, 8), _plane_fn, _config(), other, dataset_name="cx")
    p_other = other / "cx_coeff_0000.h5"
    with pytest.raises(ValueError, match="coord_digest mismatch|not co-supported"):
        read_coeff_event(p_clean, 0, coords_from=p_other)

    # and the digest is genuinely order-sensitive (a permutation is not a no-op)
    b = np.array([0, 1], np.uint8); g = np.array([0, 1], np.int32)
    w = np.array([2, 3], np.int32); t = np.array([4, 5], np.int32)
    assert coord_digest(b, g, w, t) != coord_digest(b[::-1], g[::-1], w[::-1], t[::-1])


def test_noise_mode_is_recorded_on_disk(tmp_path):
    """Two shards built with different noise are otherwise indistinguishable —
    a silently mixed-noise corpus. /config/noise_json pins it."""
    import json as _json
    set_backend("numpy")
    build_corpus(range(2), _plane_fn, _config(), tmp_path, dataset_name="cx",
                 noise=dict(kind="white", coherent=True, incoherent=True))
    with h5py.File(tmp_path / "cx_coeff_0000.h5", "r") as f:
        assert _json.loads(f["config"].attrs["noise_json"])["kind"] == "white"
    with h5py.File(tmp_path / "cx_coeff_clean_0000.h5", "r") as f:
        assert _json.loads(f["config"].attrs["noise_json"])["kind"] == "white"


# ---- 2. backend equivalence (same noisy input, numpy vs jax) --------------

def _has(bk):
    try:
        return importlib.util.find_spec(bk) is not None
    except Exception:
        return False


@pytest.mark.parametrize("bk", ["jax", "torch"])
def test_backend_equivalent_on_same_input(tmp_path, bk):
    """Every backend must produce the SAME corpus from the same input.

    Noise is fixed (the RNG streams differ per backend), so this compares the DSP
    + extraction + assembly only. It also compares the CLEAN target and
    sigma_threshold, which an earlier version did not — it looked at the noisy
    events alone, so a clean-side or sigma-side divergence would have passed.

    torch caught a real one that way: its band sigma used ``torch.median`` (the
    LOWER of two middles) where numpy uses ``np.median`` (their average). A MAD
    feeds a threshold, and a threshold is discontinuous, so torch kept 1-2 extra
    coefficients per event — always more, never fewer. See backend.torch_q50.
    """
    if not _has(bk):
        pytest.skip(f"{bk} not installed")
    if bk == "torch":
        import torch
        if not torch.cuda.is_available():
            pytest.skip("torch has no CUDA device")

    def to_dev(fn):
        def f(ev):
            n, c = fn(ev)
            if bk == "torch":
                import torch
                return ({k: torch.as_tensor(v).cuda() for k, v in n.items()},
                        {k: torch.as_tensor(v).cuda() for k, v in c.items()})
            import jax.numpy as jnp
            return ({k: jnp.asarray(v) for k, v in n.items()},
                    {k: jnp.asarray(v) for k, v in c.items()})
        return f

    cfg = _config()
    out = {}
    for name, pf in (("numpy", _plane_fn), (bk, to_dev(_plane_fn))):
        set_backend(name)
        ces, cls, norm = build_corpus(range(3), pf, cfg, tmp_path / name,
                                      dataset_name="cx", write=False)
        out[name] = (ces, norm, cls)
    set_backend("numpy")

    def canon(ce):
        """Sort rows canonically. The two backends emit the SAME rows in a
        different order — numpy loops per band, so it is band-major; jax compacts
        the raveled flat plane, so it is wire-major. Row order within an event is
        not part of the format contract, so equivalence is compared as a set."""
        o = np.lexsort((ce.tau, ce.wire, ce.band, ce.plane_gid))
        return (ce.band[o], ce.plane_gid[o], ce.wire[o], ce.tau[o], ce.value[o])

    for i, (a, b) in enumerate(zip(out["numpy"][0], out[bk][0])):
        assert a.n_coeff == b.n_coeff, f"event {i}: {a.n_coeff} vs {b.n_coeff} coeffs"
        ca, cb = canon(a), canon(b)
        for k, x, y in zip(("band", "plane_gid", "wire", "tau"), ca[:4], cb[:4]):
            np.testing.assert_array_equal(x, y, err_msg=k)
        np.testing.assert_allclose(cb[4], ca[4], rtol=1e-4, atol=1e-4)
        # the sigma field is the one that silently diverged before the fix
        np.testing.assert_allclose(b.sigma_threshold, a.sigma_threshold,
                                   rtol=1e-4, atol=1e-6)
        # ...and the CLEAN target, which shares the noisy support so it sorts the
        # same way. Comparing only the noisy events hid a backend divergence once.
        oa = np.lexsort((a.tau, a.wire, a.band, a.plane_gid))
        ob = np.lexsort((b.tau, b.wire, b.band, b.plane_gid))
        np.testing.assert_allclose(out[bk][2][i].value[ob], out["numpy"][2][i].value[oa],
                                   rtol=1e-4, atol=1e-4, err_msg=f"clean target, event {i}")
    np.testing.assert_allclose(out[bk][1], out["numpy"][1], rtol=1e-4, atol=1e-6)


# ---- 3. physics acceptance (F0) on real data ------------------------------

REAL = ("/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor/"
        "run_0027575715/sim_wire_sensor_0000.h5")


@pytest.mark.skipif(not os.path.exists(REAL), reason="production shard not reachable")
def test_f0_on_a_real_built_shard(tmp_path):
    """Reconstruct a built shard and compare to the clean truth.

    F0 = 1 - sum|recon-clean|/sum|clean| over the true-signal support — one number
    that exercises band/tau mapping, padding, sigma, gate, threshold and codec
    together.

    Thresholds derive from a MATCHED measurement (8 events x 6 planes, one run,
    noise held fixed between arms), not from the achieved value:

        shipped   gate k=3.0 npass=2   mean F0 0.9259   min 0.8772
        research  gate k=4.0 npass=1   mean F0 0.9208   min 0.8627
        legacy    R1 classic multipass mean F0 0.9192   min 0.8712

    The shipped defaults win 47 of 48 event x plane rows, so the gate re-tune
    IMPROVED fidelity. An earlier version of this docstring compared 0.905 to "the
    old pipeline's 0.91-0.96" and set the bar beneath it; that figure came from
    research/wire_denoise/RESULTS.md P10, which used a different remover (R1), a
    different per-plane kappa, a different noise injector and different events —
    it was never a like-for-like number.

    F0 is measured ON the signal support, so it is structurally blind to junk left
    OFF it — and that is exactly where the re-tune costs: a less aggressive gate
    keeps more residual coherent noise (off-support RMS 0.477 vs 0.341 ADC for
    k=4). test_off_support_residual pins that axis so a future re-tune cannot
    trade it away invisibly.
    """
    pytest.importorskip("pimm_data")
    from pimm_data.geometry import load_plane_registry
    from pimm_data.noise import generate_noise, digitize
    from helix.tpc.io import config_from_file, read_sensor_event
    from helix.tpc.pipeline import canonical_plane_gid

    set_backend("numpy")
    b0 = config_from_file(REAL)
    cfg = DetectorConfig(num_time_steps=b0.num_time_steps,
                         plane_labels=b0.plane_labels, pedestals=b0.pedestals)
    reg = load_plane_registry("cubic_wireplane_geometry.json")

    def plane_fn(ev):
        planes = read_sensor_event(REAL, ev, cfg)
        rng = np.random.default_rng(7 + ev)
        noisy, clean = {}, {}
        for lab, img in planes.items():
            g = canonical_plane_gid(lab); nw = img.shape[0]
            wl = np.asarray(reg.get(g, {}).get("wire_lengths", []), np.float64)
            wl = wl if wl.size == nw else np.full(nw, 2.33)
            ped = cfg.pedestals.get(lab.split("_")[-1], 0)
            n = generate_noise(img.shape, rng=rng, wire_lengths_m=wl, incoherent=True,
                               coherent=True, group_size=cfg.group_size)
            noisy[g] = digitize(img + n, ped)
            clean[g] = img.astype(np.float32)
        return noisy, clean

    build_corpus(range(1), plane_fn, cfg, tmp_path, dataset_name="cx")
    ce = read_coeff_event(tmp_path / "cx_coeff_0000.h5", 0)
    recon = ce.reconstruct_images(cfg.num_time_steps)
    _, clean = plane_fn(0)
    f0s, off_rms = [], []
    for gid, r in recon.items():
        c = np.asarray(clean[gid])
        rr = np.asarray(r)[:, :cfg.num_time_steps]
        sig = np.abs(c) > 0
        if sig.any():
            f0s.append(1.0 - np.abs(rr[sig] - c[sig]).sum() / np.abs(c[sig]).sum())
        if (~sig).any():
            # energy left where there is no true signal — F0 cannot see this
            off_rms.append(float(np.sqrt((rr[~sig] ** 2).mean())))
    assert f0s, "no signal support found"
    # matched-measurement floors (worst observed: min 0.877, per-event mean 0.912)
    assert min(f0s) > 0.85, f"F0 per plane too low: {[round(v,3) for v in f0s]}"
    assert np.mean(f0s) > 0.90, f"mean F0 {np.mean(f0s):.3f} below acceptance"
    # F0's blind spot: a less aggressive gate keeps more residual coherent noise
    # while F0 IMPROVES, so fidelity and cleanliness must be pinned separately.
    assert off_rms, "no off-support region found"
    assert np.mean(off_rms) < 0.6, \
        f"off-support RMS {np.mean(off_rms):.3f} ADC too high: {[round(v,3) for v in off_rms]}"
    assert max(off_rms) < 0.85, \
        f"a plane leaks off-support: {[round(v,3) for v in off_rms]}"


# ---- Tier 3: normalization indexing + token sanity -------------------------

def test_norm_sigma_is_row_indexed_not_gid_indexed():
    """norm_sigma rows follow POSITION in gids, not gid value. With a dead plane
    the two disagree, and the naive norm_sigma[gid] silently mis-normalises."""
    from helix.model.tokenize import gid_rows, sigma_for_rows
    gids = np.array([0, 1, 2, 4, 5])                     # plane 3 dead/absent
    ns = np.arange(len(gids) * 4, dtype=np.float32).reshape(len(gids), 4) + 1.0
    pg = np.array([0, 4, 5, 1])
    band = np.array([0, 1, 2, 3])
    rows = gid_rows(pg, gids)
    np.testing.assert_array_equal(rows, [0, 3, 4, 1])    # position, not value
    got = sigma_for_rows(pg, band, gids, ns)
    np.testing.assert_array_equal(got, [ns[0, 0], ns[3, 1], ns[4, 2], ns[1, 3]])
    # The naive norm_sigma[gid] is not merely different — with a dead plane the
    # highest gid indexes past the end of the table.
    with pytest.raises(IndexError):
        _ = ns[pg, band]
    # and where it does NOT overrun, it silently selects the wrong plane's sigma
    pg2, band2 = np.array([4]), np.array([1])
    assert sigma_for_rows(pg2, band2, gids, ns)[0] == ns[3, 1]     # correct: row 3
    assert ns[pg2, band2][0] == ns[4, 1]                            # naive: row 4
    assert ns[3, 1] != ns[4, 1]
    # and a gid absent from the table must raise, never wrap
    with pytest.raises(ValueError, match="absent"):
        gid_rows(np.array([3]), gids)


def test_normalization_roundtrips_and_is_sane():
    from helix.model.tokenize import normalize_values, denormalize_values
    rng = np.random.default_rng(0)
    gids = np.array([0, 1, 2])
    ns = np.array([[3.0, 2.0], [3.5, 2.5], [4.0, 3.0]], np.float32)
    n = 5000
    pg = rng.choice(gids, n); band = rng.integers(0, 2, n)
    sig = ns[np.searchsorted(gids, pg), band]
    val = (rng.standard_normal(n) * sig).astype(np.float32)   # ~1 sigma coefficients
    tok = normalize_values(val, pg, band, gids, ns)
    np.testing.assert_allclose(denormalize_values(tok, pg, band, gids, ns), val,
                               rtol=1e-4, atol=1e-4)
    # arcsinh(v/sigma) on ~1-sigma data must be O(1) and roughly symmetric —
    # a wrong sigma shows up here as a grossly mis-scaled distribution.
    assert 0.3 < np.std(tok) < 3.0, f"token std {np.std(tok):.3f} out of range"
    assert abs(np.mean(tok)) < 0.2, f"token mean {np.mean(tok):.3f} not centred"
    assert np.abs(tok).max() < 12.0


def test_noise_seed_and_external_inputs_are_recorded(tmp_path):
    """The corpus stores ONE fixed noise realisation per event by design, so the
    seed is the only thing that makes that realisation reproducible — and it is
    not derivable from the shard, because the two build modes use different seed
    formulas. Likewise basis_digest covers the wavelet/gate/threshold but says
    nothing about the plane geometry or noise spectrum the DSP consumed, so a
    change to either would make old and new shards silently incomparable.
    """
    import json as _json
    set_backend("numpy")
    prov = dict(geom_sha256="a" * 64, spectrum_sha256="b" * 64,
                seed_formula="test:fixed")
    seeds = [111, 222, 333, 444]
    build_corpus([(f"src_{e:04d}.h5", e, seeds[e]) for e in range(4)],
                 _plane_fn, _config(), tmp_path, dataset_name="cx",
                 noise=dict(kind="colored"), provenance=prov)

    for name in ("cx_coeff_0000.h5", "cx_coeff_clean_0000.h5"):
        with h5py.File(tmp_path / name, "r") as f:
            got = _json.loads(f["config"].attrs["provenance_json"])
            # The caller's fields must survive verbatim...
            assert {k: got.get(k) for k in prov} == prov, \
                f"{name}: caller provenance not round-tripped"
            # ...and the writer stamps exactly ONE field of its own: which helix
            # ran. Callers cannot be relied on to record it — none did, and a
            # corpus was duly built by two different working trees with no field
            # able to show it — so write_coeff_shard adds it rather than
            # accepting it from the caller.
            assert set(got) - set(prov) == {"code"}, \
                f"{name}: unexpected writer-added provenance {set(got) - set(prov)}"
            assert "version" in got["code"], f"{name}: code block lacks a version"
            np.testing.assert_array_equal(f["ident"]["noise_seed"][:], seeds)
            # the identity tuple's source_file must reach /ident too, not a
            # fabricated f"{dataset}_sensor_{file_index:04d}.h5"
            assert [s.decode() for s in f["ident"]["source_file"][:]] == \
                [f"src_{e:04d}.h5" for e in range(4)]

    # a build that supplies no seeds must not invent one
    plain = tmp_path / "plain"
    build_corpus(range(2), _plane_fn, _config(), plain, dataset_name="cx")
    with h5py.File(plain / "cx_coeff_0000.h5", "r") as f:
        assert "noise_seed" not in f["ident"]
        # No caller provenance -> only the writer's own code stamp. It is
        # unconditional on purpose: a shard that cannot say which helix built it
        # is exactly the gap this closes, so there is no opt-out.
        assert set(_json.loads(f["config"].attrs["provenance_json"])) == {"code"}
