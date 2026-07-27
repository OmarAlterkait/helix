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

import os

import numpy as np
import pytest
import h5py

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
    probs = audit_shard(p, strict=False)
    assert any(expect in s for s in probs), f"audit missed {corrupt}: {probs}"
    with pytest.raises(ValueError, match="shard audit failed"):
        audit_shard(p)


# ---- 2. backend equivalence (same noisy input, numpy vs jax) --------------

@pytest.mark.skipif(not pytest.importorskip("jax", reason="jax not installed"),
                    reason="jax not installed")
def test_numpy_jax_equivalent_on_same_input(tmp_path):
    """Noise RNG differs per backend, so noise is fixed and only the DSP +
    extraction + assembly are compared. Measured on real planes: support
    99.9996-100%, values and sigma at float32 epsilon."""
    cfg = _config()
    out = {}
    for bk in ("numpy", "jax"):
        set_backend(bk)
        ces, _, norm = build_corpus(range(3), _plane_fn, cfg, tmp_path / bk,
                                    dataset_name="cx", write=False)
        out[bk] = (ces, norm)
    set_backend("numpy")

    def canon(ce):
        """Sort rows canonically. The two backends emit the SAME rows in a
        different order — numpy loops per band, so it is band-major; jax compacts
        the raveled flat plane, so it is wire-major. Row order within an event is
        not part of the format contract, so equivalence is compared as a set."""
        o = np.lexsort((ce.tau, ce.wire, ce.band, ce.plane_gid))
        return (ce.band[o], ce.plane_gid[o], ce.wire[o], ce.tau[o], ce.value[o])

    for a, b in zip(out["numpy"][0], out["jax"][0]):
        assert a.n_coeff == b.n_coeff
        ca, cb = canon(a), canon(b)
        for k, x, y in zip(("band", "plane_gid", "wire", "tau"), ca[:4], cb[:4]):
            np.testing.assert_array_equal(x, y, err_msg=k)
        np.testing.assert_allclose(cb[4], ca[4], rtol=1e-4, atol=1e-4)
        # the sigma field is the one that silently diverged before the fix
        np.testing.assert_allclose(b.sigma_threshold, a.sigma_threshold,
                                   rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(out["jax"][1], out["numpy"][1], rtol=1e-4, atol=1e-6)


# ---- 3. physics acceptance (F0) on real data ------------------------------

REAL = ("/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor/"
        "run_0027575715/sim_wire_sensor_0000.h5")


@pytest.mark.skipif(not os.path.exists(REAL), reason="production shard not reachable")
def test_f0_on_a_real_built_shard(tmp_path):
    """Reconstruct a built shard and compare to the clean truth.

    F0 = 1 - sum|recon-clean|/sum|clean| over the true-signal support. Measured
    0.905 overall (Y planes 0.948, induction 0.88) against the old pipeline's
    0.91-0.96 — one number that exercises band/tau mapping, padding, sigma, gate,
    threshold and codec together.
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
    f0s = []
    for gid, r in recon.items():
        c = np.asarray(clean[gid])
        sig = np.abs(c) > 0
        if sig.any():
            f0s.append(1.0 - np.abs(np.asarray(r)[:, :cfg.num_time_steps][sig] - c[sig]).sum()
                       / np.abs(c[sig]).sum())
    assert f0s, "no signal support found"
    assert min(f0s) > 0.80, f"F0 per plane too low: {[round(v,3) for v in f0s]}"
    assert np.mean(f0s) > 0.85, f"mean F0 {np.mean(f0s):.3f} below acceptance"
