"""helix.build: building shards through the chain (the corpus, its acceptance checks on a built
shard) and the named sources."""
import pytest

torch = pytest.importorskip("torch")

import os

import numpy as np
import h5py

from helix.tpc.chain import TPCChain, sim_config
from helix.tpc import chain as tpc_chain
from helix.build import normalization_table
from _fixtures import build_corpus  # noqa: E402
from helix.core.coeff_io import read_coeff_event, read_coeff_shard
from helix.core.coeff_io import audit_shard, read_coeff_event
from _fixtures import sim_file                                     # noqa: E402
from helix.build import SOURCES, source


# ---- building a corpus (helix.build) -----------------------------------------------------

# Corpus builder on the TPC chain — keeps the old star_tpc semantics:
#
# - clean target is co-supported (clean coeffs at the NOISY support), as
#   ``val_clean = clean[gid][b][noisy_mask]``; the clean transform is the chain's
#   (``TPCChain.dwt_rows``: no gate, no threshold, the dropped finest band zeroed).
# - norm_sigma = mean over the shard's events of the per-event threshold σ (old sigma_tab).
# - padding to a 2**level multiple (old ``pad(g, (0,(-nt)%4))``) — tested on a
#   non-power-of-2 length so pad>0.
# - planes are grouped by TPC (gid // 3) and each TPC goes through the chain once.

NW = [8, 6, 5]                             # U, V, Y wires of each volume
N_VOL = 2
GIDS = list(range(3 * N_VOL))              # gid = volume * 3 + plane
NT = 102                                   # not a multiple of 4 → pad to 104


def _chain(**overrides):
    o = {"dwt": {"wavelet": "db2", "level": 2}}
    o.update(overrides)
    return TPCChain(sim_config([NW] * N_VOL, o))


def _plane_fn(ev):
    rng = np.random.default_rng(1000 + ev)
    clean, noisy = {}, {}
    for gid in GIDS:
        w = NW[gid % 3]
        c = np.zeros((w, NT), np.float32)
        for _ in range(3):
            c[rng.integers(w), rng.integers(NT)] = rng.uniform(20, 40)
        clean[gid] = c
        nb = (w + 64 - 1) // 64
        coh = (rng.standard_normal((nb, NT)).astype(np.float32) * 2.0)[
            np.minimum(np.arange(w) // 64, nb - 1)]
        incoh = rng.standard_normal((w, NT)).astype(np.float32) * 0.5
        noisy[gid] = (c + coh + incoh).astype(np.float32)
    return noisy, clean


def test_build_corpus_matches_old_semantics(tmp_path):
    ch = _chain()
    noisy, clean, norm = build_corpus(range(4), _plane_fn, ch, tmp_path, dataset_name="cx", run="run_000")
    assert len(noisy) == 4 and len(clean) == 4

    # (1) padding: basis validates; padded length = 104, raw 102
    for ce in noisy:
        ce.basis.validate()
        assert ce.basis.n_ticks_raw == NT and ce.basis.padded_length == 104
        assert list(ce.gids) == GIDS and list(ce.n_wires) == NW * N_VOL
        assert (ce.band < ce.basis.n_bands - 1).all()        # D1 dropped (drop_finest 1)

    # (2) clean is CO-SUPPORTED (identical coords) with clean values at those coords
    for ce_n, ce_c in zip(noisy, clean):
        for k in ("band", "plane_gid", "wire", "tau"):
            np.testing.assert_array_equal(getattr(ce_c, k), getattr(ce_n, k))
        np.testing.assert_array_equal(ce_c.sigma_threshold, ce_n.sigma_threshold)
    # spot-check clean values == the chain's clean transform at the noisy coords, on the target's
    # value step
    ce_n, ce_c = noisy[0], clean[0]
    _, cl = _plane_fn(0)
    clean_bands = {g: ch.dwt_rows(cl[g]) for g in GIDS}
    exact = np.array([clean_bands[int(g)][int(b)][int(w), int(t)]
                      for g, b, w, t in zip(ce_n.plane_gid, ce_n.band, ce_n.wire, ce_n.tau)], np.float32)
    step = tpc_chain.value_step(ce_n.sigma_threshold, ch.cfg["store"]["step_target"])
    np.testing.assert_array_equal(ce_c.value, tpc_chain.quantize(exact, ce_n.band, ce_n.plane_gid, ce_n.gids, step))

    # (3) norm_sigma = mean of the events' sigma_threshold
    np.testing.assert_allclose(norm, normalization_table(noisy))
    np.testing.assert_allclose(norm, np.stack([e.sigma_threshold for e in noisy]).mean(0), rtol=1e-6)

    # (4) shards written + round-trip; /config carries norm_sigma
    for i, ce in enumerate(read_coeff_shard(tmp_path / "cx_coeff_0000.h5")):
        for k in ("band", "wire", "tau", "value"):
            np.testing.assert_array_equal(getattr(ce, k), getattr(noisy[i], k))
    with h5py.File(tmp_path / "cx_coeff_0000.h5", "r") as f:
        np.testing.assert_allclose(f["config"]["norm_sigma"][:], norm)
    # coeff_clean is a VALUES-ONLY shard: co-supported, so it stores no coords and
    # inherits the noisy shard's through the coord_digest-checked join.
    p_noisy = tmp_path / "cx_coeff_0000.h5"
    p_clean = tmp_path / "cx_coeff_clean_0000.h5"
    with h5py.File(p_clean, "r") as f:
        assert not bool(f["config"].attrs["has_coords"])
        assert not any(k in f["coord"] for k in ("band", "plane_gid", "wire", "tau"))
    cc = read_coeff_event(p_clean, 0, coords_from=p_noisy)
    np.testing.assert_array_equal(cc.band, noisy[0].band)
    np.testing.assert_array_equal(cc.value, clean[0].value)
    # reading it standalone must fail loudly, not yield coord-less garbage
    with pytest.raises(ValueError, match="values-only"):
        read_coeff_event(p_clean, 0)


def test_build_corpus_is_the_chain_per_tpc(tmp_path):
    """The builder adds nothing to the chain: per TPC, ``process_images`` on the planes in
    order, then ``coeff_event`` over every TPC's results."""
    ch = _chain()
    (ce,), _, _ = build_corpus([("f.h5", 7)], _plane_fn, ch, tmp_path, run="r", with_clean=False)
    noisy, _ = _plane_fn(7)
    res = [r for t in range(N_VOL) for r in ch.process_images([noisy[3 * t + p] for p in range(3)], t)]
    ref = tpc_chain.coeff_event(ch, res, run="r", source_file="f.h5", event=7)
    for k in ("band", "plane_gid", "wire", "tau", "value", "gids", "n_wires", "sigma_threshold"):
        np.testing.assert_array_equal(getattr(ce, k), getattr(ref, k), err_msg=k)
    assert ce.basis == ref.basis and (ce.run, ce.source_file, ce.event) == ("r", "f.h5", 7)


def test_build_corpus_no_clean(tmp_path):
    noisy, clean, norm = build_corpus(range(2), _plane_fn, _chain(), tmp_path, dataset_name="cx", with_clean=False)
    assert clean == []
    assert not (tmp_path / "cx_coeff_clean_0000.h5").exists()
    assert (tmp_path / "cx_coeff_0000.h5").exists()


# ---- regression: audit fixes ----------------------------------------------

def test_one_event_shard_has_a_table(tmp_path):
    n, c, norm = build_corpus(range(1), _plane_fn, _chain(), tmp_path, dataset_name="cx")
    assert norm.shape[0] == len(GIDS) and norm.max() > 0


def test_global_norm_sigma_is_frozen_across_shards(tmp_path):
    """A supplied norm_sigma must be written verbatim to EVERY shard, so the same
    coefficient normalises identically no matter which shard it landed in."""
    ch = _chain()
    # shard A derives its own table; shard B is built from different events but
    # must reuse A's table when it is passed in.
    a, _, norm_a = build_corpus(range(3), _plane_fn, ch, tmp_path / "a", dataset_name="cx")
    b, _, norm_b = build_corpus(range(3, 6), _plane_fn, ch, tmp_path / "b",
                                dataset_name="cx", norm_sigma=norm_a)
    np.testing.assert_array_equal(norm_b, norm_a)
    with h5py.File(tmp_path / "b" / "cx_coeff_0000.h5") as f:
        np.testing.assert_allclose(f["config"]["norm_sigma"][:], norm_a)
    # a self-derived table for different events would NOT have matched
    _, _, norm_own = build_corpus(range(3, 6), _plane_fn, ch, tmp_path / "c",
                                  dataset_name="cx")
    assert not np.array_equal(norm_own, norm_a)
    with pytest.raises(ValueError, match="norm_sigma: shape"):
        build_corpus(range(2), _plane_fn, ch, tmp_path / "d", dataset_name="cx",
                     norm_sigma=np.ones((99, 3), np.float32))


def test_clean_geometry_mismatch_raises():
    ch = _chain()
    noisy, clean = _plane_fn(0)
    ce = tpc_chain.coeff_event(ch, ch.process_planes(noisy))
    bad_clean = {g: v[: v.shape[0] // 2] for g, v in clean.items()}    # half the wires
    with pytest.raises(ValueError, match="wires"):
        tpc_chain.clean_coeff_event(ch, ce, bad_clean)
    missing = {g: clean[g] for g in GIDS[1:]}                           # drop a gid
    with pytest.raises(ValueError, match="missing gid"):
        tpc_chain.clean_coeff_event(ch, ce, missing)


def test_noisy_planes_must_fill_each_tpc():
    """The chain processes a TPC's planes together: a TPC with a plane missing, a plane
    with the wrong wire count, or a gid past the readout must raise, not misalign."""
    ch = _chain()
    noisy, _ = _plane_fn(0)
    with pytest.raises(ValueError, match="the chain needs all 3"):
        ch.process_planes({g: v for g, v in noisy.items() if g != 4})
    with pytest.raises(ValueError, match="geometry must match"):
        ch.process_planes({**noisy, 1: noisy[1][:-1]})
    with pytest.raises(ValueError, match="tpc 2"):
        ch.process_planes({**noisy, 6: noisy[0], 7: noisy[1], 8: noisy[2]})
    # a whole TPC absent is fine: its gids are simply not in the event
    ce = tpc_chain.coeff_event(ch, ch.process_planes({g: noisy[g] for g in (3, 4, 5)}))
    assert list(ce.gids) == [3, 4, 5]


def test_sigma_threshold_and_norm_are_populated(tmp_path):
    """A shard whose sigma_threshold is zero has a meaningless norm_sigma — the
    tokenizer's normalization table. This regression guards the path where a
    flat-bands branch skipped the sigma assignment entirely. The dropped finest
    band's column is 0 by construction (no coefficients there); every kept band
    must be populated.
    """
    noisy, clean, norm = build_corpus(range(3), _plane_fn, _chain(), tmp_path, dataset_name="cx")
    for ce in noisy:
        assert ce.sigma_threshold.shape[0] == len(GIDS)
        assert np.isfinite(ce.sigma_threshold).all()
        assert (ce.sigma_threshold[:, :-1] > 0).all(), "a kept band's sigma_threshold is zero"
        assert (ce.sigma_threshold[:, -1] == 0).all(), "D1 is dropped: its sigma is 0"
    assert norm.max() > 0, "norm_sigma is all zero"
    with h5py.File(tmp_path / "cx_coeff_0000.h5", "r") as f:
        assert f["config"]["norm_sigma"][:].max() > 0
        assert f["coord"]["sigma_threshold"][:].max() > 0


def test_clean_target_is_written_only_when_asked(tmp_path):
    """A source with noise-free planes writes no clean shard by default (nothing trains on it)."""
    from _fixtures import PlanesSource
    from helix.build import build
    src = PlanesSource(range(2), _plane_fn)
    build(src, str(tmp_path / "cx"), indices=[0, 1], chain=_chain(), events_per_shard=2)
    assert (tmp_path / "cx_coeff_0000.h5").exists()
    assert not (tmp_path / "cx_coeff_clean_0000.h5").exists()


# ---- acceptance checks on a built corpus -------------------------------------------------

# Acceptance checks for a BUILT corpus — semantic, not just structural.
#
# The lesson these encode: a shard whose ``sigma_threshold`` (and hence
# ``norm_sigma``) was entirely zero passed round-trip, identity, digest and
# co-support checks. Nothing asserted that a value must be *populated*, must
# *vary*, or must *reconstruct the physics*. These do.
#
#   1. degenerate-field audit  — catches all-zero / constant / out-of-range content
#   2. physics acceptance (F0) — reconstruct a built shard vs the clean truth projected
#                                onto the bands the chain keeps

NW = [8, 6, 5]                             # U, V, Y wires of each volume
N_VOL = 2
GIDS = list(range(3 * N_VOL))
NT = 102


def _chain(device=None):
    """The simulation chain for this toy layout (db2 L2 so 102 ticks pad to 104)."""
    return TPCChain(sim_config([NW] * N_VOL, {"dwt": {"wavelet": "db2", "level": 2}}), device=device)


def _plane_fn(ev):
    rng = np.random.default_rng(1000 + ev)
    clean, noisy = {}, {}
    for gid in GIDS:
        w = NW[gid % 3]
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
    build_corpus(range(4), _plane_fn, _chain(), tmp_path, dataset_name="cx")
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
    build_corpus(range(4), _plane_fn, _chain(), tmp_path, dataset_name="cx")
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
            f["config/gids"][...] = np.array([1, 0, 2, 3, 4, 5], np.int32)
        elif corrupt == "gids_duplicate":
            f["config/gids"][...] = np.array([0, 0, 2, 3, 4, 5], np.int32)
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
    build_corpus(range(4), _plane_fn, _chain(), tmp_path, dataset_name="cx")
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
    build_corpus(range(4, 8), _plane_fn, _chain(), other, dataset_name="cx")
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
    build_corpus(range(2), _plane_fn, _chain(), tmp_path, dataset_name="cx",
                 noise=dict(kind="white", coherent=True, incoherent=True))
    with h5py.File(tmp_path / "cx_coeff_0000.h5", "r") as f:
        assert _json.loads(f["config"].attrs["noise_json"])["kind"] == "white"
    with h5py.File(tmp_path / "cx_coeff_clean_0000.h5", "r") as f:
        assert _json.loads(f["config"].attrs["noise_json"])["kind"] == "white"


# ---- 2. physics acceptance (F0) on real data ------------------------------


REAL = sim_file("sensor")


@pytest.mark.skipif(not os.path.exists(REAL), reason="campaign task not reachable")
def test_f0_on_a_real_built_shard(tmp_path):
    """Reconstruct a built shard and compare to the clean truth.

    F0 = 1 - sum|recon-truth|/sum|truth| over the true-signal support (|clean| > 0) —
    one number that exercises band/tau mapping, padding, sigma, gate, threshold and
    codec together. ``truth`` is the noise-free image PROJECTED onto the bands the
    chain keeps (its own transform with D1 zeroed, reconstructed): D1 is dropped by
    design (the FM tokenizes A4, D4, D3, D2), so the dropped band is not a fidelity
    loss to charge the DSP with. The projection alone keeps 0.989-0.995 of the clean
    signal on this event.

    Floors from the chain on wire_test_00_00_02 run_0027575715 shard 0 (events 0-3 x 6 planes,
    this noise); they hold on the campaign task j000085-000089, which this now reads:

        per plane   F0 min 0.871        per-event mean F0 0.910-0.917
        off-support RMS(recon - truth) per-event mean 0.43-0.52, max 0.692 ADC

    The earlier simulation path (since retired: gate, D1 kept) on the same event-0 input gave
    the same F0 to 0.001 per plane, against the raw clean image. The floors below are
    the ones that path used.

    F0 is measured ON the signal support, so it is structurally blind to junk left
    OFF it — a less aggressive gate keeps more residual coherent noise while F0
    improves. The off-support RMS pins that axis so a future re-tune cannot trade it
    away invisibly.
    """
    from helix.tpc.readout import load_plane_registry
    from _noise_reference import generate_noise, digitize
    from helix.tpc.io import read_event, sensor_layout

    layout = sensor_layout(REAL)
    chain = TPCChain(sim_config(layout.num_wires))
    group_size = int(chain.cfg["readout"]["group_size"])
    reg = load_plane_registry("cubic_wireplane_geometry.json")
    nt = layout.num_time_steps
    made = {}

    def plane_fn(ev):
        planes = read_event(REAL, ev, layout)
        rng = np.random.default_rng(7 + ev)
        noisy, clean = {}, {}
        for lab, img in planes.items():
            g = layout.plane_gid(lab); nw = img.shape[0]
            wl = np.asarray(reg.get(g, {}).get("wire_lengths", []), np.float64)
            wl = wl if wl.size == nw else np.full(nw, 2.33)
            n = generate_noise(img.shape, rng=rng, wire_lengths_m=wl, incoherent=True,
                               coherent=True, group_size=group_size)
            noisy[g] = digitize(img + n, layout.pedestal(lab))
            clean[g] = img.astype(np.float32)
        made[ev] = clean
        return noisy, clean

    build_corpus(range(1), plane_fn, chain, tmp_path, dataset_name="cx")
    ce = read_coeff_event(tmp_path / "cx_coeff_0000.h5", 0)
    recon = ce.reconstruct_images(nt)
    d = chain.cfg["dwt"]
    f0s, off_rms = [], []
    for gid, r in recon.items():
        c = np.asarray(made[0][gid])
        truth = chain._rec(chain.dwt_rows(c), d["wavelet"], d["level"], nt).cpu().numpy()[:, :nt]
        rr = np.asarray(r)[:, :nt]
        sig = np.abs(c) > 0
        if sig.any():
            f0s.append(1.0 - np.abs(rr[sig] - truth[sig]).sum() / np.abs(truth[sig]).sum())
        if (~sig).any():
            # energy left where there is no true signal — F0 cannot see this
            off_rms.append(float(np.sqrt(((rr - truth)[~sig] ** 2).mean())))
    assert f0s, "no signal support found"
    assert min(f0s) > 0.85, f"F0 per plane too low: {[round(v,3) for v in f0s]}"
    assert np.mean(f0s) > 0.90, f"mean F0 {np.mean(f0s):.3f} below acceptance"
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
    prov = dict(geom_sha256="a" * 64, spectrum_sha256="b" * 64,
                seed_formula="test:fixed")
    seeds = [111, 222, 333, 444]
    build_corpus([(f"src_{e:04d}.h5", e, seeds[e]) for e in range(4)],
                 _plane_fn, _chain(), tmp_path, dataset_name="cx",
                 noise=dict(kind="colored"), provenance=prov)

    for name in ("cx_coeff_0000.h5", "cx_coeff_clean_0000.h5"):
        with h5py.File(tmp_path / name, "r") as f:
            got = _json.loads(f["config"].attrs["provenance_json"])
            # The caller's fields must survive verbatim...
            assert {k: got.get(k) for k in prov} == prov, \
                f"{name}: caller provenance not round-tripped"
            # ...and the build adds two of its own: which helix ran (the writer) and
            # on which device (the DSP is architecture-sensitive). Callers cannot be relied on to record it — none did, and a
            # corpus was duly built by two different working trees with no field
            # able to show it — so write_coeff_shard adds it rather than
            # accepting it from the caller.
            assert set(got) - set(prov) == {"code", "device"}, \
                f"{name}: unexpected writer-added provenance {set(got) - set(prov)}"
            assert "version" in got["code"], f"{name}: code block lacks a version"
            np.testing.assert_array_equal(f["ident"]["noise_seed"][:], seeds)
            # the identity tuple's source_file must reach /ident too, not a
            # fabricated f"{dataset}_sensor_{file_index:04d}.h5"
            assert [s.decode() for s in f["ident"]["source_file"][:]] == \
                [f"src_{e:04d}.h5" for e in range(4)]

    # a build that supplies no seeds must not invent one
    plain = tmp_path / "plain"
    build_corpus(range(2), _plane_fn, _chain(), plain, dataset_name="cx")
    with h5py.File(plain / "cx_coeff_0000.h5", "r") as f:
        assert "noise_seed" not in f["ident"]
        # No caller provenance -> only the code and device stamps. They are
        # unconditional on purpose: a shard that cannot say which helix built it
        # is exactly the gap this closes, so there is no opt-out.
        assert set(_json.loads(f["config"].attrs["provenance_json"])) == {"code", "device"}


# ---- the sources -------------------------------------------------------------------------

# Every named source has what helix.build needs (``python -m helix.build SOURCE``).

@pytest.mark.parametrize("name", sorted(SOURCES))
def test_every_source_has_what_build_needs(name):
    """What :func:`helix.build.main` and :func:`helix.build.plan` read of a source class."""
    Source = source(name)
    for attr in ("systems", "dataset", "pattern", "events", "shard_attrs", "__len__"):
        assert hasattr(Source, attr), f"{name}: {Source.__name__} has no {attr}"
    assert hasattr(Source, "chain_config") or hasattr(Source, "optical_chain")
    assert not hasattr(Source, "chain_config") or hasattr(Source, "bind")


def test_an_unknown_source_is_refused():
    with pytest.raises(SystemExit, match="unknown source"):
        source("nope")
