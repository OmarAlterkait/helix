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
from helix.core.coeff_io import audit_shard
from _fixtures import sim_file                                     # noqa: E402
from helix.build import SOURCES, source


# ---- building a corpus (helix.build) -----------------------------------------------------

# Corpus builder on the TPC chain — keeps the old star_tpc semantics:
#
# - norm_sigma = mean over the shard's events of the per-event threshold σ (old sigma_tab).
# - padding to a 2**level multiple (old ``pad(g, (0,(-nt)%4))``) — tested on a
#   non-power-of-2 length so pad>0.
# - planes are grouped by TPC (gid // 3) and each TPC goes through the chain once.

NW = [8, 6, 5]                             # U, V, Y wires of each volume
N_VOL = 2
GIDS = list(range(3 * N_VOL))              # gid = volume * 3 + plane
NT = 102                                   # not a multiple of 4 → pad to 104


def _chain(device=None, **overrides):
    """The simulation chain for this toy layout (db2 L2, so 102 ticks pad to 104)."""
    o = {"dwt": {"wavelet": "db2", "level": 2}}
    o.update(overrides)
    return TPCChain(sim_config([NW] * N_VOL, o), device=device)


def _plane_fn(ev):
    rng = np.random.default_rng(1000 + ev)
    noisy = {}
    for gid in GIDS:
        w = NW[gid % 3]
        c = np.zeros((w, NT), np.float32)
        for _ in range(3):
            c[rng.integers(w), rng.integers(NT)] = rng.uniform(20, 40)
        nb = (w + 64 - 1) // 64
        coh = (rng.standard_normal((nb, NT)).astype(np.float32) * 2.0)[
            np.minimum(np.arange(w) // 64, nb - 1)]
        incoh = rng.standard_normal((w, NT)).astype(np.float32) * 0.5
        noisy[gid] = (c + coh + incoh).astype(np.float32)
    return noisy


def test_build_corpus_matches_old_semantics(tmp_path):
    ch = _chain()
    noisy, norm = build_corpus(range(4), _plane_fn, ch, tmp_path, dataset_name="cx", run="run_000")
    assert len(noisy) == 4

    # (1) padding: basis validates; padded length = 104, raw 102
    for ce in noisy:
        ce.basis.validate()
        assert ce.basis.n_ticks_raw == NT and ce.basis.padded_length == 104
        assert list(ce.gids) == GIDS and list(ce.n_wires) == NW * N_VOL
        assert (ce.band < ce.basis.n_bands - 1).all()        # D1 dropped (drop_finest 1)

    # (3) norm_sigma = mean of the events' sigma_threshold
    np.testing.assert_allclose(norm, normalization_table(noisy))
    np.testing.assert_allclose(norm, np.stack([e.sigma_threshold for e in noisy]).mean(0), rtol=1e-6)

    # (4) shards written + round-trip; /config carries norm_sigma
    for i, ce in enumerate(read_coeff_shard(tmp_path / "cx_coeff_0000.h5")):
        for k in ("band", "wire", "tau", "value"):
            np.testing.assert_array_equal(getattr(ce, k), getattr(noisy[i], k))
    with h5py.File(tmp_path / "cx_coeff_0000.h5", "r") as f:
        np.testing.assert_allclose(f["config"]["norm_sigma"][:], norm)


def test_build_corpus_is_the_chain_per_tpc(tmp_path):
    """The builder adds nothing to the chain: per TPC, ``process_images`` on the planes in
    order, then ``coeff_event`` over every TPC's results."""
    ch = _chain()
    (ce,), _ = build_corpus([("f.h5", 7)], _plane_fn, ch, tmp_path, run="r")
    noisy = _plane_fn(7)
    res = [r for t in range(N_VOL) for r in ch.process_images([noisy[3 * t + p] for p in range(3)], t)]
    ref = tpc_chain.coeff_event(ch, res, run="r", source_file="f.h5", event=7)
    for k in ("band", "plane_gid", "wire", "tau", "value", "gids", "n_wires", "sigma_threshold"):
        np.testing.assert_array_equal(getattr(ce, k), getattr(ref, k), err_msg=k)
    assert ce.basis == ref.basis and (ce.run, ce.source_file, ce.event) == ("r", "f.h5", 7)


# ---- regression: audit fixes ----------------------------------------------

def test_one_event_shard_has_a_table(tmp_path):
    n, norm = build_corpus(range(1), _plane_fn, _chain(), tmp_path, dataset_name="cx")
    assert norm.shape[0] == len(GIDS) and norm.max() > 0


def test_global_norm_sigma_is_frozen_across_shards(tmp_path):
    """A supplied norm_sigma must be written verbatim to EVERY shard, so the same
    coefficient normalises identically no matter which shard it landed in."""
    ch = _chain()
    # shard A derives its own table; shard B is built from different events but
    # must reuse A's table when it is passed in.
    a, norm_a = build_corpus(range(3), _plane_fn, ch, tmp_path / "a", dataset_name="cx")
    b, norm_b = build_corpus(range(3, 6), _plane_fn, ch, tmp_path / "b",
                                dataset_name="cx", norm_sigma=norm_a)
    np.testing.assert_array_equal(norm_b, norm_a)
    with h5py.File(tmp_path / "b" / "cx_coeff_0000.h5") as f:
        np.testing.assert_allclose(f["config"]["norm_sigma"][:], norm_a)
    # a self-derived table for different events would NOT have matched
    _, norm_own = build_corpus(range(3, 6), _plane_fn, ch, tmp_path / "c",
                                  dataset_name="cx")
    assert not np.array_equal(norm_own, norm_a)
    with pytest.raises(ValueError, match="norm_sigma: shape"):
        build_corpus(range(2), _plane_fn, ch, tmp_path / "d", dataset_name="cx",
                     norm_sigma=np.ones((99, 3), np.float32))


def test_noisy_planes_must_fill_each_tpc():
    """The chain processes a TPC's planes together: a TPC with a plane missing, a plane
    with the wrong wire count, or a gid past the readout must raise, not misalign."""
    ch = _chain()
    noisy = _plane_fn(0)
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
    noisy, norm = build_corpus(range(3), _plane_fn, _chain(), tmp_path, dataset_name="cx")
    for ce in noisy:
        assert ce.sigma_threshold.shape[0] == len(GIDS)
        assert np.isfinite(ce.sigma_threshold).all()
        assert (ce.sigma_threshold[:, :-1] > 0).all(), "a kept band's sigma_threshold is zero"
        assert (ce.sigma_threshold[:, -1] == 0).all(), "D1 is dropped: its sigma is 0"
    assert norm.max() > 0, "norm_sigma is all zero"
    with h5py.File(tmp_path / "cx_coeff_0000.h5", "r") as f:
        assert f["config"]["norm_sigma"][:].max() > 0
        assert f["coord"]["sigma_threshold"][:].max() > 0


# ---- acceptance checks on a built corpus -------------------------------------------------

# Semantic, not just structural: a shard whose sigma_threshold (and so norm_sigma) was all zero
# passed round-trip, identity and digest checks. The audit asks that values be populated, vary and
# lie in range; F0 that a built shard reconstructs the physics.

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


REAL = sim_file("sensor")


@pytest.mark.skipif(not os.path.exists(REAL), reason="campaign task not reachable")
def test_f0_on_a_real_built_shard(tmp_path):
    """F0 = 1 - sum|recon - truth| / sum|truth| over the true-signal support: band/tau mapping,
    padding, sigma, gate, threshold and codec in one number. ``truth`` is the noise-free image
    projected onto the bands the chain keeps (D1 is dropped by design). F0 is blind to junk left
    OFF the support, so the off-support RMS pins that axis. Measured on wire_test run_0027575715
    and on this task: per-plane F0 >= 0.871, per-event mean 0.910-0.917; off-support RMS per-event
    mean 0.43-0.52, max 0.692 ADC."""
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
        return noisy

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
    """What makes a shard reproducible and comparable, and is not derivable from it: the per-event
    noise seed (one fixed realisation per event), the noise model, the DSP's external inputs
    (geometry, spectrum: basis_digest covers neither), and -- added by the writer, never trusted
    to callers -- which helix ran and on which device. A build with no seeds invents none."""
    import json as _json
    prov = dict(geom_sha256="a" * 64, spectrum_sha256="b" * 64, seed_formula="test:fixed")
    seeds = [111, 222, 333, 444]
    build_corpus([(f"src_{e:04d}.h5", e, seeds[e]) for e in range(4)], _plane_fn, _chain(), tmp_path,
                 dataset_name="cx", noise=dict(kind="colored"), provenance=prov)
    with h5py.File(tmp_path / "cx_coeff_0000.h5", "r") as f:
        got = _json.loads(f["config"].attrs["provenance_json"])
        assert {k: got.get(k) for k in prov} == prov and set(got) - set(prov) == {"code", "device"}
        assert "version" in got["code"]
        assert _json.loads(f["config"].attrs["noise_json"])["kind"] == "colored"
        np.testing.assert_array_equal(f["ident"]["noise_seed"][:], seeds)
        assert [s.decode() for s in f["ident"]["source_file"][:]] == [f"src_{e:04d}.h5" for e in range(4)]
    plain = tmp_path / "plain"
    build_corpus(range(2), _plane_fn, _chain(), plain, dataset_name="cx")
    with h5py.File(plain / "cx_coeff_0000.h5", "r") as f:
        assert "noise_seed" not in f["ident"]
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


def test_no_script_names_a_retired_entry_point():
    """The builders are ``python -m helix.build SOURCE`` (they replaced build_coeff_corpus.py,
    helix.tpc.sbnd_run, helix.optical.goop_run and then scripts/build_{sim,sbnd,goop}.py, with the
    --backend / --mode flags), the verifier is ``helix.data.verify``, the bin grid ``python -m
    helix.data.bins`` and training ``scripts/submit.sh``: a shipped script or study still naming
    a retired one fails at its first task."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    retired = ("build_coeff_corpus.py", "helix.tpc.sbnd_run", "helix.optical.goop_run", "--backend",
               "--mode serial", "scripts/build_sim.py", "scripts/build_sbnd.py", "scripts/build_goop.py",
               "scripts/train.sh", "helix.data.coeff_verify", "derive_coeff_bins.py")
    files = [*(root / "scripts").glob("*.sh"), *(root / "scripts").glob("*.py"), *(root / "configs").rglob("*.yaml")]
    for p in sorted(files):
        code = [ln for ln in p.read_text().splitlines() if not ln.strip().startswith("#")]
        used = [ln for ln in code if any(r in ln for r in retired)]
        assert not used, f"{p.relative_to(root)}: {used}"
