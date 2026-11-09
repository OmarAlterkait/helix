"""helix.data.verify: what corpus this is (the identity guard), and whether it is whole."""
import json
import os
import shutil

import h5py
import numpy as np
import pytest

pytest.importorskip("torch")

from helix.build import build                                  # noqa: E402
from helix.data import verify                                   # noqa: E402
from helix.data.verify import check_corpus_matches, corpus_identity, resolve_corpus  # noqa: E402
from _fixtures import PlanesSource, write_coeff_shard          # noqa: E402
from test_build import _chain, _plane_fn                        # noqa: E402


# ---- the identity guard ------------------------------------------------------------------

# coeff_tpc and coeff_tpc_r1 share a run name and differ only in the coherent-removal gate (r1
# records tau=0.05): same wavelet, bands, gids, sigma_norm and noise model, different surviving
# coefficients. A reader pointed at the wrong one is perfectly satisfied, so the failure is a
# plausible number, not an error.

def _shard(tmp_path, digest, removal, name="sim_wire_coeff_0000.h5"):
    p = tmp_path / name
    with h5py.File(p, "w") as f:
        cfg = f.create_group("config")
        cfg.attrs["basis_digest"] = digest
        cfg.attrs["removal_json"] = removal
        cfg.attrs["sigma_norm"] = 2.6
    return p


R1 = ('8c4542b6', '{"kgate": 3.0, "npass": 2, "tau": 0.05}')
LEGACY = ('7f954a84', '{"kgate": 3.0, "npass": 2}')


def test_identity_is_read_from_the_shard_and_matching_digests_pass(tmp_path):
    _shard(tmp_path, *R1)
    got = corpus_identity(str(tmp_path))
    assert got["basis_digest"] == R1[0] and "tau" in got["removal_json"]
    assert "OK" in check_corpus_matches(got, got, where="x")


def test_a_missing_corpus_is_not_a_mismatch(tmp_path):
    """'not a corpus' and 'the wrong corpus' are different failures."""
    assert corpus_identity(str(tmp_path)) is None
    with pytest.raises(ValueError, match="not a corpus"):
        check_corpus_matches({"basis_digest": R1[0]}, None, where="x")


def test_the_gate_difference_is_refused(tmp_path):
    """Two corpora that differ ONLY in tau must not be swapped, and the message shows both gates."""
    _shard(tmp_path, *R1)
    d2 = tmp_path / "legacy"; d2.mkdir()
    _shard(d2, *LEGACY)
    with pytest.raises(ValueError) as e:
        check_corpus_matches(corpus_identity(str(tmp_path)), corpus_identity(str(d2)), where="eval")
    assert "corpus mismatch" in str(e.value) and "tau" in str(e.value) and "different DSP" in str(e.value)


def test_an_unstamped_run_warns_but_does_not_fail(tmp_path):
    """Checkpoints trained before the stamp have no record: unchecked, and saying so."""
    _shard(tmp_path, *R1)
    actual = corpus_identity(str(tmp_path))
    for recorded in (None, {}, {"basis_digest": ""}):
        assert "NOT RECORDED" in check_corpus_matches(recorded, actual, where="x")


def test_resolve_corpus_finds_a_moved_corpus_by_its_name(tmp_path, monkeypatch):
    """A run trained at S3DF names its corpus by the S3DF path. Here it is this site's copy of
    the same name -- under HELIX_CORPUS_ROOT, or HELIX_CORPUS's parent -- held to the run's
    recorded identity; a path that exists is used as recorded, ``--corpus`` wins, and nothing
    found is refused with the cure named."""
    here = tmp_path / "scratch"
    corpus = here / "coeff_tpc_d1m"
    (corpus / "j000000").mkdir(parents=True)
    _shard(corpus / "j000000", *R1)
    ident = corpus_identity(str(corpus / "j000000"))
    rec = "/sdf/data/neutrino/omara/coeff_tpc_d1m"
    monkeypatch.setenv("HELIX_CORPUS_ROOT", str(here))
    monkeypatch.setenv("HELIX_CORPUS", str(tmp_path / "unrelated" / "j000000"))
    path, note = resolve_corpus(rec, identity=ident, split=["j000000"])
    assert path == str(corpus) and "identity OK" in note
    monkeypatch.setenv("HELIX_CORPUS_ROOT", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("HELIX_CORPUS", str(corpus / "j000000"))
    assert resolve_corpus(rec, identity=ident, split=["j000000"])[0] == str(corpus)
    assert resolve_corpus(str(corpus), identity=ident, split="j000000")[0] == str(corpus)
    other = tmp_path / "copy" / "j000000"
    other.mkdir(parents=True)
    _shard(other, *R1)
    assert resolve_corpus(rec, override=str(other.parent), identity=ident, split=["j000000"])[0] == str(other.parent)
    with pytest.raises(FileNotFoundError, match="--corpus"):
        resolve_corpus("/sdf/data/neutrino/omara/coeff_tpc_nope", identity=ident)


def test_resolve_corpus_refuses_another_dsp_and_passes_an_unstamped_run(tmp_path):
    run = tmp_path / "coeff_tpc" / "j000000"
    run.mkdir(parents=True)
    _shard(run, *LEGACY)
    with pytest.raises(ValueError, match="corpus mismatch"):
        resolve_corpus(str(run.parent), identity={"basis_digest": R1[0], "removal_json": R1[1]}, split=["j000000"])
    path, note = resolve_corpus(str(run.parent), identity=None, split=["j000000"])
    assert path == str(run.parent) and "unchecked" in note


def test_a_corpus_mismatch_on_RESUME_reaches_the_caller(tmp_path, monkeypatch):
    """The guard must escape the training hook's handler that swallows provenance errors: the
    ValueError was once caught, logged, and training continued against the wrong corpus."""
    import types
    pytest.importorskip("pimm")
    from helix.integrations.pimm.hooks import HelixProvenance

    save = tmp_path / "run"
    save.mkdir()
    (save / "provenance.json").write_text(json.dumps([{
        "corpus": {"basis_digest": "a" * 64, "removal_json": "{}", "sigma_norm": 2.6,
                   "shard": "sim_wire_coeff_0000.h5"}}]))
    hook = HelixProvenance.__new__(HelixProvenance)
    logged = []
    hook.trainer = types.SimpleNamespace(
        cfg=types.SimpleNamespace(save_path=str(save)), global_step=0,
        logger=types.SimpleNamespace(info=logged.append, warning=logged.append,
                                     exception=lambda *a, **k: logged.append("EXC")))
    monkeypatch.setattr(hook, "_corpus_identity", lambda: {
        "basis_digest": "b" * 64, "removal_json": "{}", "sigma_norm": 2.6,
        "shard": "sim_wire_coeff_0000.h5"}, raising=False)
    with pytest.raises(ValueError, match="corpus mismatch"):
        hook._stamp_provenance()
    assert "EXC" not in logged, "the mismatch was swallowed by the provenance handler"


def test_the_packaged_noise_spectrum_is_the_one_the_corpus_was_built_with():
    """helix ships its own copy of the DSP's external input (byte-identical to the JAXTPC
    checkout's, md5 verified 2026-09-17). A change is a new DSP distribution: a deliberate rebuild."""
    import hashlib
    from helix.paths import packaged
    p = packaged("noise_spectrum.npz")
    assert p.exists(), f"helix no longer ships {p.name}; the corpus builder's default is broken"
    assert hashlib.md5(p.read_bytes()).hexdigest() == "fd80df5c4df6d761264b2db95c29dfb6"


# ---- a corpus against its plan -----------------------------------------------------------

def _corpus(root, runs=("runA", "runB"), n=4, eps=2):
    """Two runs of ``n`` events, ``eps`` per shard, one global norm table, and their plan."""
    os.makedirs(os.path.join(root, "_calib"))
    lines, chain = [], _chain()
    for i, run in enumerate(runs):
        prefix = os.path.join(root, run, "cx")
        src = PlanesSource([(f"src_{run}.h5", e) for e in range(n * i, n * (i + 1))], _plane_fn, run=run)
        if i == 0:
            cal = os.path.join(root, "_calib", "norm_sigma_global.npy")
            build(src, prefix, indices=list(range(n)), chain=chain, calibrate=cal)
        build(src, prefix, indices=list(range(n)), chain=chain, events_per_shard=eps,
              norm_sigma=np.load(cal))
        lines.append(f"{prefix} 0 {n} {eps} 0 src_{run}.h5")
    open(os.path.join(root, "_calib", "CHUNKS.txt"), "w").write("\n".join(lines) + "\n")
    return root


def _fatal(probs):
    """What a test build cannot avoid: notes, and the tree the test runs from being dirty."""
    return [p for p in probs if not p.startswith("NOTE:") and "DIRTY working tree" not in p]


def test_a_complete_corpus_passes(tmp_path):
    root = _corpus(str(tmp_path))
    assert _fatal(verify.verify(root)) == []


def test_a_missing_shard_a_stray_one_and_a_short_one_are_reported(tmp_path):
    root = _corpus(str(tmp_path))
    os.remove(os.path.join(root, "runB", "cx_coeff_0001.h5"))
    os.rename(os.path.join(root, "runA", "cx_coeff_0001.h5"), os.path.join(root, "runA", "cx_coeff_0007.h5"))
    probs = " | ".join(verify.verify(root))
    assert "runB/cx: shard 1 missing" in probs
    assert "runA/cx: shard 7 is not in the plan" in probs
    lines = open(os.path.join(root, "_calib", "CHUNKS.txt")).read().replace(" 0 4 2 0 src_runA", " 0 2 1 0 src_runA")
    open(os.path.join(root, "_calib", "CHUNKS.txt"), "w").write(lines)
    assert "cx_coeff_0000.h5: 2 events, planned 1" in " | ".join(verify.verify(root))


def test_a_gap_is_judged_against_the_plan(tmp_path):
    """An input absent upstream leaves a COMPLETE corpus with a shard number unused: the plan is
    the evidence. One run directory alone, with no plan, reports every gap."""
    root = _corpus(str(tmp_path), runs=("runA",))
    os.rename(os.path.join(root, "runA", "cx_coeff_0001.h5"), os.path.join(root, "runA", "cx_coeff_0003.h5"))
    prefix = os.path.join(root, "runA", "cx")
    open(os.path.join(root, "_calib", "CHUNKS.txt"), "w").write(
        f"{prefix} 0 2 2 0 src_runA.h5\n{prefix} 2 2 2 3 src_runA.h5\n")
    assert _fatal(verify.verify(root)) == []
    assert any("shards [1, 2]" in p for p in verify.verify(os.path.join(root, "runA"), "cx"))


def test_a_moved_corpus_resolves_its_plan_under_its_new_root(tmp_path):
    """The plan records the absolute prefix the corpus was built at; a copy elsewhere (staged,
    transferred) resolves each prefix under the root it is verified at, by run and dataset."""
    built = _corpus(str(tmp_path / "built"))
    moved = str(tmp_path / "moved")
    shutil.copytree(built, moved)
    shutil.rmtree(built)
    assert _fatal(verify.verify(moved)) == []
    os.remove(os.path.join(moved, "runB", "cx_coeff_0001.h5"))
    assert "runB/cx: shard 1 missing" in " | ".join(verify.verify(moved))


def test_a_shard_with_its_own_norm_table_is_reported(tmp_path):
    root = _corpus(str(tmp_path))
    np.save(os.path.join(root, "_calib", "norm_sigma_global.npy"),
            np.load(os.path.join(root, "_calib", "norm_sigma_global.npy")) * 2)
    assert any("norm_sigma is not" in p for p in verify.verify(root))


def test_compare_sees_identical_rebuilds_and_a_changed_value(tmp_path):
    a, b = _corpus(str(tmp_path / "a")), _corpus(str(tmp_path / "b"))
    pa, pb = (os.path.join(r, "runA", "cx_coeff_0000.h5") for r in (a, b))
    assert verify.compare(pa, pb) == []
    with h5py.File(pb, "r+") as f:
        f["value"][0] += 1
    assert verify.compare(pa, pb) == ["value: data differ"]


# ---- what spans shards, on one run directory ---------------------------------------------

from test_coeff_dataset import BAND_LENGTHS, BASIS_ATTRS, FIXTURE_PROV, GIDS, N_WIRES, _ev  # noqa: E402

NS = np.full((len(GIDS), len(BAND_LENGTHS)), 2.0, np.float32)


def _run(d, shards, norm=(), prov=()):
    """``cx_coeff_<k>.h5`` of the given event ids per shard k; ``norm`` / ``prov``: per-shard
    overrides of the norm table and the provenance."""
    d.mkdir(exist_ok=True)
    for k, ids in enumerate(shards):
        rng = np.random.default_rng(k)
        write_coeff_shard(d / f"cx_coeff_{k:04d}.h5", [_ev(rng, e) for e in ids], band_lengths=BAND_LENGTHS,
                          gids=GIDS, n_wires=N_WIRES, basis_attrs=BASIS_ATTRS, dataset_name="cx", file_index=k,
                          norm_sigma=dict(norm).get(k, NS), provenance=dict(prov).get(k, FIXTURE_PROV))
    return str(d)


def test_cross_shard_damage_is_reported(tmp_path):
    """audit_shard is per file and cannot see these: each leaves every shard valid while changing
    what the corpus holds."""
    ok = _run(tmp_path / "ok", [[0, 1], [2, 3], [4, 5]])
    assert verify.verify(ok, "cx", expect_events=6) == []
    assert verify.main([ok, "--dataset-name", "cx", "--modalities", "coeff", "--expect", "6"]) == 0      # RUNBOOK's form
    gap = _run(tmp_path / "gap", [[0, 1], [2, 3], [4, 5]])
    os.remove(os.path.join(gap, "cx_coeff_0001.h5"))
    assert any("file_index gap" in p for p in verify.verify(gap, "cx"))
    norm = _run(tmp_path / "norm", [[0, 1], [2, 3]], norm=[(1, NS * 4.5)])
    assert any("disagree on norm_sigma" in p for p in verify.verify(norm, "cx"))
    dup = _run(tmp_path / "dup", [[0, 1, 2], [2, 3, 4]])
    assert any("more than once" in p for p in verify.verify(dup, "cx"))
    assert any("expected 7" in p for p in verify.verify(dup, "cx", expect_events=7))


def test_the_builder_and_the_DSP_inputs_must_be_one(tmp_path):
    """One helix commit, a clean tree, one geometry / spectrum / noise model across the corpus."""
    other = dict(FIXTURE_PROV, code=dict(FIXTURE_PROV["code"], git="1" * 40))
    probs = verify.verify(_run(tmp_path / "commit", [[0, 1], [2, 3]], prov=[(1, other)]), "cx")
    assert any("DIFFERENT helix commits" in p for p in probs), probs
    dirty = dict(FIXTURE_PROV, code=dict(FIXTURE_PROV["code"], git_dirty=True))
    assert any("DIRTY" in p for p in verify.verify(_run(tmp_path / "dirty", [[0, 1]], prov=[(0, dirty)]), "cx"))
    nogit = dict(FIXTURE_PROV, code=dict(version="0.0.0-test"))
    assert _fatal(verify.verify(_run(tmp_path / "timeout", [[0, 1], [2, 3]], prov=[(1, nogit)]), "cx")) == []
    assert any("records the helix commit" in p
               for p in verify.verify(_run(tmp_path / "none", [[0, 1]], prov=[(0, nogit)]), "cx"))
    for name, second in (("geom", dict(FIXTURE_PROV, geom_sha256="b" * 8)),
                         ("spectrum", dict(FIXTURE_PROV, spectrum_sha256=None))):
        probs = verify.verify(_run(tmp_path / name, [[0, 1], [2, 3]], prov=[(1, second)]), "cx")
        assert any("DSP inputs" in p for p in probs), f"{name}: {probs}"
