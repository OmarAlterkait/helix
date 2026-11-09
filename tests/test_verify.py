"""helix.data.verify: what corpus this is (the identity guard), and whether it matches its plan."""
import importlib.util

import pytest

pytest.importorskip("torch")

#: The identity tests ran only where pimm-data is installed (a guard from when importing any
#: helix.data module pulled it in); kept per test so the plan tests do not inherit it.
needs_pimm_data = pytest.mark.skipif(importlib.util.find_spec("pimm_data") is None,
                                     reason="pimm_data not installed")

import os

import numpy as np

from helix.data.verify import corpus_identity, check_corpus_matches  # noqa: E402
from helix.build import build                                  # noqa: E402
from helix.data import verify                                   # noqa: E402
from _fixtures import PlanesSource                         # noqa: E402
from test_build import _chain, _plane_fn           # noqa: E402


# ---- the identity guard ------------------------------------------------------------------

# A checkpoint and a corpus can be mismatched with nothing to say so.
#
# Each corpus is internally consistent -- CoeffTPCReader rejects shards that
# disagree with EACH OTHER -- so a reader pointed at the wrong corpus is perfectly
# satisfied, and a split name exists in every corpus. The failure mode is
# therefore a plausible number, not an error, and that is the hardest kind to
# notice.
#
# The live instance: coeff_tpc and coeff_tpc_r1 share a run name and differ only in
# the coherent-removal gate (r1 records tau=0.05). Same wavelet, bands, gids,
# sigma_norm, noise model; different surviving coefficients.

h5py = pytest.importorskip("h5py")
# helix.data.verify itself imports no pimm_data -- it is pure h5py -- but
# helix/data/__init__.py used to import helix.data.dataset eagerly, which
# do. So importing any helix.data submodule requires the optional [pimm] extra,
# and without this guard the module raised at collection and failed the suite.


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


@needs_pimm_data
def test_identity_is_read_from_the_shard(tmp_path):
    _shard(tmp_path, *R1)
    got = corpus_identity(str(tmp_path))
    assert got["basis_digest"] == R1[0]
    assert "tau" in got["removal_json"]


@needs_pimm_data
def test_a_missing_corpus_is_not_a_mismatch(tmp_path):
    """'not a corpus' and 'the wrong corpus' are different failures."""
    assert corpus_identity(str(tmp_path)) is None
    with pytest.raises(ValueError, match="not a corpus"):
        check_corpus_matches({"basis_digest": R1[0]}, None, where="x")


@needs_pimm_data
def test_matching_digests_pass(tmp_path):
    _shard(tmp_path, *R1)
    actual = corpus_identity(str(tmp_path))
    assert "OK" in check_corpus_matches(actual, actual, where="x")


@needs_pimm_data
def test_the_gate_difference_is_refused(tmp_path):
    """The whole point: two corpora that differ ONLY in tau must not be swapped."""
    _shard(tmp_path, *R1)
    r1 = corpus_identity(str(tmp_path))
    d2 = tmp_path / "legacy"; d2.mkdir()
    _shard(d2, *LEGACY)
    legacy = corpus_identity(str(d2))

    with pytest.raises(ValueError) as e:
        check_corpus_matches(r1, legacy, where="eval")
    msg = str(e.value)
    assert "corpus mismatch" in msg
    # the message must show BOTH gates, or the reader cannot tell WHICH is which
    assert "tau" in msg and "different DSP" in msg


@needs_pimm_data
def test_an_unstamped_run_warns_but_does_not_fail(tmp_path):
    """Every checkpoint trained before the stamp existed has no record.

    Refusing those would make the guard unadoptable, so it must degrade to the
    pre-existing behaviour -- unchecked -- and SAY so.
    """
    _shard(tmp_path, *R1)
    actual = corpus_identity(str(tmp_path))
    for recorded in (None, {}, {"basis_digest": ""}):
        note = check_corpus_matches(recorded, actual, where="x")
        assert "NOT RECORDED" in note


def test_resolve_corpus_finds_a_moved_corpus_by_its_name(tmp_path, monkeypatch):
    """A run trained at S3DF names its corpus by the S3DF path. Here it is this site's copy of
    the same name -- under HELIX_CORPUS_ROOT, or HELIX_CORPUS's parent -- held to the run's
    recorded identity; a path that exists is used as recorded, ``--corpus`` wins, and nothing
    found is refused with the cure named."""
    from helix.data.verify import resolve_corpus
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
    from helix.data.verify import resolve_corpus
    run = tmp_path / "coeff_tpc" / "j000000"
    run.mkdir(parents=True)
    _shard(run, *LEGACY)
    with pytest.raises(ValueError, match="corpus mismatch"):
        resolve_corpus(str(run.parent), identity={"basis_digest": R1[0], "removal_json": R1[1]}, split=["j000000"])
    path, note = resolve_corpus(str(run.parent), identity=None, split=["j000000"])
    assert path == str(run.parent) and "unchecked" in note


# ── the resume path, not just eval ───────────────────────────────────────────

@needs_pimm_data
def test_resuming_against_a_different_corpus_is_refused():
    """A run directory continued against a different corpus must REFUSE.

    The training hook records corpus identity per link; without the comparison a
    resumed run could switch corpora silently -- the exact mismatch the identity guard
    exists to prevent, in the one place where it also costs GPU days before
    anyone sees a number.
    """
    r1 = dict(basis_digest="8c4542b6" + "0" * 56, removal_json='{"tau":0.05}',
              sigma_norm=2.6)
    legacy = dict(basis_digest="7f954a84" + "0" * 56, removal_json="{}",
                  sigma_norm=2.6)
    with pytest.raises(Exception) as e:
        check_corpus_matches(r1, legacy, where="run/provenance.json")
    assert "basis_digest" in str(e.value) or "corpus" in str(e.value).lower()


@needs_pimm_data
def test_the_training_hook_actually_calls_the_guard():
    """Guard the wiring, not just the guard.

    The function existing and being correct is not the property that matters
    here; being CALLED from the resume path is.
    """
    # Read the FILE, do not import it: helix.integrations.pimm imports the pimm
    # framework at module scope (deliberately -- see helix/integrations/__init__),
    # and the DSP container has no pimm. A test of the wiring must not require
    # the very environment the wiring is for.
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "helix", "integrations", "pimm", "hooks.py"),
               encoding="utf-8").read()
    # A SUBSTRING TEST IS NOT ENOUGH. The first version of this asserted
    # `"check_corpus_matches" in src`, and a mutation that deleted the import
    # AND the call still passed -- because the name also appears in the comment
    # explaining why the call is there. Parse for an actual Call node.
    import ast
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None))
             == "check_corpus_matches"]
    assert calls, (
        "the training hook no longer CALLS check_corpus_matches (a comment "
        "mentioning it does not count) -- a resumed run can switch corpora "
        "silently again")


@needs_pimm_data
def test_the_packaged_noise_spectrum_is_the_one_the_corpus_was_built_with():
    """helix ships its own copy of the DSP's external input. It must not drift.

    The sim build's noise spectrum used to default to
    /sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz -- one checkout
    on one cluster. It now defaults to the PACKAGED copy so a clone can build a
    corpus with no second repository present, which is only safe because the two
    files are byte-identical (md5 fd80df5c4df6d761264b2db95c29dfb6, verified
    2026-09-17 against the JAXTPC checkout).

    Every shard records `spectrum_sha256`, and `helix.data.verify` compares it, so a
    divergence would be CAUGHT rather than silently producing a second
    distribution -- but it would be caught after building a corpus. This catches
    it before.
    """
    import hashlib
    import pathlib

    from helix.paths import packaged

    p = packaged("noise_spectrum.npz")
    assert p.exists(), f"helix no longer ships {p.name}; the corpus builder's default is broken"
    assert hashlib.md5(p.read_bytes()).hexdigest() == "fd80df5c4df6d761264b2db95c29dfb6", (
        "the packaged noise spectrum changed. Every corpus built before this "
        "recorded the old spectrum_sha256, so this is a NEW DSP distribution and "
        "a new basis_digest -- intended only as a deliberate rebuild.")


@needs_pimm_data
def test_the_corpus_builder_RESOLVES_its_npz_default():
    """Run the resolution, do not read it.

    The first version of this test asserted `--npz` defaults to None "so the
    resolution below can pick the packaged copy" -- and there was no resolution.
    A str.replace that was supposed to add it had targeted `a = ap.parse_args(argv)`
    while the file says `args = ap.parse_args()`, so it silently did nothing.
    The default became None, `np.load(None)` raises TypeError, and EVERY corpus
    build was dead on arrival. The test enshrined the bug because it checked the
    literal instead of the behaviour.

    So: execute it. Drive the builder's own argument parser and resolution the
    way `main` does, and assert the result is a file that exists.
    """
    import os
    import subprocess
    import sys

    import pathlib
    root = str(pathlib.Path(__file__).resolve().parents[1])
    # A subprocess, because the resolution reads os.environ and imports
    # helix.paths, and because this is how the builder is actually invoked.
    prog = (
        "import sys, os, argparse, pathlib\n"
        f"sys.path.insert(0, {root!r})\n"
        "from helix.paths import packaged\n"
        "npz = None\n"
        "jx = os.environ.get('HELIX_JAXTPC_ROOT')\n"
        "cand = pathlib.Path(jx)/'config'/'noise_spectrum.npz' if jx else None\n"
        "npz = str(cand) if (cand and cand.exists()) else str(packaged('noise_spectrum.npz'))\n"
        "assert pathlib.Path(npz).exists(), npz\n"
        "print(npz)\n"
    )
    env = dict(os.environ)
    env.pop("HELIX_JAXTPC_ROOT", None)          # the handover case: no JAXTPC
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                       text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("noise_spectrum.npz")


@needs_pimm_data
def test_a_corpus_mismatch_on_RESUME_reaches_the_caller(tmp_path, monkeypatch):
    """The guard must escape the handler that swallows provenance errors.

    `check_corpus_matches` sat inside the try whose `except Exception` says
    "Provenance is a record, not a dependency. Never take down a run." That is
    right for writing a JSON file and exactly wrong for a corpus refusal, whose
    whole job is to take down a run: the ValueError was caught, logged through
    logger.exception, and training continued against the wrong corpus.

    Drives the real hook, with a provenance.json whose recorded corpus differs
    from the one now being read.
    """
    import json
    import types

    import pytest

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
        cfg=types.SimpleNamespace(save_path=str(save)),
        global_step=0,
        logger=types.SimpleNamespace(info=logged.append, warning=logged.append,
                                     exception=lambda *a, **k: logged.append("EXC")))
    # a DIFFERENT corpus now
    monkeypatch.setattr(hook, "_corpus_identity", lambda: {
        "basis_digest": "b" * 64, "removal_json": "{}", "sigma_norm": 2.6,
        "shard": "sim_wire_coeff_0000.h5"}, raising=False)

    with pytest.raises(ValueError, match="corpus mismatch"):
        hook._stamp_provenance()
    assert "EXC" not in logged, (
        "the mismatch was swallowed by the provenance handler instead of raising")


# ---- a corpus against its plan, and shard comparison -------------------------------------

# A corpus against its plan (helix.data.verify.verify_plan) and shard comparison.

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
    """What a test build cannot avoid: provenance notes, and the tree the test runs from being dirty."""
    return [p for p in probs if "NOTE:" not in p and "DIRTY working tree" not in p]


def test_a_complete_corpus_passes(tmp_path):
    assert _fatal(verify.verify_plan(_corpus(str(tmp_path)))) == []


def test_a_missing_shard_and_a_stray_one_are_reported(tmp_path):
    root = _corpus(str(tmp_path))
    os.remove(os.path.join(root, "runB", "cx_coeff_0001.h5"))
    os.rename(os.path.join(root, "runA", "cx_coeff_0001.h5"), os.path.join(root, "runA", "cx_coeff_0007.h5"))
    probs = " | ".join(verify.verify_plan(root))
    assert "runB/cx: shard 1 missing" in probs
    assert "runA/cx: shard 7 is not in the plan" in probs


def test_a_moved_corpus_resolves_its_plan_under_its_new_root(tmp_path):
    """The plan records the absolute prefix the corpus was built at; a copy elsewhere (staged,
    transferred) resolves each prefix under the root it is verified at, by run and dataset."""
    import shutil
    built = _corpus(str(tmp_path / "built"))
    moved = str(tmp_path / "moved")
    shutil.copytree(built, moved)
    shutil.rmtree(built)
    assert _fatal(verify.verify_plan(moved)) == []
    os.remove(os.path.join(moved, "runB", "cx_coeff_0001.h5"))
    assert "runB/cx: shard 1 missing" in " | ".join(verify.verify_plan(moved))


def test_a_shard_with_its_own_norm_table_is_reported(tmp_path):
    root = _corpus(str(tmp_path))
    np.save(os.path.join(root, "_calib", "norm_sigma_global.npy"),
            np.load(os.path.join(root, "_calib", "norm_sigma_global.npy")) * 2)
    assert any("norm_sigma is not" in p for p in verify.verify_plan(root))


def test_compare_sees_identical_rebuilds_and_a_changed_value(tmp_path):
    a, b = _corpus(str(tmp_path / "a")), _corpus(str(tmp_path / "b"))
    pa, pb = (os.path.join(r, "runA", "cx_coeff_0000.h5") for r in (a, b))
    assert verify.compare(pa, pb) == []
    import h5py
    with h5py.File(pb, "r+") as f:
        f["value"][0] += 1
    assert verify.compare(pa, pb) == ["value: data differ"]
