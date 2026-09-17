"""A checkpoint and a corpus can be mismatched with nothing to say so.

Each corpus is internally consistent -- CoeffTPCReader rejects shards that
disagree with EACH OTHER -- so a reader pointed at the wrong corpus is perfectly
satisfied. eval_checkpoint validated only that the split name exists, which every
corpus satisfies. The failure mode was therefore a plausible number, not an
error, and that is the hardest kind to notice.

The live instance: coeff_tpc and coeff_tpc_r1 share a run name and differ only in
the coherent-removal gate (r1 records tau=0.05). Same wavelet, bands, gids,
sigma_norm, noise model; different surviving coefficients.
"""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
# identity.py itself imports no pimm_data -- it is pure h5py -- but
# helix/data/__init__.py eagerly imports coeff_reader and coeff_dataset, which
# do. So importing any helix.data submodule requires the optional [pimm] extra,
# and without this guard the module raised at collection and failed the suite.
pytest.importorskip("pimm_data")

from helix.data.identity import corpus_identity, check_corpus_matches  # noqa: E402


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


def test_identity_is_read_from_the_shard(tmp_path):
    _shard(tmp_path, *R1)
    got = corpus_identity(str(tmp_path))
    assert got["basis_digest"] == R1[0]
    assert "tau" in got["removal_json"]


def test_a_missing_corpus_is_not_a_mismatch(tmp_path):
    """'not a corpus' and 'the wrong corpus' are different failures."""
    assert corpus_identity(str(tmp_path)) is None
    with pytest.raises(ValueError, match="not a corpus"):
        check_corpus_matches({"basis_digest": R1[0]}, None, where="x")


def test_matching_digests_pass(tmp_path):
    _shard(tmp_path, *R1)
    actual = corpus_identity(str(tmp_path))
    assert "OK" in check_corpus_matches(actual, actual, where="x")


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


# ── the resume path, not just eval ───────────────────────────────────────────

def test_resuming_against_a_different_corpus_is_refused():
    """A run directory continued against a different corpus must REFUSE.

    check_corpus_matches was wired into scripts/eval_checkpoint.py alone. The
    training hook RECORDED corpus identity per link and never compared it, so a
    resumed run could switch corpora silently -- which is the exact mismatch
    identity.py exists to prevent, in the one place where it also costs GPU
    days before anyone sees a number.
    """
    r1 = dict(basis_digest="8c4542b6" + "0" * 56, removal_json='{"tau":0.05}',
              sigma_norm=2.6)
    legacy = dict(basis_digest="7f954a84" + "0" * 56, removal_json="{}",
                  sigma_norm=2.6)
    with pytest.raises(Exception) as e:
        check_corpus_matches(r1, legacy, where="run/provenance.json")
    assert "basis_digest" in str(e.value) or "corpus" in str(e.value).lower()


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


def test_write_holdout_exists_and_is_wired():
    """The split must be materialisable, not only computable.

    configs/pimm/coeff_fm_train.py says the split is "Resolved once and written
    to holdout.json beside the corpus" -- and nothing wrote it.
    CoeffTPCDataset.holdout_manifest() computed it, dump_probe_truth.py and
    run_probe.py READ the file, and no code put it there. Result: the probe could
    not run on any corpus built from scratch, and in fact 7 of the 8 production
    runs have no holdout.json either.
    """
    import ast, os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(here, "scripts", "write_holdout.py")
    assert os.path.exists(src), "scripts/write_holdout.py is gone; the probe cannot run"
    tree = ast.parse(open(src, encoding="utf-8").read())
    calls = {getattr(n.func, "attr", None) for n in ast.walk(tree)
             if isinstance(n, ast.Call)}
    assert "holdout_manifest" in calls, (
        "write_holdout.py no longer calls holdout_manifest() -- it must derive the "
        "split from the dataset, not re-implement the identity hash")


def test_the_packaged_noise_spectrum_is_the_one_the_corpus_was_built_with():
    """helix ships its own copy of the DSP's external input. It must not drift.

    `scripts/build_coeff_corpus.py --npz` used to default to
    /sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz -- one checkout
    on one cluster. It now defaults to the PACKAGED copy so a clone can build a
    corpus with no second repository present, which is only safe because the two
    files are byte-identical (md5 fd80df5c4df6d761264b2db95c29dfb6, verified
    2026-09-17 against the JAXTPC checkout).

    Every shard records `spectrum_sha256`, and `coeff_verify` compares it, so a
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


def test_the_corpus_builder_defaults_to_something_that_exists():
    """The default must resolve without a second repository on the machine."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_coeff_corpus.py"
    import ast as _ast

    src = path.read_text()
    # Parse the argparse call, do not grep the file: the old path still appears
    # in a COMMENT explaining why it was removed, and a substring test flagged
    # that as a regression. Assert on the default VALUE.
    tree = _ast.parse(src)
    npz = [c for c in _ast.walk(tree)
           if isinstance(c, _ast.Call)
           and getattr(c.func, "attr", "") == "add_argument"
           and c.args and getattr(c.args[0], "value", "") == "--npz"]
    assert len(npz) == 1, "expected exactly one --npz argument"
    default = [k.value for k in npz[0].keywords if k.arg == "default"]
    assert default and isinstance(default[0], _ast.Constant) and default[0].value is None, (
        "--npz must default to None so the resolution can pick the packaged "
        "copy or $HELIX_JAXTPC_ROOT; a literal path default cannot fall back, "
        "and the literal it used to carry was one person's checkout")
    # ...and the resolution must EXIST. A None default with nothing resolving it
    # is strictly worse than the hardcoded path it replaced.
    assert "if args.npz is None:" in src, (
        "--npz defaults to None and nothing resolves it; np.load(None) raises "
        "TypeError and every corpus build dies at startup")
    assert "packaged(" in src, "the resolution must reach helix's packaged copy"
