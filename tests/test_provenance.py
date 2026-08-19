"""A run directory must say which code produced it.

Nothing recorded this. A run directory held weights, a config and a log, and no
statement of which helix commit produced them — so reconstructing the setup for
an existing checkpoint meant trying configurations until the goldens matched.
"""
import importlib.util
import json
import os
import subprocess
import types

import pytest

from helix.integrations._bootstrap import describe_checkout, provenance


def _pimm_importable():
    try:
        if importlib.util.find_spec("pimm") is None:
            return False
        import pimm.datasets.builder  # noqa: F401
        return True
    except Exception:
        return False


#: `describe_checkout`/`provenance` are pure helix and run anywhere; only the
#: hook that CALLS them needs pimm. Splitting the guard keeps the git-probing
#: tests running in the container that has no pimm, which is where they belong.
needs_pimm = pytest.mark.skipif(not _pimm_importable(), reason="pimm not importable")


def _git(cwd, *a):
    subprocess.run(("git",) + a, cwd=cwd, check=True, capture_output=True)


def test_describe_checkout_reports_commit_branch_and_clean(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    _git(d, "init", "-q", "-b", "main")
    _git(d, "config", "user.email", "t@t")
    _git(d, "config", "user.name", "t")
    (d / "a.txt").write_text("x")
    _git(d, "add", "a.txt")
    _git(d, "commit", "-qm", "one")

    got = describe_checkout(str(d))
    assert got["commit"] and len(got["commit"]) == 40
    assert got["branch"] == "main"
    assert got["dirty"] is False


def test_dirty_is_reported(tmp_path):
    """The field that matters: a hash alone says what was COMMITTED, not what ran."""
    d = tmp_path / "repo"
    d.mkdir()
    _git(d, "init", "-q", "-b", "main")
    _git(d, "config", "user.email", "t@t")
    _git(d, "config", "user.name", "t")
    (d / "a.txt").write_text("x")
    _git(d, "add", "a.txt")
    _git(d, "commit", "-qm", "one")
    (d / "a.txt").write_text("modified")           # uncommitted edit
    assert describe_checkout(str(d))["dirty"] is True


def test_a_non_git_directory_degrades_instead_of_raising(tmp_path):
    got = describe_checkout(str(tmp_path))
    assert got == {"root": str(tmp_path), "commit": None, "dirty": None,
                   "branch": None}


def test_a_gitdir_FILE_is_still_a_repo(tmp_path):
    """The bug that produced an all-null record from a real run.

    A worktree (and a submodule) has `.git` as a FILE pointing at the real
    gitdir, so `os.path.isdir(root/'.git')` is False for a perfectly good
    checkout — which is exactly what helix-extraction is.
    """
    d = tmp_path / "repo"
    d.mkdir()
    _git(d, "init", "-q", "-b", "main")
    _git(d, "config", "user.email", "t@t")
    _git(d, "config", "user.name", "t")
    (d / "a.txt").write_text("x")
    _git(d, "add", "a.txt")
    _git(d, "commit", "-qm", "one")
    wt = tmp_path / "wt"
    _git(d, "worktree", "add", "-q", str(wt))
    assert not (wt / ".git").is_dir() and (wt / ".git").is_file(), "not a worktree"
    got = describe_checkout(str(wt))
    assert got["commit"], "a worktree must still report its commit"
    assert got["dirty"] is False


def test_a_subdirectory_resolves_to_the_repo_root(tmp_path):
    """pimm_data is a src-layout package, so its `root` is <repo>/src."""
    d = tmp_path / "repo"
    (d / "src").mkdir(parents=True)
    _git(d, "init", "-q", "-b", "main")
    _git(d, "config", "user.email", "t@t")
    _git(d, "config", "user.name", "t")
    (d / "src" / "a.txt").write_text("x")
    _git(d, "add", ".")
    _git(d, "commit", "-qm", "one")
    got = describe_checkout(str(d / "src"))
    assert got["commit"], "a package below the repo root must still be described"
    assert os.path.realpath(got["root"]) == os.path.realpath(str(d))


def test_provenance_covers_helix_and_pimm_data():
    p = provenance()
    for k in ("helix", "pimm_data", "python", "hostname"):
        assert k in p, f"provenance is missing {k}"
    assert p["helix"]["root"].endswith("helix-extraction") or p["helix"]["commit"]


@needs_pimm
def test_stamp_appends_so_a_resumed_link_does_not_erase_the_first(tmp_path):
    """A run resumed from a DIFFERENT tree is the case worth catching."""
    from helix.integrations.pimm import HelixPathBootstrap

    h = HelixPathBootstrap()
    logged = []
    h.trainer = types.SimpleNamespace(
        cfg=types.SimpleNamespace(save_path=str(tmp_path)),
        global_step=11,
        logger=types.SimpleNamespace(info=logged.append, warning=logged.append,
                                     exception=lambda *a, **k: logged.append("EXC")))
    h._stamp_provenance()
    h.trainer.global_step = 22
    h._stamp_provenance()

    assert "EXC" not in logged, f"stamping raised: {logged}"
    log = json.loads((tmp_path / "provenance.json").read_text())
    assert isinstance(log, list) and len(log) == 2, "each link must leave a record"
    assert [r["step"] for r in log] == [11, 22]
    assert log[0]["helix"]["root"]


@needs_pimm
def test_stamp_upgrades_a_single_dict_written_by_an_older_run(tmp_path):
    from helix.integrations.pimm import HelixPathBootstrap

    (tmp_path / "provenance.json").write_text(json.dumps({"step": 0, "helix": {}}))
    h = HelixPathBootstrap()
    h.trainer = types.SimpleNamespace(
        cfg=types.SimpleNamespace(save_path=str(tmp_path)), global_step=5,
        logger=types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None,
                                     exception=lambda *a, **k: pytest.fail("raised")))
    h._stamp_provenance()
    log = json.loads((tmp_path / "provenance.json").read_text())
    assert len(log) == 2 and log[1]["step"] == 5


@needs_pimm
def test_stamping_never_takes_down_a_run(tmp_path):
    """Provenance is a record, not a dependency."""
    from helix.integrations.pimm import HelixPathBootstrap

    h = HelixPathBootstrap()
    seen = []
    h.trainer = types.SimpleNamespace(
        cfg=types.SimpleNamespace(save_path=str(tmp_path / "does" / "not" / "exist")),
        global_step=0,
        logger=types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None,
                                     exception=lambda *a, **k: seen.append("logged")))
    h._stamp_provenance()                          # must not raise
    assert seen == ["logged"]
