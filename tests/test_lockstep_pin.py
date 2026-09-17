"""The pimm-data revision is named in three places. They must agree.

The AddNoise/Digitize registration is LOCKSTEP: pimm-data dropped the LArTPC
forward model and helix picked it up, and `pimm_data/_registry.py` raises
KeyError on a duplicate registration. So an environment that straddles the
boundary move does not degrade — it fails to import.

The revision is therefore pinned in helix's `pyproject.toml` (for a dev install),
in `container/helix-train.def` (for the image), and in pimm's `coeff-fm` branch
(for the trainer's environment). The first two live in this repo and are checked
here. The third is in another repository and cannot be, which is exactly why the
pin carries a comment saying so in all three.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_REV = re.compile(r"\b([0-9a-f]{40})\b")


def _pyproject_rev():
    txt = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'pimm-data\s*=\s*\{[^}]*rev\s*=\s*"([0-9a-f]{40})"', txt)
    assert m, "pyproject.toml no longer pins pimm-data by an explicit 40-char rev"
    return m.group(1)


def _container_rev():
    txt = (ROOT / "container" / "helix-train.def").read_text()
    m = re.search(r"PIMM_DATA_REV=([0-9a-f]{40})", txt)
    assert m, "helix-train.def no longer sets PIMM_DATA_REV to an explicit rev"
    return m.group(1)


def test_the_container_and_the_project_pin_the_same_pimm_data():
    assert _container_rev() == _pyproject_rev(), (
        "container/helix-train.def and pyproject.toml pin DIFFERENT pimm-data "
        "revisions. One of the two environments straddles the forward-model "
        "boundary move, and the one that does raises KeyError on a duplicate "
        "AddNoise registration at import.")


def test_the_lockfile_agrees_with_the_pin():
    """A lockfile that disagrees with the pin is the failure mode this format has.

    `uv lock` records what it RESOLVED, not what was asked for. The original pin
    named the repository with no rev, so the lockfile was the only record of
    which revision got used — and it had silently settled on a pre-boundary one.
    """
    lock = ROOT / "uv.lock"
    if not lock.exists():
        pytest.skip("no uv.lock yet (run `uv lock`)")
    txt = lock.read_text()
    assert _pyproject_rev() in txt, (
        "uv.lock does not mention the pinned pimm-data revision; re-run `uv lock`")


def test_no_stale_revision_is_left_anywhere():
    """Catches a half-done bump: one file updated, another missed."""
    revs = {}
    for rel in ("pyproject.toml", "container/helix-train.def"):
        for r in _REV.findall((ROOT / rel).read_text()):
            revs.setdefault(r, []).append(rel)
    pimm_data_revs = {r for r, files in revs.items() if len(files) == 2}
    assert len(pimm_data_revs) == 1, (
        f"expected exactly one revision named in both files, found "
        f"{sorted(pimm_data_revs)} — a bump was applied to one file only")
