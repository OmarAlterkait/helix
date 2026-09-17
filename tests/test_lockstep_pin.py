"""The pimm-data revision is named in three places. They must agree.

The AddNoise/Digitize registration is LOCKSTEP: pimm-data dropped the LArTPC
forward model and helix picked it up, and `pimm_data/_registry.py` raises
KeyError on a duplicate registration. So an environment that straddles the
boundary move does not degrade — it fails to import.

The revision is therefore pinned in helix's `pyproject.toml` (for a dev install),
in `container/helix-train.def` (for the image), and in pimm's `coeff-fm` branch
(for the trainer's environment). The first two always live here. The THIRD is
checked too whenever a pimm checkout is reachable -- it is the one that has
actually gone stale: pimm's `eval-contract` branch pins a revision from before
the forward model moved, and merging it would put the environment on the wrong
side of the lockstep.
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


def _pimm_pyproject():
    """pimm's pyproject, if a checkout is reachable. Skip rather than guess."""
    import os

    root = os.environ.get("HELIX_PIMM_ROOT")
    if not root:
        try:
            from helix.paths import root as _r
            root = str(_r("HELIX_PIMM_ROOT"))
        except Exception:
            return None
    p = pathlib.Path(root) / "pyproject.toml"
    return p if p.exists() else None


def test_the_pimm_checkout_pins_the_same_pimm_data():
    """The third pin, checked when it is reachable.

    This is the one that goes stale unnoticed: it lives in another repository,
    on a branch, and nothing in helix's CI can see it. pimm's `eval-contract`
    branch pins 74cfe5f (pimm-data 0.3.0, pre-boundary) while `coeff-fm` pins
    6b2656a -- so which branch is checked out decides whether the environment
    imports at all.
    """
    p = _pimm_pyproject()
    if p is None:
        pytest.skip("no pimm checkout reachable (set HELIX_PIMM_ROOT)")
    m = re.search(r'pimm-data\s*=\s*\{[^}]*rev\s*=\s*"([0-9a-f]{40})"', p.read_text())
    if not m:
        pytest.skip(f"{p} does not pin pimm-data by an explicit rev")
    assert m.group(1) == _pyproject_rev(), (
        f"{p} pins pimm-data {m.group(1)[:8]} but helix pins "
        f"{_pyproject_rev()[:8]}. An environment built from that pimm checkout "
        f"straddles the forward-model boundary move: whichever side registers "
        f"AddNoise second raises KeyError at import. Check which BRANCH is "
        f"checked out -- eval-contract pins a pre-boundary revision.")
