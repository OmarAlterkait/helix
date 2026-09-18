"""The pimm-data revision must be named ONCE here, and agree with pimm's.

The AddNoise/Digitize registration is LOCKSTEP: pimm-data dropped the LArTPC
forward model and helix picked it up, and `pimm_data/_registry.py` raises
KeyError on a duplicate registration. So an environment that straddles the
boundary move does not degrade — it fails to import.

It used to be pinned in TWO files here -- `pyproject.toml` and
`container/helix-train.def` -- and these tests existed to catch a bump applied to
one of them. They no longer can disagree: `container/install.sh` READS the
revision out of `pyproject.toml`, and both the def and the Dockerfile call it. So
what is checked here is that the single source still exists and is still a
40-character revision, and that nothing has quietly reintroduced a second copy.

The pin in pimm's `coeff-fm` branch is the one that can still go stale, because
it lives in another repository and nothing in helix's CI can see it. It is
checked whenever a pimm checkout is reachable: pimm's `eval-contract` branch pins
a revision from before the forward model moved, so which branch is checked out
decides whether the environment imports at all.
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


#: Everything that builds an environment, and must therefore not carry its own
#: copy of the revision.
_BUILD_FILES = ("container/install.sh", "container/helix-train.def",
                "container/Dockerfile")


def test_the_build_reads_the_pin_rather_than_repeating_it():
    """install.sh derives the revision from pyproject.toml.

    This is what makes the def and the Dockerfile incapable of disagreeing: they
    both call install.sh, and it reads one line out of one file.
    """
    txt = (ROOT / "container" / "install.sh").read_text()
    assert "HELIX_PYPROJECT" in txt and "pimm-data" in txt, (
        "container/install.sh no longer derives the pimm-data revision from "
        "pyproject.toml. If the revision is hardcoded again, the image and a dev "
        "install can name different ones, and the one that is wrong does not "
        "degrade -- it fails to import.")


def test_no_build_file_hardcodes_a_revision():
    """A second copy is how the two sides drifted before.

    An explicit PIMM_DATA_REV is still honoured as an override at build time; what
    must not come back is a 40-char revision committed into a build file, because
    then bumping pyproject.toml silently stops changing the image.
    """
    for rel in _BUILD_FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        found = _REV.findall(p.read_text())
        assert not found, (
            f"{rel} hardcodes revision(s) {found} instead of reading "
            f"pyproject.toml. Bumping the pin would then leave this file behind.")


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


def test_exactly_one_revision_is_named_in_the_repo():
    """One pin, in one file. Anything else is a copy waiting to go stale."""
    revs = {}
    for rel in ("pyproject.toml",) + _BUILD_FILES:
        p = ROOT / rel
        if p.exists():
            for r in _REV.findall(p.read_text()):
                revs.setdefault(r, []).append(rel)
    assert list(revs) == [_pyproject_rev()], (
        f"expected pyproject.toml to be the only file naming a pimm-data "
        f"revision, found {({r: f for r, f in revs.items()})}")


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
