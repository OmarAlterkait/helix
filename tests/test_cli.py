"""Shipped scripts build through a source's script and helix.build's one command line."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_no_script_names_a_retired_builder():
    """``scripts/build_<source>.py`` (each ending in ``helix.build.main``) replaced
    build_coeff_corpus.py, helix.tpc.sbnd_run, helix.optical.goop_run and ``python -m helix.build
    SOURCE`` (and the --backend / --mode / --kgate flags): a shipped script still naming one fails
    at its first task."""
    retired = ("build_coeff_corpus.py", "helix.tpc.sbnd_run", "helix.optical.goop_run", "-m helix.build",
               "--backend", "--mode serial")
    for p in sorted((ROOT / "scripts").glob("*.sh")):
        code = [ln for ln in p.read_text().splitlines() if not ln.strip().startswith("#")]
        used = [ln for ln in code if any(r in ln for r in retired)]
        assert not used, f"{p.name}: {used}"


@pytest.mark.parametrize("name", ["sim", "sbnd", "goop"])
def test_every_source_script_has_what_build_needs(name):
    """What :func:`helix.build.main` and :func:`helix.build.plan` read of a source class."""
    sys.path.insert(0, str(ROOT / "scripts"))
    mod = __import__(f"build_{name}")
    Source = next(v for k, v in vars(mod).items() if k.endswith("Source") and isinstance(v, type))
    for attr in ("systems", "dataset", "pattern", "events", "shard_attrs", "__len__"):
        assert hasattr(Source, attr), f"build_{name}.{Source.__name__} has no {attr}"
    assert hasattr(Source, "chain_config") or hasattr(Source, "optical_chain")
    assert not hasattr(Source, "chain_config") or hasattr(Source, "bind")
