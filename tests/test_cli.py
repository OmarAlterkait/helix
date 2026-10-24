"""Shipped scripts call the one build entry point with arguments it accepts."""
from __future__ import annotations

from pathlib import Path


def test_no_script_names_a_retired_builder():
    """``python -m helix.build {sim,sbnd,goop}`` replaced build_coeff_corpus.py, helix.tpc.sbnd_run
    and helix.optical.goop_run (and their --backend / --mode / --kgate flags): a shipped script
    still naming one fails at its first task."""
    root = Path(__file__).resolve().parent.parent
    retired = ("build_coeff_corpus.py", "helix.tpc.sbnd_run", "helix.optical.goop_run", "--backend", "--mode serial")
    for p in sorted((root / "scripts").glob("*.sh")):
        code = [ln for ln in p.read_text().splitlines() if not ln.strip().startswith("#")]
        used = [ln for ln in code if any(r in ln for r in retired)]
        assert not used, f"{p.name}: {used}"
