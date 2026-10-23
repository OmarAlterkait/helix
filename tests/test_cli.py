"""Shipped-script smoke tests: no corpus build script passes the retired --backend flag."""
from __future__ import annotations


def test_production_scripts_pass_no_backend():
    """The DSP has one backend (torch) and the builder no --backend flag: a shipped script
    still passing it would fail at argument parsing, in an 800-job production build."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for name in ("build_coeff_corpus.py", "submit_coeff_corpus.sh", "calibrate_norm_sigma.sh"):
        text = (root / "scripts" / name).read_text()
        used = [ln for ln in text.splitlines() if "--backend" in ln and not ln.strip().startswith("#")]
        assert not used, f"{name}: {used}"
