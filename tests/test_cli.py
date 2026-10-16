"""Shipped-script smoke tests: the corpus build scripts agree on the backend."""
from __future__ import annotations


def test_production_scripts_use_the_intended_backend():
    """The shipped scripts must not silently override the build default.

    This is a real failure that happened: --backend was defaulted to torch in
    build_coeff_corpus.py, but submit_coeff_corpus.sh and calibrate_norm_sigma.sh
    each passed a different --backend EXPLICITLY, so the 800-job production build
    would still have run the other backend. A default is not a decision if every
    caller overrides it.
    """
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent

    # the builder's own default
    src = (root / "scripts" / "build_coeff_corpus.py").read_text()
    m = re.search(r'"--backend",\s*choices=(\[[^\]]*\]),\s*default="([a-z]+)"', src)
    assert m, "could not find the --backend default in build_coeff_corpus.py"
    default = m.group(2)
    assert default == "torch", f"build default is {default!r}, expected 'torch'"
    # the builder runs the TPC chain, which is numpy or torch: no other backend is left
    offered = set(re.findall(r'"([a-z]+)"', m.group(1)))
    assert offered <= {"numpy", "torch"}, f"build_coeff_corpus.py offers {sorted(offered)}"

    # and every shipped script must agree with it on its EXECUTABLE lines
    for name in ("submit_coeff_corpus.sh", "calibrate_norm_sigma.sh"):
        text = (root / "scripts" / name).read_text()
        used = [ln.strip() for ln in text.splitlines()
                if "--backend" in ln and not ln.strip().startswith("#")]
        assert used, f"{name}: no explicit --backend line found"
        for ln in used:
            got = re.search(r"--backend\s+(\S+)", ln).group(1)
            assert got == default, f"{name} passes --backend {got}, default is {default}"
