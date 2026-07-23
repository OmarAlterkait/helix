"""Golden-output capture/check for the helix consolidation program (Phase 0.2).

Purpose: every consolidation step must reproduce these outputs BIT-IDENTICALLY
before it lands (the check-as-we-go contract). Capture once on a reference
node, re-run with --check after every change.

Goldens:
  tpc_rows   : star_tpc.prep_tpc_rows for pinned events — exercises the full
               GPU production pipeline (pimm-data extract -> Densify ->
               AddNoise -> Digitize -> torch DWT -> smart gate -> per-band
               threshold -> row emit + sigma table).
  optical    : helix.optical.process_event on a pinned event of the goop
               east/west file, numpy backend — exercises core sparsify +
               optical io/pipeline/metrics.
  suite      : the helix pytest suite must be green (recorded count).

Determinism notes: TPC rows are bit-stable on ONE device class (torch CUDA
RNG + cuFFT; Densify's index_add_ is collision-free on unique COO). The
golden file records the device name; compare only on a matching device.

Usage (cwd = research/coeff_foundation_model):
  python ../goldens/capture.py            # capture -> ../goldens/goldens.json
  python ../goldens/capture.py --check    # recompute and compare
"""
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))   # helix repo root (package not pip-installed)
GOLDEN_PATH = os.path.join(HERE, "goldens.json")
TPC_EVENTS = (2, 7)          # pinned; NOT the sigma-calibration events (0, 1)
OPTICAL_FILE = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
OPTICAL_EVENT = "event_003"


def _h(arr):
    import numpy as np
    a = np.ascontiguousarray(arr)
    return f"{a.dtype}|{a.shape}|{hashlib.sha256(a.tobytes()).hexdigest()[:16]}"


def golden_tpc():
    sys.path.insert(0, os.path.join(HERE, "..", "coeff_foundation_model"))
    import star_tpc as stp
    out = {}
    for ev in TPC_EVENTS:
        rows = stp.prep_tpc_rows(ev)
        out[f"ev{ev}"] = {k: _h(v) for k, v in sorted(rows.items())}
    P = stp._pipeline()
    out["sigma_tab"] = {str(g): [round(float(x), 10) for x in v]
                       for g, v in sorted(P["sigma_tab"].items())}
    return out


def golden_optical():
    from helix.core import backend
    from helix.optical import config_from_file, process_event
    backend.set_backend("numpy")
    cfg = config_from_file(OPTICAL_FILE)
    r = process_event(OPTICAL_FILE, OPTICAL_EVENT, cfg)
    return {
        "n_kept": r.sparse.n_kept, "n_total": r.sparse.n_total,
        "bands": [_h(c) for c in r.sparse.coeffs],
        "metrics": {k: round(float(v), 10) for k, v in sorted(r.metrics.items())},
        "lengths": _h(r.lengths), "pmt_id": _h(r.pmt_id),
    }


def golden_suite():
    helix_root = os.path.join(HERE, "..", "..")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "tests"], cwd=helix_root, capture_output=True, text=True)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-200:]
    tail = re.sub(r" in [0-9.]+s", "", tail)          # strip wall time (non-deterministic)
    return {"exit": p.returncode, "summary": tail}


def main():
    import torch
    check = "--check" in sys.argv
    got = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "suite": golden_suite(),
        "optical": golden_optical(),
        "tpc": golden_tpc(),
    }
    assert got["suite"]["exit"] == 0, f"suite not green: {got['suite']}"
    if check:
        with open(GOLDEN_PATH) as f:
            ref = json.load(f)
        if ref["device"] != got["device"]:
            print(f"SKIP tpc compare: device {got['device']!r} != golden {ref['device']!r}")
            ref = {k: v for k, v in ref.items() if k != "tpc"}
            got = {k: v for k, v in got.items() if k != "tpc"}
        mismatches = [k for k in ref if ref[k] != got.get(k)]
        if mismatches:
            for k in mismatches:
                print(f"MISMATCH {k}:\n  golden: {ref[k]}\n  got:    {got[k]}")
            sys.exit(1)
        print(f"GOLDENS OK ({', '.join(k for k in ref if k != 'device')})")
    else:
        with open(GOLDEN_PATH, "w") as f:
            json.dump(got, f, indent=1, sort_keys=True)
        print(f"captured -> {GOLDEN_PATH}")
        print(json.dumps({k: v for k, v in got.items() if k in ("suite", "device")}, indent=1))


if __name__ == "__main__":
    main()
