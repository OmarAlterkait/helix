"""CLI smoke tests for `helix-tpc` — the --to-coeffs corpus path end to end."""
from __future__ import annotations

import sys

import numpy as np
import h5py

from helix.tpc import run as R

N_TICKS = 64
PLANES = ("U", "V", "Y")
N_WIRES = {"U": 40, "V": 40, "Y": 30}
PEDESTAL = {"U": 1843, "V": 1843, "Y": 410}


def _write_sensor(path, n_events=2, n_volumes=2, seed=0):
    """Minimal current-schema sensor shard (delta-encoded COO)."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        cfg = f.create_group("config")
        cfg.attrs["num_time_steps"] = N_TICKS
        cfg.attrs["n_volumes"] = n_volumes
        cfg.attrs["readout_type"] = "wire"
        cfg["num_wires"] = np.array([[N_WIRES[p] for p in PLANES]] * n_volumes, np.int32)
        cfg["pedestals"] = np.array([[PEDESTAL[p] for p in PLANES]] * n_volumes, np.int32)
        for e in range(n_events):
            evt = f.create_group(f"event_{e:03d}")
            for v in range(n_volumes):
                vol = evt.create_group(f"volume_{v}")
                for p in PLANES:
                    nw = N_WIRES[p]
                    n = 40
                    flat = np.sort(rng.choice(nw * N_TICKS, size=n, replace=False))
                    wire, time = np.divmod(flat, N_TICKS)
                    values = rng.integers(3, 300, size=n)
                    g = vol.create_group(p)
                    g["delta_wire"] = np.diff(wire, prepend=wire[0]).astype(np.int16)
                    g["delta_time"] = np.concatenate([[0], np.diff(time)]).astype(np.int16)
                    g["values"] = (values + PEDESTAL[p]).astype(np.uint16)
                    g.attrs["wire_start"] = int(wire[0])
                    g.attrs["time_start"] = int(time[0])
                    g.attrs["pedestal"] = PEDESTAL[p]


def test_cli_to_coeffs(tmp_path, monkeypatch):
    inp = tmp_path / "sim_wire_sensor_0000.h5"
    out = tmp_path / "sim_wire_coeff_0000.h5"
    _write_sensor(inp)
    monkeypatch.setattr(sys, "argv", [
        "helix-tpc", "--input", str(inp), "--output", str(out),
        "--to-coeffs", "--removal", "gate", "--backend", "numpy"])
    R.main()

    from helix.core.coeff_io import n_events, read_coeff_event
    assert n_events(out) == 2
    ce = read_coeff_event(out, 0)
    ce.basis.validate()
    assert ce.n_coeff > 0
    assert set(ce.gids.tolist()) == set(range(6))       # 2 volumes × 3 planes → gids 0..5
    assert ce.run == inp.parent.name and ce.source_file == inp.name


def test_cli_legacy_path(tmp_path, monkeypatch):
    inp = tmp_path / "sensor.h5"
    out = tmp_path / "processed.h5"
    _write_sensor(inp, n_events=1)
    monkeypatch.setattr(sys, "argv", [
        "helix-tpc", "--input", str(inp), "--output", str(out), "--removal", "none"])
    R.main()
    with h5py.File(out, "r") as f:
        assert "event_000" in f                          # legacy per-event group written
