"""Tests for helix.tpc.io against BOTH sensor schemas.

Current schema (doraemon productions): /config group + tables,
event_NNN/volume_V/<P> with delta-encoded COO + uint16 digitized values.
Legacy flat schema: event_N/<plane>/{wire,time,values} raw COO.

Fixtures are written uncompressed (the reader is agnostic to HDF5 filters;
production files additionally need hdf5plugin, imported best-effort by io).
"""
import numpy as np
import h5py
import pytest

from helix.tpc.io import (config_from_file, count_events, read_sensor_event,
                          read_sensor_plane)

N_TICKS = 64
PLANES = ("U", "V", "Y")
N_WIRES = {"U": 40, "V": 40, "Y": 30}
PEDESTAL = {"U": 1843, "V": 1843, "Y": 410}


def _synth_plane(rng, n_wires):
    """Sparse synthetic plane: (wire, time, adc>0) with sorted-unique COO."""
    n = 25
    flat = rng.choice(n_wires * N_TICKS, size=n, replace=False)
    flat.sort()
    wire, time = np.divmod(flat, N_TICKS)
    values = rng.integers(3, 300, size=n)
    return wire.astype(np.int64), time.astype(np.int64), values.astype(np.int64)


def _write_current(path, n_events=2, n_volumes=2, seed=0):
    """Write the current volume-nested delta-encoded schema."""
    rng = np.random.default_rng(seed)
    truth = {}
    with h5py.File(path, "w") as f:
        cfg = f.create_group("config")
        cfg.attrs["num_time_steps"] = N_TICKS
        cfg.attrs["n_volumes"] = n_volumes
        cfg.attrs["readout_type"] = "wire"
        cfg["num_wires"] = np.array([[N_WIRES[p] for p in PLANES]] * n_volumes,
                                    np.int32)
        cfg["pedestals"] = np.array([[PEDESTAL[p] for p in PLANES]] * n_volumes,
                                    np.int32)
        for e in range(n_events):
            evt = f.create_group(f"event_{e:03d}")
            for v in range(n_volumes):
                vol = evt.create_group(f"volume_{v}")
                for p in PLANES:
                    wire, time, values = _synth_plane(rng, N_WIRES[p])
                    g = vol.create_group(p)
                    # delta-encode exactly as the producer does
                    g["delta_wire"] = np.diff(wire, prepend=wire[0]).astype(np.int16)
                    g["delta_time"] = np.concatenate(
                        [[0], np.diff(time)]).astype(np.int16)
                    g["values"] = (values + PEDESTAL[p]).astype(np.uint16)
                    g.attrs["wire_start"] = int(wire[0])
                    g.attrs["time_start"] = int(time[0])
                    g.attrs["pedestal"] = PEDESTAL[p]
                    g.attrs["n_pixels"] = len(wire)
                    dense = np.zeros((N_WIRES[p], N_TICKS), np.float32)
                    dense[wire, time] = values          # pedestal-subtracted truth
                    truth[(e, f"volume_{v}_{p}")] = dense
    return truth


def _write_legacy(path, n_events=2, seed=1):
    rng = np.random.default_rng(seed)
    truth = {}
    with h5py.File(path, "w") as f:
        for e in range(n_events):
            evt = f.create_group(f"event_{e}")           # legacy: unpadded keys
            evt.attrs["num_time_steps"] = N_TICKS
            for p in PLANES:
                wire, time, values = _synth_plane(rng, N_WIRES[p])
                g = evt.create_group(p)
                g["wire"] = wire.astype(np.int32)
                g["time"] = time.astype(np.int32)
                g["values"] = (values + PEDESTAL[p]).astype(np.float32)
                g.attrs["n_wires"] = N_WIRES[p]
                g.attrs["pedestal"] = PEDESTAL[p]
                dense = np.zeros((N_WIRES[p], N_TICKS), np.float32)
                dense[wire, time] = values
                truth[(e, p)] = dense
    return truth


# ── current schema ──────────────────────────────────────────────────────────

def test_current_config_from_file(tmp_path):
    path = tmp_path / "cur.h5"
    _write_current(path)
    cfg = config_from_file(path)
    assert cfg.num_time_steps == N_TICKS
    assert cfg.plane_labels == tuple(
        f"volume_{v}_{p}" for v in range(2) for p in PLANES)
    assert cfg.pedestals == PEDESTAL


def test_current_roundtrip_event(tmp_path):
    path = tmp_path / "cur.h5"
    truth = _write_current(path)
    cfg = config_from_file(path)
    for e in range(2):
        planes = read_sensor_event(path, e, cfg)
        assert set(planes) == set(cfg.plane_labels)
        for label, img in planes.items():
            ref = truth[(e, label)]
            assert img.shape == ref.shape          # n_wires from /config/num_wires
            np.testing.assert_array_equal(img, ref)


def test_current_plane_n_wires_from_config(tmp_path):
    """n_wires must come from /config, not max(wire)+1 (fixed grid contract)."""
    path = tmp_path / "cur.h5"
    _write_current(path)
    img = read_sensor_plane(path, 0, "volume_0_Y")
    assert img.shape == (N_WIRES["Y"], N_TICKS)


def test_pixel_file_rejected(tmp_path):
    path = tmp_path / "pix.h5"
    with h5py.File(path, "w") as f:
        f.create_group("config").attrs["readout_type"] = "pixel"
        f.create_group("event_000")
    with pytest.raises(ValueError, match="pixel"):
        config_from_file(path)


# ── legacy flat schema ──────────────────────────────────────────────────────

def test_legacy_roundtrip(tmp_path):
    path = tmp_path / "leg.h5"
    truth = _write_legacy(path)
    cfg = config_from_file(path)
    assert cfg.num_time_steps == N_TICKS
    assert set(cfg.plane_labels) == set(PLANES)
    planes = read_sensor_event(path, 0, cfg)
    for label, img in planes.items():
        np.testing.assert_array_equal(img, truth[(0, label)])


def test_count_events(tmp_path):
    path = tmp_path / "cur.h5"
    _write_current(path, n_events=3)
    assert count_events(path) == 3


# ── real production shard (site-gated) ──────────────────────────────────────

from _paths import sensor_shard                                # noqa: E402

REAL = sensor_shard("run_0027575715", "sim_wire_sensor_0000.h5")


@pytest.mark.skipif(not __import__("os").path.exists(REAL),
                    reason="production shard not reachable")
def test_real_shard_reads():
    cfg = config_from_file(REAL)
    assert cfg.num_time_steps == 4321
    assert "volume_0_U" in cfg.plane_labels
    img = read_sensor_plane(REAL, 0, "volume_0_U")
    assert img.shape == (1969, 4321)               # /config/num_wires, not max+1
    assert img.dtype == np.float32
    assert (img != 0).sum() > 1000                 # sparse but populated
    # digitized-with-2-ADC-threshold data: nonzero magnitudes start at >=2-ish
    nz = img[img != 0]
    assert np.abs(nz).min() >= 1.0


# ---- empty plane groups (no charge in a volume) ---------------------------

def _write_with_empty_volume(path, empty_event=1, empty_vol=0, n_events=3,
                             n_volumes=2, seed=7):
    """The current schema, but one event has one volume written as three EMPTY
    plane groups — no datasets, no attrs.

    This is what the doraemon producer emits when an event deposits no charge in
    that TPC volume, and it occurs in real production data: 12 of 100 source
    files in run_0027575715 contain exactly one such event, which is why 12
    corpus shards failed to build."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        cfg = f.create_group("config")
        cfg.attrs["num_time_steps"] = N_TICKS
        cfg.attrs["n_volumes"] = n_volumes
        cfg.attrs["readout_type"] = "wire"
        cfg["num_wires"] = np.array([[N_WIRES[p] for p in PLANES]] * n_volumes,
                                    np.int32)
        cfg["pedestals"] = np.array([[PEDESTAL[p] for p in PLANES]] * n_volumes,
                                    np.int32)
        for e in range(n_events):
            evt = f.create_group(f"event_{e:03d}")
            for v in range(n_volumes):
                vol = evt.create_group(f"volume_{v}")
                for p in PLANES:
                    g = vol.create_group(p)
                    if e == empty_event and v == empty_vol:
                        continue                        # empty: no charge here
                    wire, time, values = _synth_plane(rng, N_WIRES[p])
                    g["delta_wire"] = np.diff(wire, prepend=wire[0]).astype(np.int16)
                    g["delta_time"] = np.concatenate([[0], np.diff(time)]).astype(np.int16)
                    g["values"] = (values + PEDESTAL[p]).astype(np.uint16)
                    g.attrs["wire_start"] = int(wire[0])
                    g.attrs["time_start"] = int(time[0])
                    g.attrs["pedestal"] = PEDESTAL[p]


def test_empty_plane_reads_as_zero_hits(tmp_path):
    """An empty plane group must decode as zero hits at the config's shape, not
    raise. It previously fell through to the legacy branch and died with
    KeyError: 'wire'."""
    from helix.tpc.io import read_sensor_plane_coo
    p = str(tmp_path / "empty_vol.h5")
    _write_with_empty_volume(p)

    for pl in PLANES:
        wire, time, values, nw, nt = read_sensor_plane_coo(p, 1, f"volume_0_{pl}")
        assert wire.size == time.size == values.size == 0
        assert (nw, nt) == (N_WIRES[pl], N_TICKS)       # shape from config
        img = read_sensor_plane(p, 1, f"volume_0_{pl}")
        assert img.shape == (N_WIRES[pl], N_TICKS) and not img.any()


def test_empty_volume_does_not_affect_its_neighbour(tmp_path):
    """The OTHER volume of the same event still carries its charge — an empty
    volume must not be read as an empty event."""
    from helix.tpc.io import read_sensor_plane_coo
    p = str(tmp_path / "empty_vol.h5")
    _write_with_empty_volume(p)
    wire, _, values, _, _ = read_sensor_plane_coo(p, 1, "volume_1_U")
    assert wire.size > 0 and values.any()


def test_empty_plane_survives_whole_event_read(tmp_path):
    """read_sensor_event_coo over the affected event must return every plane,
    the empty ones included, so downstream plane bookkeeping stays aligned."""
    from helix.tpc.io import read_sensor_event_coo, config_from_file
    p = str(tmp_path / "empty_vol.h5")
    _write_with_empty_volume(p)
    cfg = config_from_file(p)
    out = read_sensor_event_coo(p, 1, cfg)
    assert set(out) == set(cfg.plane_labels)
    assert all(out[f"volume_0_{pl}"][0].size == 0 for pl in PLANES)
    assert any(out[f"volume_1_{pl}"][0].size > 0 for pl in PLANES)


# ---- non-contiguous event ids ---------------------------------------------

def test_list_events_reports_actual_ids(tmp_path):
    """Event ids are not guaranteed to be 0..n-1. `sim_wire_sensor_0065.h5` of
    run_0027575715 holds 199 events spanning 0..199 with 167 absent, while its
    own config.n_events attr still says 200 — so anything deriving indices from
    a COUNT asks for a missing id and drops a real one off the tail."""
    from helix.tpc.io import list_events, count_events
    p = str(tmp_path / "holey.h5")
    truth = _write_current(p, n_events=5)
    with h5py.File(p, "a") as f:
        del f["event_002"]                       # punch a hole in the middle

    ids = list_events(p)
    assert ids == (0, 1, 3, 4)                   # the real ids, not range(4)
    assert count_events(p) == 4                  # count alone would say 0..3
    assert max(ids) not in range(count_events(p))  # id 4 is past the count
    assert truth                                  # fixture actually wrote data
