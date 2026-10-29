"""helix.tpc.io and the simulation source (scripts/build_sim.py) on a JAXTPC wire campaign.

Synthetic tasks in the campaign's layout and format (tests/_sim_fixture.py, format 2: entry_NNNNNN,
/dprod/events, campaign event ids), written uncompressed; production files additionally need
hdf5plugin, which io imports best-effort. Then the real campaign task, site-gated.
"""
import os
import runpy

import h5py
import numpy as np
import pytest

from _sim_fixture import CAMPAIGN, N_TICKS, N_WIRES, PEDESTAL, PLANES, write_geometry, write_task
from helix.tpc import io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source():
    return runpy.run_path(os.path.join(ROOT, "scripts", "build_sim.py"))["SimSource"]


def test_layout_from_config(tmp_path):
    t = write_task(str(tmp_path))
    lay = io.sensor_layout(t["sensor"])
    assert lay.num_time_steps == N_TICKS and lay.campaign == CAMPAIGN
    assert lay.plane_labels == tuple(f"volume_{v}_{p}" for v in range(2) for p in PLANES)
    assert lay.pedestals == PEDESTAL
    assert [lay.n_wires(f"volume_1_{p}") for p in PLANES] == [N_WIRES[p] for p in PLANES]
    assert [lay.plane_gid(lab) for lab in lay.plane_labels] == list(range(6))


def test_event_roundtrip_with_an_empty_volume(tmp_path):
    """Every plane decodes to its pedestal-subtracted truth at the config's shape; a volume with
    nothing deposited (empty plane groups, no datasets, no attrs) is zero hits, and its neighbour
    keeps its charge."""
    t = write_task(str(tmp_path), n_events=3, empty={(1, 0)})
    lay = io.sensor_layout(t["sensor"])
    for pos in range(3):
        planes = io.read_event(t["sensor"], pos, lay)
        assert set(planes) == set(lay.plane_labels)
        for lab, img in planes.items():
            np.testing.assert_array_equal(img, t["truth"][(pos, lab)])
    with h5py.File(t["sensor"], "r") as f:
        coo = io.read_event_coo(f[io.entry_key(1)], lay)
    assert all(coo[f"volume_0_{p}"][0].size == 0 for p in PLANES)
    assert all(coo[f"volume_1_{p}"][0].size > 0 and coo[f"volume_1_{p}"][2].min() >= 3 for p in PLANES)


def test_entries_are_checked_against_the_campaign_identity(tmp_path):
    t = write_task(str(tmp_path), first_job=85, n_events=3)
    with h5py.File(t["sensor"], "r") as f:
        ids = io.event_ids(f)
        assert ids.tolist() == [17000, 17001, 17002]
        assert io.entry(f, 2, 17002).name == "/entry_000002"
        with pytest.raises(ValueError, match="dprod"):
            io.entry(f, 2, 17001)


def test_format_1_and_pixel_files_are_refused(tmp_path):
    p = tmp_path / "old.h5"
    with h5py.File(p, "w") as f:
        f.create_group("config").attrs["num_time_steps"] = 4321
        f.create_group("event_000")
    with pytest.raises(ValueError, match="format-2"):
        io.sensor_layout(p)
    t = write_task(str(tmp_path))
    with h5py.File(t["sensor"], "a") as f:
        f["config"].attrs["readout_type"] = "pixel"
    with pytest.raises(ValueError, match="pixel"):
        io.sensor_layout(t["sensor"])


def test_task_paths(tmp_path):
    t = write_task(str(tmp_path), group="j001000", first_job=1085)
    assert io.task_files(t["sensor"]) == {k: t[k] for k in ("sensor", "step", "hits")}
    assert io.sensor_path(str(tmp_path), "j001000", os.path.basename(t["sensor"])) == t["sensor"]


# ------------------------------------------------------------------------------- the build source
def test_sim_source_events_carry_the_campaign_identity(tmp_path):
    """The source's events: ident (group, file, campaign event id), the noise seed from it, every
    plane present (the empty volume's as zeros), the clean planes the file's own."""
    pytest.importorskip("pimm_data")
    Sim = _source()
    t = write_task(str(tmp_path), first_job=85, n_events=3, empty={(1, 1)})
    src = Sim([t["sensor"]], "cpu", geom=write_geometry(tmp_path / "geom.json"))
    src.group_size = 8
    assert len(src) == 3 and src.run == "j000000" and src.campaign == CAMPAIGN
    evs = list(src.events(range(3), 1))
    name = os.path.basename(t["sensor"])
    assert [e["ident"]["event"] for e in evs] == [17000, 17001, 17002]
    assert {e["ident"]["run"] for e in evs} == {"j000000"} and {e["ident"]["source_file"] for e in evs} == {name}
    assert [e["ident"]["noise_seed"] for e in evs] == [src.seed(t["sensor"], ev) for ev in (17000, 17001, 17002)]
    assert len({e["ident"]["noise_seed"] for e in evs}) == 3
    lay = io.sensor_layout(t["sensor"])
    for pos, e in enumerate(evs):
        assert set(e["planes"]) == set(e["clean"]) == set(range(6))
        for lab in lay.plane_labels:
            g = lay.plane_gid(lab)
            np.testing.assert_array_equal(e["clean"][g].numpy(), t["truth"][(pos, lab)])
            assert e["planes"][g].shape == (lay.n_wires(lab), N_TICKS)
            assert float(e["planes"][g].std()) > 0                 # noise is there, empty plane or not
    assert all(not e.any() for g, e in evs[1]["clean"].items() if g >= 3)
    attrs = src.shard_attrs()["provenance"]
    assert attrs["sim"] == dict(campaign=CAMPAIGN, jaxtpc=["abc"])


def test_sim_source_refuses_mixed_groups_and_a_wrong_geometry(tmp_path):
    Sim = _source()
    a = write_task(str(tmp_path), group="j000000", first_job=0)
    b = write_task(str(tmp_path), group="j001000", first_job=1000)
    geom = write_geometry(tmp_path / "geom.json")
    with pytest.raises(SystemExit, match="groups"):
        Sim([a["sensor"], b["sensor"]], None, geom=geom)
    with pytest.raises(SystemExit, match="plane"):
        Sim([a["sensor"]], None)                   # the packaged geometry is the full-size detector


def test_plan_numbers_shards_by_campaign_event(tmp_path):
    """One task per sensor file; shard k holds campaign events k*eps .. (k+1)*eps-1, so a task of
    jobs 85-89 (events 17000-17999) writes shards 85-89 at 200 per shard."""
    from helix.build import plan
    Sim = _source()
    root = tmp_path / "sim"
    t0 = write_task(str(root), first_job=0, n_events=1000, n_tracks=1, n_volumes=1)
    t1 = write_task(str(root), first_job=85, n_events=1000, n_tracks=1, n_volumes=1)
    lines = plan(Sim, "OUT", [str(root / "j000000")], events_per_task=500, events_per_shard=200)
    assert lines == [f"OUT/j000000/sim_wire 0 1000 200 0 {t0['sensor']}",
                     f"OUT/j000000/sim_wire 0 1000 200 85 {t1['sensor']}"]
    with pytest.raises(ValueError, match="multiple of 300"):
        Sim.file_plan(t1["sensor"], 300)


# ------------------------------------------------------------------------------ the real campaign
from _paths import sim_file                                     # noqa: E402

REAL = sim_file("sensor")


@pytest.mark.skipif(not os.path.exists(REAL), reason="campaign task not reachable")
def test_real_task_reads():
    """The campaign's task j000085-000089: 1000 events, ids 17000-17999, the detector's layout;
    entry 202 has nothing in volume 1."""
    lay = io.sensor_layout(REAL)
    assert lay.num_time_steps == 4321 and lay.num_wires.tolist() == [[1969, 1969, 1443]] * 2
    assert lay.pedestals == {"U": 1843, "V": 1843, "Y": 410}
    assert lay.campaign == "doraemon_2026_1M_lartpc"
    with h5py.File(REAL, "r") as f:
        ids = io.event_ids(f)
        assert ids.tolist() == list(range(17000, 18000))
        coo = io.read_event_coo(io.entry(f, 202, 17202), lay)
    assert all(coo[f"volume_1_{p}"][0].size == 0 for p in PLANES)
    for p in PLANES:
        w, t, v = coo[f"volume_0_{p}"]
        assert w.size > 1000 and 0 <= w.min() and w.max() < lay.n_wires(f"volume_0_{p}")
        assert 0 <= t.min() and t.max() < 4321 and np.abs(v).min() >= 1.0     # 2-ADC zero suppression
