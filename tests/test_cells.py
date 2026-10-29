"""helix.data.cells: the simulation truth on the fine grid, one file per coefficient shard.

A synthetic campaign task (tests/_sim_fixture.py). Pinned: the stored record is the pixels
reduced (per cell, per (cell, track), per (plane, track), per track, and each cell's dominant
group with its 3D centroid), computed here independently; it is complete for the evaluation's
windows (a grid-aligned box holds per track exactly its pixels' charge); the build is resumable
and atomic; the dataset serves it identity-checked.
"""
import os
import sys

import h5py
import numpy as np
import pytest

pytest.importorskip("hdf5plugin")

from _coeff_fixture import write_coeff_shard  # noqa: E402
from _sim_fixture import write_task  # noqa: E402
from helix.data import cells as C  # noqa: E402
from helix.eval import windows as Wn  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN, N_EV, FIRST_JOB = "j000000", 3, 85
EV0 = FIRST_JOB * 200                 # the task's first campaign event id
BASIS = dict(wavelet="db2", dwt_level=2, dwt_mode="periodization", n_ticks_raw=32, pad=0, sigma_norm=2.6,
             removal_json="{}", threshold_json="{}")


SOURCE = "sim_wire_sensor_jaxtpc_wire_j000085-000089.h5"


def _coeff_shard(path, events, run=RUN, source=SOURCE):
    rng = np.random.default_rng(1)
    evs = [dict(band=np.zeros(4, np.uint8), plane_gid=np.zeros(4, np.int32), wire=rng.integers(0, 4, 4),
                tau=rng.integers(0, 8, 4), value=rng.standard_normal(4).astype(np.float32),
                sigma_threshold=rng.random((1, 3)).astype(np.float32), run=run, source_file=source, event=e)
           for e in events]
    write_coeff_shard(path, evs, band_lengths=[8, 8, 16], gids=[0], n_wires=[4], basis_attrs=BASIS,
                      dataset_name="sim_wire")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("cells")
    hits = write_task(str(d / "sim"), RUN, FIRST_JOB, N_EV)["hits"]
    run_dir = d / "corpus" / RUN
    run_dir.mkdir(parents=True)
    _coeff_shard(run_dir / "sim_wire_coeff_0000.h5", [EV0 + i for i in range(N_EV)])
    out, msg = C.build_shard(str(run_dir / "sim_wire_coeff_0000.h5"), str(d / "sim"))
    return dict(dir=d, run_dir=run_dir, hits=hits, cells=out, msg=msg)


def _pixels(hits, ev):
    """Per sample: g, w, t, q, (volume, track id), event-global group and its (y, z) centroid,
    independently of helix.data.cells."""
    G, W, T, Q, V, TID, GRP, YZ = [], [], [], [], [], [], [], []
    with h5py.File(hits, "r") as f, h5py.File(str(hits).replace("hits", "step"), "r") as fs:
        gbase = 0
        for vk in sorted(x for x in f[f"entry_{ev:06d}"] if x.startswith("volume_")):
            hv, sv, vi = f[f"entry_{ev:06d}"][vk], fs[f"entry_{ev:06d}"][vk], int(vk.split("_")[1])
            g2t, d2g = hv["group_to_track"][:], hv["deposit_to_group"][:]
            origin = np.array([sv.attrs[f"pos_origin_{a}"] for a in "xyz"])
            pos = origin + sv["positions"][:] * float(sv.attrs["pos_step_mm"])
            ch = sv["charge"][:].astype(np.float64)
            cen = {}
            for grp in np.unique(d2g):
                m = d2g == grp
                cen[int(grp)] = (ch[m] @ pos[m]) / ch[m].sum()
            for pi, pl in enumerate(C.PLANES):
                s = C.decode_hits_plane(hv[pl])
                G.append(np.full(len(s["q"]), vi * 3 + pi)); W.append(s["wire"]); T.append(s["tick"])
                Q.append(s["q"]); V.append(np.full(len(s["q"]), vi)); TID.append(g2t[s["group"]])
                GRP.append(gbase + s["group"])
                YZ.append(np.array([cen.get(int(x), (np.nan,) * 3)[1:] for x in s["group"]]))
            gbase += len(g2t)
    return [np.concatenate(x) for x in (G, W, T, Q, V, TID, GRP, YZ)]


def test_record_is_the_pixels_reduced(corpus):
    """Per cell charge, per (cell, track) charge, per (plane, track) pixel centroid, per track step energy."""
    G, W, T, Q, V, TID, _, _ = _pixels(corpus["hits"], 1)
    rec, _ = C.open_event(corpus["cells"], 1)
    trk = list(zip(rec["trk_volume"].tolist(), rec["trk_id"].tolist()))
    assert trk == sorted(set(trk))
    cell = C.fkey(G, W // C.FW, T // C.FT)
    t_of = {(v, t): i for i, (v, t) in enumerate(trk)}
    ti = np.array([t_of[(v, t)] for v, t in zip(V.tolist(), TID.tolist())])
    want = {}
    for c, t, q in zip(cell.tolist(), ti.tolist(), Q.tolist()):
        want[(c, t)] = want.get((c, t), 0.0) + q
    got = {(int(rec["key"][c]), int(t)): float(q) for c, t, q in zip(rec["ct_cell"], rec["ct_track"], rec["ct_q"])}
    assert got.keys() == want.keys()
    np.testing.assert_allclose([got[k] for k in want], list(want.values()), rtol=1e-6)
    assert len({k[0] for k in want}) == len(rec["key"]) < len(want)          # some cells hold two tracks
    uk, inv = np.unique(cell, return_inverse=True)
    np.testing.assert_array_equal(rec["key"], uk)
    np.testing.assert_allclose(rec["q"], np.bincount(inv.ravel(), weights=Q), rtol=1e-6)
    for g, t, w_, t_ in zip(rec["pt_gid"], rec["pt_track"], rec["pt_wire"], rec["pt_tick"]):
        m = (G == g) & (ti == t)
        assert w_ == pytest.approx((Q[m] * W[m]).sum() / Q[m].sum(), rel=1e-12)
        assert t_ == pytest.approx((Q[m] * T[m]).sum() / Q[m].sum(), rel=1e-12)
    with h5py.File(corpus["hits"], "r") as fh, h5py.File(str(corpus["hits"]).replace("hits", "step"), "r") as fs:
        for i, (v, t) in enumerate(trk):
            d2g = fh[f"entry_000001/volume_{v}/deposit_to_group"][:]
            g2t = fh[f"entry_000001/volume_{v}/group_to_track"][:]
            de = fs[f"entry_000001/volume_{v}/de"][:].astype(np.float64)
            assert rec["trk_energy"][i] == pytest.approx(de[g2t[d2g] == t].sum(), rel=1e-6)


def test_each_cell_records_its_dominant_group_and_its_centroid(corpus):
    """The along-wire probe's label: the largest group's share of the cell, and its deposits'
    charge-weighted (y, z)."""
    G, W, T, Q, _, _, GRP, YZ = _pixels(corpus["hits"], 0)
    rec, _ = C.open_event(corpus["cells"], 0)
    cell = C.fkey(G, W // C.FW, T // C.FT)
    pos = {int(k): i for i, k in enumerate(rec["key"])}
    per = {}
    for c, g, q, yz in zip(cell.tolist(), GRP.tolist(), Q.tolist(), YZ):
        d = per.setdefault(c, {})
        d[g] = (d.get(g, (0.0, yz))[0] + q, yz)
    n_shared = 0
    for c, d in per.items():
        g, (qg, yz) = max(d.items(), key=lambda kv: kv[1][0])
        i = pos[c]
        tot = sum(v[0] for v in d.values())
        assert rec["dom_frac"][i] == pytest.approx(qg / tot, rel=1e-5)
        np.testing.assert_allclose([rec["dom_y"][i], rec["dom_z"][i]], yz, rtol=1e-5)
        n_shared += len(d) > 1
    assert n_shared > 0 and (rec["dom_frac"] <= 1 + 1e-6).all()


def test_grid_aligned_boxes_see_the_same_charge_per_track_as_pixels(corpus):
    """Why cells suffice: a box on the 2 x 16 grid holds, per track, exactly its pixels' charge."""
    G, W, T, Q, V, TID, _, _ = _pixels(corpus["hits"], 0)
    rec, _ = C.open_event(corpus["cells"], 0)
    cg, cw, ct = C.unkey(rec["key"][rec["ct_cell"]])
    t_of = {(v, t): i for i, (v, t) in enumerate(zip(rec["trk_volume"].tolist(), rec["trk_id"].tolist()))}
    ti = np.array([t_of[(v, t)] for v, t in zip(V.tolist(), TID.tolist())])
    rng = np.random.default_rng(0)
    for _ in range(200):
        g, w0, t0 = int(rng.integers(0, 6)), int(rng.integers(0, 450)) * C.FW, int(rng.integers(0, 190)) * C.FT
        pin = (G == g) & (W >= w0) & (W < w0 + Wn.WW) & (T >= t0) & (T < t0 + Wn.WT)
        cin = (cg == g) & (cw * C.FW >= w0) & (cw * C.FW < w0 + Wn.WW) & (ct * C.FT >= t0) & (ct * C.FT < t0 + Wn.WT)
        for t in np.unique(np.concatenate([ti[pin], rec["ct_track"][cin]])):
            assert rec["ct_q"][cin & (rec["ct_track"] == t)].sum() == pytest.approx(Q[pin & (ti == t)].sum(), rel=1e-5)


def test_the_windows_of_a_record(corpus):
    """event_truth on a stored record: every window kind, reproducible, and the windows' truth
    is the record's charge."""
    rec, _ = C.open_event(corpus["cells"], 2)
    rng = np.random.default_rng(3)
    coeff = np.concatenate([rec["key"][::3], C.fkey(rng.integers(0, 6, 400), rng.integers(0, 500, 400),
                                                    rng.integers(0, 200, 400))])
    a = Wn.event_truth(rec, coeff, np.random.default_rng(7))
    b = Wn.event_truth(rec, coeff, np.random.default_rng(7))
    assert {"iso", "bg"} <= {m["kind"] for m in a["meta"]}
    for k in ("mrows", "wkey", "wid", "wq", "mq"):
        np.testing.assert_array_equal(a[k], b[k])
    q = dict(zip(rec["key"].tolist(), rec["q"].tolist()))
    np.testing.assert_allclose(a["wq"], [q.get(k, 0.0) for k in a["wkey"].tolist()])


def test_file_layout_resume_and_atomic_write(corpus):
    assert corpus["msg"].startswith(f"{N_EV} events")
    assert os.path.basename(corpus["cells"]) == "sim_wire_cells_0000.h5"
    assert not any(p.name.endswith(".part") for p in corpus["run_dir"].iterdir())
    with h5py.File(corpus["cells"], "r") as f:
        cfg = f["config"]
        assert cfg.attrs["format"] == C.FORMAT and (cfg.attrs["grid_fw"], cfg.attrs["grid_ft"]) == (C.FW, C.FT)
        import json
        assert json.loads(cfg.attrs["label_json"])["hits_threshold"] == 25.0
        assert C.read_ident(f)["event"].tolist() == [EV0 + i for i in range(N_EV)]
        for g in C.GROUPS:
            assert f[g]["event_offset"].shape == (N_EV + 1,)
    before = os.stat(corpus["cells"]).st_mtime_ns
    assert C.build_shard(str(corpus["run_dir"] / "sim_wire_coeff_0000.h5"), str(corpus["dir"] / "sim"))[1] == "exists"
    assert os.stat(corpus["cells"]).st_mtime_ns == before


def test_reader_refuses_another_format(corpus, tmp_path):
    p = tmp_path / "sim_wire_cells_0000.h5"
    p.write_bytes(open(corpus["cells"], "rb").read())
    with h5py.File(p, "a") as f:
        f["config"].attrs["grid_ft"] = 8
    with pytest.raises(ValueError, match="grid"):
        C.open_event(str(p), 0)
    with h5py.File(p, "a") as f:
        f["config"].attrs["format"] = "helix-cells-1"
    with pytest.raises(ValueError, match="rebuild"):
        C.open_event(str(p), 0)


# ------------------------------------------------------------------------------- dataset modality
def test_dataset_serves_cells_identity_checked(corpus, tmp_path):
    pytest.importorskip("pimm_data")
    from helix.data.coeff_dataset import CoeffTPCDataset
    ds = CoeffTPCDataset(data_root=str(corpus["run_dir"]), dataset_name="sim_wire", modalities=("coeff", "cells"))
    assert len(ds) == N_EV
    for i in range(N_EV):
        s = ds.get_data(i)
        rec, ident = C.open_event(corpus["cells"], i)
        assert s["ident"] == ident
        for k, v in rec.items():
            np.testing.assert_array_equal(s["cells"][k], v)
    assert "cells" not in CoeffTPCDataset(data_root=str(corpus["run_dir"]), dataset_name="sim_wire").get_data(0)
    with pytest.raises(ValueError, match="with 'coeff'"):
        CoeffTPCDataset(data_root=str(corpus["run_dir"]), dataset_name="sim_wire", modalities=("cells",))


def test_dataset_refuses_cells_of_other_events(corpus, tmp_path):
    pytest.importorskip("pimm_data")
    from helix.data.coeff_dataset import CoeffTPCDataset
    d = tmp_path / RUN
    d.mkdir()
    _coeff_shard(d / "sim_wire_coeff_0000.h5", [EV0 + i for i in range(N_EV)],
                 source="sim_wire_sensor_jaxtpc_wire_j000090-000094.h5")
    (d / "sim_wire_cells_0000.h5").write_bytes(open(corpus["cells"], "rb").read())
    with pytest.raises(ValueError):
        CoeffTPCDataset(data_root=str(d), dataset_name="sim_wire", modalities=("coeff", "cells"))


# ------------------------------------------------------------------------------------- real data
def _real():
    from _paths import sim_file
    return sim_file("hits"), sim_file("step")


@pytest.mark.skipif(not os.path.exists(_real()[0]), reason="campaign task not reachable")
def test_real_events_reduce_consistently():
    """Events of the campaign task j000085-000089, entry 202 (nothing in volume 1) among them: the
    record's tables agree with one another, almost every cell has a dominant group with a centroid,
    and the centroids lie inside the detector (step positions dequantised by their own origin and
    step)."""
    from helix.tpc import io
    hp, sp = _real()
    with h5py.File(hp, "r") as fh, h5py.File(sp, "r") as fs:
        ids = io.event_ids(fh)
        for pos in (0, 1, 202):
            rec = C.event_record(io.entry(fh, pos, ids[pos]), io.entry(fs, pos, ids[pos]))
            assert len(rec["key"]) > 1_000 and (np.diff(rec["key"]) > 0).all()
            np.testing.assert_allclose(np.bincount(rec["ct_cell"], weights=rec["ct_q"], minlength=len(rec["key"])),
                                       rec["q"], rtol=1e-5)
            assert ((rec["dom_frac"] > 0) & (rec["dom_frac"] <= 1 + 1e-6)).all()
            assert np.isfinite(rec["dom_y"]).mean() > 0.99
            yz = np.stack([rec["dom_y"], rec["dom_z"]])[:, np.isfinite(rec["dom_y"])]
            assert (np.abs(yz) <= 2160 + 1).all()
            if pos == 202:
                assert (C.unkey(rec["key"])[0] < 3).all() and (rec["trk_volume"] == 0).all()


def test_a_volume_with_no_deposit_adds_nothing(tmp_path):
    """The simulator writes a volume with nothing deposited (n_actual 0) without its per-deposit
    arrays; that event's cells and tracks are the other volume's alone."""
    hits = write_task(str(tmp_path / "sim"), RUN, FIRST_JOB, 2, empty={(1, 1)})["hits"]
    run_dir = tmp_path / "corpus" / RUN
    run_dir.mkdir(parents=True)
    _coeff_shard(run_dir / "sim_wire_coeff_0000.h5", [EV0, EV0 + 1])
    out, _ = C.build_shard(str(run_dir / "sim_wire_coeff_0000.h5"), str(tmp_path / "sim"))
    rec, _ = C.open_event(out, 1)
    assert len(rec["key"]) and (C.unkey(rec["key"])[0] < 3).all()       # volume 0's planes only
    assert (rec["trk_volume"] == 0).all()
    with h5py.File(hits, "r") as f:
        q = sum(C.decode_hits_plane(f["entry_000001"]["volume_0"][pl])["q"].sum() for pl in C.PLANES)
    np.testing.assert_allclose(rec["q"].sum(), q, rtol=1e-5)
    full, _ = C.open_event(out, 0)
    assert set(full["trk_volume"].tolist()) == {0, 1}
