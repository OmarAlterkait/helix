"""helix.data.cells: the simulation truth on the fine grid, one file per coefficient shard.

A synthetic campaign task (tests/_fixtures.py). Pinned: the stored record is the pixels
reduced (per cell, per (cell, track), per (plane, track), per track, and each cell's dominant
group with its 3D centroid), computed here independently; each track carries its labl classes; it is complete for the evaluation's
windows (a grid-aligned box holds per track exactly its pixels' charge); the build is resumable
and atomic; the dataset serves it identity-checked. The grid is the file's: written by the
builder, read back with every record, refused when absent or mixed; a finer truth (1 x 4, 1 x 1)
sums to the 2 x 16 one exactly (helix.data.cells.aggregate), on the synthetic task and, where the
1 x 1 finetruth is reachable, on the stored d1m events.
"""
import os
import sys

import h5py
import numpy as np
import pytest

pytest.importorskip("hdf5plugin")

from _fixtures import write_coeff_shard  # noqa: E402
from _fixtures import write_task  # noqa: E402
from helix.data import cells as C  # noqa: E402
from helix.eval import events as Wn  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN, N_EV, FIRST_JOB = "j000000", 3, 85
GRID = C.Grid(2, 16)                  # the d1m truth's grid
EV0 = FIRST_JOB * 200                 # the task's first campaign event id
BASIS = dict(wavelet="db2", dwt_level=2, dwt_mode="periodization", n_ticks_raw=32, pad=0, sigma_norm=2.6,
             removal_json="{}", threshold_json="{}")


SOURCE = "sim_wire_sensor_jaxtpc_wire_j000085-000089.h5"


def _event(path, pos):
    """``(record, ident)`` of event ``pos`` of a cells shard, format-checked."""
    with h5py.File(path, "r") as f:
        C.require_format(f["config"], str(path))
        ident = C.read_ident(f)
        return C.read_event(f, pos), dict(run=ident["run"][pos], source_file=ident["source_file"][pos],
                                          event=int(ident["event"][pos]))


def _coeff_shard(path, events, run=RUN, source=SOURCE):
    rng = np.random.default_rng(1)
    evs = [dict(band=np.zeros(4, np.uint8), plane_gid=np.zeros(4, np.int32), wire=rng.integers(0, 4, 4),
                tau=rng.integers(0, 8, 4), value=rng.standard_normal(4).astype(np.float32),
                sigma_threshold=rng.random((2, 3)).astype(np.float32), run=run, source_file=source, event=e)
           for e in events]
    # a stand-in for the identity join, but its planes must hold the cells' truth: the dataset drops
    # an event whose truth lies off its coeff shard's planes (helix.data.cells.valid_cells)
    write_coeff_shard(path, evs, band_lengths=[8, 8, 16], gids=list(range(6)), n_wires=[1 << 16] * 6,
                      basis_attrs=BASIS, dataset_name="sim_wire")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("cells")
    task = write_task(str(d / "sim"), RUN, FIRST_JOB, N_EV)
    hits = task["hits"]
    run_dir = d / "corpus" / RUN
    run_dir.mkdir(parents=True)
    _coeff_shard(run_dir / "sim_wire_coeff_0000.h5", [EV0 + i for i in range(N_EV)])
    out, msg = C.build_shard(str(run_dir / "sim_wire_coeff_0000.h5"), str(d / "sim"), grid=GRID)
    return dict(dir=d, run_dir=run_dir, hits=hits, labl=task["labl"], cells=out, msg=msg)


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
    rec, _ = _event(corpus["cells"], 1)
    trk = list(zip(rec["trk_volume"].tolist(), rec["trk_id"].tolist()))
    assert trk == sorted(set(trk))
    cell = C.fkey(G, W // GRID.fw, T // GRID.ft)
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


def test_tracks_carry_their_labl_classes_and_cells_their_dominant_tracks(corpus):
    """A track's motif is its fragment's semantic type in pimm-data's vocabulary, its pid its
    instance's sign-blind PDG (a low-energy instance -> le, another PDG -> -1), -1 for an unowned
    track; a cell takes the class of the track carrying most of its charge."""
    rec, _ = _event(corpus["cells"], 1)
    motif_of_sem = {1: 0, 2: 1, 3: 3, 4: 2, 5: 4}
    pid_of_pdg = {22: 0, 11: 1, 13: 2, 211: 3, 2212: 4}
    with h5py.File(corpus["labl"], "r") as f:
        e = f["entry_000001"]
        tid, frag = list(e["track/id"][:]), e["track/fragment"][:]
        sem, isem, pdg = (e[f"particle/{k}"][:] for k in ("fragment_sem_type", "instance_sem_type", "pdg"))
    seen = set()
    for t, m, p in zip(rec["trk_id"], rec["trk_motif"], rec["trk_pid"]):
        fr = frag[tid.index(t)]
        want_m = -1 if fr < 0 else motif_of_sem[int(sem[fr])]
        want_p = -1 if fr < 0 else 5 if isem[fr] == 5 else pid_of_pdg.get(abs(int(pdg[fr])), -1)
        assert (m, p) == (want_m, want_p)
        seen.add((want_m >= 0, want_p))
    assert (False, -1) in seen and any(m for m, _ in seen) and {p for _, p in seen} >= {-1, 5}
    cls = C.cell_class(rec, "trk_motif")
    for i in range(0, len(rec["key"]), 7):
        rows = np.nonzero(rec["ct_cell"] == i)[0]
        top = rows[np.argmax(rec["ct_q"][rows])]
        assert cls[i] == rec["trk_motif"][rec["ct_track"][top]]


def test_each_cell_records_its_dominant_group_and_its_centroid(corpus):
    """The along-wire probe's label: the largest group's share of the cell, and its deposits'
    charge-weighted (y, z)."""
    G, W, T, Q, _, _, GRP, YZ = _pixels(corpus["hits"], 0)
    rec, _ = _event(corpus["cells"], 0)
    cell = C.fkey(G, W // GRID.fw, T // GRID.ft)
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
    rec, _ = _event(corpus["cells"], 0)
    cg, cw, ct = C.unkey(rec["key"][rec["ct_cell"]])
    t_of = {(v, t): i for i, (v, t) in enumerate(zip(rec["trk_volume"].tolist(), rec["trk_id"].tolist()))}
    ti = np.array([t_of[(v, t)] for v, t in zip(V.tolist(), TID.tolist())])
    rng = np.random.default_rng(0)
    for _ in range(200):
        g, w0, t0 = int(rng.integers(0, 6)), int(rng.integers(0, 450)) * GRID.fw, int(rng.integers(0, 190)) * GRID.ft
        pin = (G == g) & (W >= w0) & (W < w0 + Wn.WW) & (T >= t0) & (T < t0 + Wn.WT)
        cin = (cg == g) & (cw * GRID.fw >= w0) & (cw * GRID.fw < w0 + Wn.WW) & (ct * GRID.ft >= t0) & (ct * GRID.ft < t0 + Wn.WT)
        for t in np.unique(np.concatenate([ti[pin], rec["ct_track"][cin]])):
            assert rec["ct_q"][cin & (rec["ct_track"] == t)].sum() == pytest.approx(Q[pin & (ti == t)].sum(), rel=1e-5)


def test_the_windows_of_a_record(corpus):
    """event_truth on a stored record: every window kind, reproducible, and no background window
    holding charge."""
    rec, _ = _event(corpus["cells"], 2)
    rng = np.random.default_rng(3)
    coeff = np.concatenate([rec["key"][::3], C.fkey(rng.integers(0, 6, 400), rng.integers(0, 500, 400),
                                                    rng.integers(0, 200, 400))])
    assert rec["grid"] == GRID
    a = Wn.event_truth(rec, coeff, np.random.default_rng(7))
    b = Wn.event_truth(rec, coeff, np.random.default_rng(7))
    assert {"iso", "bg"} <= {m["kind"] for m in a["meta"]}
    np.testing.assert_array_equal(a["wkey"], b["wkey"])
    assert [str(m) for m in a["meta"]] == [str(m) for m in b["meta"]]           # str: NaN energies compare
    q = dict(zip(rec["key"].tolist(), rec["q"].tolist()))
    wq =np.array([q.get(k, 0.0) for k in a["wkey"].tolist()]).reshape(len(a["meta"]), -1).sum(1)
    assert all(w == 0 for w, m in zip(wq, a["meta"]) if m["kind"] == "bg")
    assert all(w > 0 for w, m in zip(wq, a["meta"]) if m["kind"] == "iso")


def test_file_layout_resume_and_atomic_write(corpus):
    assert corpus["msg"].startswith(f"{N_EV} events")
    assert os.path.basename(corpus["cells"]) == "sim_wire_cells_0000.h5"
    assert not any(p.name.endswith(".part") for p in corpus["run_dir"].iterdir())
    with h5py.File(corpus["cells"], "r") as f:
        cfg = f["config"]
        assert cfg.attrs["format"] == C.FORMAT and (cfg.attrs["grid_fw"], cfg.attrs["grid_ft"]) == (GRID.fw, GRID.ft)
        import json
        assert json.loads(cfg.attrs["label_json"])["hits_threshold"] == 25.0
        assert C.read_ident(f)["event"].tolist() == [EV0 + i for i in range(N_EV)]
        for g in C.GROUPS:
            assert f[g]["event_offset"].shape == (N_EV + 1,)
    before = os.stat(corpus["cells"]).st_mtime_ns
    assert C.build_shard(str(corpus["run_dir"] / "sim_wire_coeff_0000.h5"), str(corpus["dir"] / "sim"),
                         grid=GRID)[1] == "exists"
    assert os.stat(corpus["cells"]).st_mtime_ns == before
    with pytest.raises(ValueError, match="grid 2x16, not 1x4"):            # another grid is another truth
        C.build_shard(str(corpus["run_dir"] / "sim_wire_coeff_0000.h5"), str(corpus["dir"] / "sim"), grid=(1, 4))


def test_the_truth_script_builds_every_shard_of_a_run(corpus, tmp_path):
    """scripts/build_truth.py --corpus RUN_DIR: a cells file beside each coeff shard, its events the
    shard's, read from the campaign at HELIX_SIM_ROOT -- the same records build_shard writes."""
    import subprocess
    run = tmp_path / RUN
    run.mkdir()
    _coeff_shard(run / "sim_wire_coeff_0000.h5", [EV0, EV0 + 1])
    _coeff_shard(run / "sim_wire_coeff_0001.h5", [EV0 + 2])
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "build_truth.py")
    r = subprocess.run([sys.executable, script, "--corpus", str(run)], capture_output=True, text=True,
                       env=dict(os.environ, HELIX_SIM_ROOT=str(corpus["dir"] / "sim")))
    assert r.returncode == 0, r.stderr
    for k, events in ((0, [EV0, EV0 + 1]), (1, [EV0 + 2])):
        path = run / f"sim_wire_cells_{k:04d}.h5"
        for pos, ev in enumerate(events):
            rec, ident = _event(path, pos)
            want, _ = _event(corpus["cells"], ev - EV0)
            assert ident["event"] == ev and set(rec) == set(want)
            for n in want:
                np.testing.assert_array_equal(rec[n], want[n], err_msg=n)       # NaN == NaN here


def test_reader_refuses_another_format_or_no_grid(corpus, tmp_path):
    p = tmp_path / "sim_wire_cells_0000.h5"
    p.write_bytes(open(corpus["cells"], "rb").read())
    with h5py.File(p, "a") as f:
        del f["config"].attrs["grid_ft"]
    with pytest.raises(ValueError, match="records no grid"):
        _event(str(p), 0)
    with h5py.File(p, "a") as f:
        f["config"].attrs["grid_ft"] = 0
    with pytest.raises(ValueError, match=">= 1"):
        _event(str(p), 0)
    with h5py.File(p, "a") as f:
        f["config"].attrs["format"] = "helix-cells-1"
    with pytest.raises(ValueError, match="rebuild"):
        _event(str(p), 0)


# ------------------------------------------------------------------------------- dataset modality
def test_dataset_serves_cells_identity_checked(corpus, tmp_path):
    pytest.importorskip("pimm_data")
    from helix.data.dataset import CoeffTPCDataset
    ds = CoeffTPCDataset(data_root=str(corpus["run_dir"]), dataset_name="sim_wire", modalities=("coeff", "cells"))
    assert len(ds) == N_EV
    for i in range(N_EV):
        s = ds.get_data(i)
        rec, ident = _event(corpus["cells"], i)
        assert s["ident"] == ident
        for k, v in rec.items():
            np.testing.assert_array_equal(s["cells"][k], v)
    # training reads only key and q (configs/pimm/denoise.py)
    two = CoeffTPCDataset(data_root=str(corpus["run_dir"]), dataset_name="sim_wire", modalities=("coeff", "cells"),
                          cells_columns=("key", "q")).get_data(1)["cells"]
    assert set(two) == {"key", "q", "grid"} and two["grid"] == GRID
    for k in two:
        np.testing.assert_array_equal(two[k], _event(corpus["cells"], 1)[0][k])
    assert "cells" not in CoeffTPCDataset(data_root=str(corpus["run_dir"]), dataset_name="sim_wire").get_data(0)
    with pytest.raises(ValueError, match="with 'coeff'"):
        CoeffTPCDataset(data_root=str(corpus["run_dir"]), dataset_name="sim_wire", modalities=("cells",))


def test_dataset_refuses_cells_of_other_events(corpus, tmp_path):
    pytest.importorskip("pimm_data")
    from helix.data.dataset import CoeffTPCDataset
    d = tmp_path / RUN
    d.mkdir()
    _coeff_shard(d / "sim_wire_coeff_0000.h5", [EV0 + i for i in range(N_EV)],
                 source="sim_wire_sensor_jaxtpc_wire_j000090-000094.h5")
    (d / "sim_wire_cells_0000.h5").write_bytes(open(corpus["cells"], "rb").read())
    with pytest.raises(ValueError):
        CoeffTPCDataset(data_root=str(d), dataset_name="sim_wire", modalities=("coeff", "cells"))


# ------------------------------------------------------------------------------------- real data
def _real():
    from _fixtures import sim_file
    return sim_file("hits"), sim_file("step"), sim_file("labl")


@pytest.mark.skipif(not os.path.exists(_real()[0]), reason="campaign task not reachable")
def test_real_events_reduce_consistently():
    """Events of the campaign task j000085-000089, entry 202 (nothing in volume 1) among them: the
    record's tables agree with one another, almost every cell has a dominant group with a centroid,
    and the centroids lie inside the detector (step positions dequantised by their own origin and
    step)."""
    from helix.tpc import io
    hp, sp, lp = _real()
    with h5py.File(hp, "r") as fh, h5py.File(sp, "r") as fs, h5py.File(lp, "r") as fl:
        ids = io.event_ids(fh)
        for pos in (0, 1, 202):
            rec = C.event_record(*(io.entry(f, pos, ids[pos]) for f in (fh, fs, fl)), GRID)
            assert len(rec["key"]) > 1_000 and (np.diff(rec["key"]) > 0).all()
            np.testing.assert_allclose(np.bincount(rec["ct_cell"], weights=rec["ct_q"], minlength=len(rec["key"])),
                                       rec["q"], rtol=1e-5)
            assert ((rec["dom_frac"] > 0) & (rec["dom_frac"] <= 1 + 1e-6)).all()
            assert np.isfinite(rec["dom_y"]).mean() > 0.99
            yz = np.stack([rec["dom_y"], rec["dom_z"]])[:, np.isfinite(rec["dom_y"])]
            assert (np.abs(yz) <= 2160 + 1).all()
            # nearly all charge is on a labelled track
            cm = C.cell_class(rec, "trk_motif")
            assert rec["q"][cm >= 0].sum() > 0.99 * rec["q"].sum()
            if pos == 202:
                assert (C.unkey(rec["key"])[0] < 3).all() and (rec["trk_volume"] == 0).all()


def test_a_volume_with_no_deposit_adds_nothing(tmp_path):
    """The simulator writes a volume with nothing deposited (n_actual 0) without its per-deposit
    arrays; that event's cells and tracks are the other volume's alone."""
    hits = write_task(str(tmp_path / "sim"), RUN, FIRST_JOB, 2, empty={(1, 1)})["hits"]
    run_dir = tmp_path / "corpus" / RUN
    run_dir.mkdir(parents=True)
    _coeff_shard(run_dir / "sim_wire_coeff_0000.h5", [EV0, EV0 + 1])
    out, _ = C.build_shard(str(run_dir / "sim_wire_coeff_0000.h5"), str(tmp_path / "sim"), grid=GRID)
    rec, _ = _event(out, 1)
    assert len(rec["key"]) and (C.unkey(rec["key"])[0] < 3).all()       # volume 0's planes only
    assert (rec["trk_volume"] == 0).all()
    with h5py.File(hits, "r") as f:
        q = sum(C.decode_hits_plane(f["entry_000001"]["volume_0"][pl])["q"].sum() for pl in C.PLANES)
    np.testing.assert_allclose(rec["q"].sum(), q, rtol=1e-5)
    full, _ = _event(out, 0)
    assert set(full["trk_volume"].tolist()) == {0, 1}


def test_a_cell_off_the_plane_is_dropped_and_the_track_split_renumbered():
    """A negative key (a fine wire below 0) is not a cell: it goes, with its cell_track rows, and
    the remaining ct_cell indices follow their cells."""
    from helix.data.cells import fkey, valid_cells
    keys = np.array([fkey(0, 3, 5), fkey(1, -1, 5), fkey(1, 4, 6)], np.int64)
    assert keys[1] < 0
    rec = dict(key=keys, q=np.array([1.0, 2.0, 3.0], np.float32), dom_frac=np.ones(3, np.float32),
               dom_y=np.zeros(3, np.float32), dom_z=np.zeros(3, np.float32),
               ct_cell=np.array([0, 1, 2, 2], np.int32), ct_track=np.array([7, 8, 9, 10], np.int32),
               ct_q=np.array([1.0, 2.0, 2.5, 0.5], np.float32))
    out = valid_cells(rec)
    assert out["key"].tolist() == [keys[0], keys[2]] and out["q"].tolist() == [1.0, 3.0]
    assert out["ct_cell"].tolist() == [0, 1, 1] and out["ct_track"].tolist() == [7, 9, 10]
    assert valid_cells(dict(rec, key=np.abs(keys)))["key"].shape == (3,)     # nothing to drop: untouched


def test_cell_truth_drops_samples_off_the_plane():
    """A sample below wire 0 (or before tick 0) is not a cell: it goes, not onto the edge."""
    from helix.data.cells import cell_truth, unkey
    g = np.array([0, 0, 0]); w = np.array([4, -1, 5]); t = np.array([32, 32, 33])
    q = np.array([1.0, 9.0, 2.0]); trk = np.array([0, 0, 1]); grp = np.array([0, 0, 1])
    rec = cell_truth(g, w, t, q, trk, grp, np.zeros((3, 2)), GRID)
    assert (np.asarray(rec["key"]) >= 0).all() and np.asarray(rec["q"]).sum() == 3.0
    assert set(unkey(rec["key"])[1].tolist()) == {2}                 # wires 4 and 5 share fine wire 2


def test_valid_cells_keeps_only_the_events_planes_wires_and_ticks():
    """With the event's planes: a cell beyond its plane's last wire, on a plane the event lacks, or
    outside [0, n_ticks) goes."""
    from helix.data.cells import fkey, valid_cells
    for grid in (GRID, C.Grid(1, 4)):                               # in wires and ticks, on the record's grid
        fw, ft = grid
        keys = np.array([fkey(0, 10, 5), fkey(0, 500 // fw, 5), fkey(3, 1, 5), fkey(1, 2, 4352 // ft),
                         fkey(1, 2, 1), fkey(0, 398 // fw, 5)], np.int64)
        rec = dict(key=keys, q=np.arange(6, dtype=np.float32), dom_frac=np.ones(6, np.float32),
                   dom_y=np.zeros(6, np.float32), dom_z=np.zeros(6, np.float32),
                   ct_cell=np.arange(6, dtype=np.int32), ct_track=np.arange(6, dtype=np.int32),
                   ct_q=np.ones(6, np.float32), grid=grid)
        out = valid_cells(rec, gids=[0, 1], n_wires=[400, 300], n_ticks=4336)
        assert out["key"].tolist() == [keys[0], keys[4], keys[5]] and out["ct_cell"].tolist() == [0, 1, 2]
        assert out["grid"] == grid


# ------------------------------------------------------------------------------------ derived grids
def _within_ulp(a, b):
    """``a`` and ``b`` (float32) at most one float32 ulp apart, elementwise: the residue of
    rounding each sub-cell before summing them where the coarse build rounds once per cell."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return bool((np.abs(a.astype(np.float64) - b) <= np.spacing(np.maximum(np.abs(a), np.abs(b)))).all())


def _assert_derives(fine, coarse):
    """``aggregate(fine)`` is ``coarse`` -- both stored records -- where the rule says it is exact:
    the same cells and (cell, track) rows, charges within one float32 ulp, the grid-free tables
    equal, and every cell's class; ``dom_frac`` a lower bound, equal where a coarse cell has one
    sub-cell."""
    got = C.as_stored(C.aggregate(fine, coarse["grid"]))
    assert got["grid"] == coarse["grid"]
    np.testing.assert_array_equal(got["key"], coarse["key"])
    assert _within_ulp(got["q"], coarse["q"])
    for k in ("ct_cell", "ct_track"):
        np.testing.assert_array_equal(got[k], coarse[k])
    assert _within_ulp(got["ct_q"], coarse["ct_q"])
    for k in ("pt_gid", "pt_track", "pt_wire", "pt_tick", "trk_volume", "trk_id", "trk_energy", "trk_motif", "trk_pid"):
        np.testing.assert_array_equal(got[k], coarse[k])
    for col in ("trk_motif", "trk_pid"):
        np.testing.assert_array_equal(C.cell_class(got, col), C.cell_class(coarse, col))
    assert (got["dom_frac"] <= coarse["dom_frac"] * (1 + 1e-6) + 1e-7).all()
    g, w, t = C.unkey(fine["key"])
    r = (coarse["grid"].fw // fine["grid"].fw, coarse["grid"].ft // fine["grid"].ft)
    alone = np.bincount(np.searchsorted(coarse["key"], C.fkey(g, w // r[0], t // r[1])), minlength=len(coarse["key"])) == 1
    np.testing.assert_allclose(got["dom_frac"][alone], coarse["dom_frac"][alone], rtol=1e-6)
    np.testing.assert_array_equal(got["dom_y"][alone], coarse["dom_y"][alone])
    return got


@pytest.mark.parametrize("fine", [(1, 1), (1, 4), (2, 8)])
def test_a_finer_build_sums_to_the_2x16_build(corpus, tmp_path, fine):
    """The builder writes any grid; a finer one, summed (helix.data.cells.aggregate), is the
    2 x 16 build of the same events -- cells, (cell, track) split, classes, tracks."""
    run_dir = corpus["run_dir"]
    out, msg = C.build_shard(str(run_dir / "sim_wire_coeff_0000.h5"), str(corpus["dir"] / "sim"),
                             str(tmp_path), grid=fine)
    assert msg.endswith(f"({fine[0]}x{fine[1]})")
    with h5py.File(out, "r") as f:
        assert C.require_format(f["config"]) == fine
    for pos in range(N_EV):
        fr, cr = _event(out, pos)[0], _event(corpus["cells"], pos)[0]
        assert fr["grid"] == fine and len(fr["key"]) > len(cr["key"])
        _assert_derives(fr, cr)


def test_derive_shard_writes_the_coarser_truth(corpus, tmp_path):
    """derive_shard: a 1 x 4 truth summed to 2 x 16 is a file the reader takes as the 2 x 16 one."""
    run_dir = corpus["run_dir"]
    fine, _ = C.build_shard(str(run_dir / "sim_wire_coeff_0000.h5"), str(corpus["dir"] / "sim"),
                            str(tmp_path / "f"), grid=(1, 4))
    out, msg = C.derive_shard(fine, str(tmp_path / "sim_wire_cells_0000.h5"), GRID)
    assert "1x4 -> 2x16" in msg and C.derive_shard(fine, out, GRID)[1] == "exists"
    import json
    with h5py.File(out, "r") as f, h5py.File(corpus["cells"], "r") as g:
        assert C.require_format(f["config"]) == GRID
        np.testing.assert_array_equal(C.read_ident(f)["event"], C.read_ident(g)["event"])
        assert json.loads(f["config"].attrs["provenance_json"])["derived_from"]["grid"] == "1x4"
    for pos in range(N_EV):
        a, b = _event(out, pos)[0], _event(corpus["cells"], pos)[0]
        np.testing.assert_array_equal(a["key"], b["key"])
        assert _within_ulp(a["q"], b["q"]) and _within_ulp(a["ct_q"], b["ct_q"])


def test_aggregate_refuses_a_grid_that_splits_cells_and_keeps_only_given_columns():
    rec = C.cell_truth(np.zeros(4, int), np.array([0, 1, 2, 3]), np.array([0, 5, 9, 17]), np.ones(4),
                       np.zeros(4, int), np.arange(4), np.zeros((4, 2)), (1, 4))
    with pytest.raises(ValueError, match="not a multiple"):
        C.aggregate(rec, (2, 6))                                    # 6 ticks are not whole 4-tick cells
    two = C.aggregate(dict(key=rec["key"], q=rec["q"], grid=rec["grid"]), GRID)
    assert set(two) == {"key", "q", "grid"}
    assert two["key"].tolist() == [C.fkey(0, 0, 0), C.fkey(0, 1, 0), C.fkey(0, 1, 1)] and two["q"].tolist() == [2.0, 1.0, 1.0]
    with pytest.raises(KeyError, match="grid"):
        C.aggregate(dict(key=rec["key"], q=rec["q"]), GRID)          # a record without its grid


def test_a_dataset_serves_a_1x4_truth_whole_and_refuses_mixed_grids(corpus, tmp_path):
    """The dataset reads the grid from the file: a 1 x 4 truth keeps every wire of the plane
    (on a hard-coded 2 x 16, wires beyond n_wires / 2 read as off the plane), and shards of two
    grids are not one truth."""
    pytest.importorskip("pimm_data")
    from helix.data.dataset import CoeffTPCDataset
    d = tmp_path / RUN
    d.mkdir()
    _coeff_shard(d / "sim_wire_coeff_0000.h5", [EV0 + i for i in range(N_EV)])
    C.build_shard(str(d / "sim_wire_coeff_0000.h5"), str(corpus["dir"] / "sim"), grid=(1, 4))
    ds = CoeffTPCDataset(data_root=str(d), dataset_name="sim_wire", modalities=("coeff", "cells"))
    for i in range(N_EV):
        rec = ds.get_data(i)["cells"]
        assert rec["grid"] == (1, 4)
        np.testing.assert_array_equal(rec["key"], _event(str(d / "sim_wire_cells_0000.h5"), i)[0]["key"])
    _coeff_shard(d / "sim_wire_coeff_0001.h5", [EV0 + i for i in range(N_EV)],
                 source="sim_wire_sensor_jaxtpc_wire_j000090-000094.h5")
    (d / "sim_wire_cells_0001.h5").write_bytes(open(corpus["cells"], "rb").read())
    with pytest.raises(ValueError, match="grid"):
        CoeffTPCDataset(data_root=str(d), dataset_name="sim_wire", modalities=("coeff", "cells"))


# The 1 x 1 finetruth (HELIX_FINETRUTH: per-pixel cells files of 10 d1m shards, built from the same
# hits as the stored 2 x 16 truth beside the corpus) against that stored truth, on the events of
# every shard both hold. Event 132912 (shard 0664) is left out: its stored 2 x 16 plane 3 is
# off the plane (helix.data.cells.valid_cells), its 1 x 1 plane 3 dropped by the finetruth's guard
# (/event_qc/reject).
N_FINE_EVENTS = int(os.environ.get("HELIX_FINETRUTH_EVENTS", "3"))


def _finetruth_pairs():
    from _fixtures import CORPUS, FINETRUTH
    if not (FINETRUTH and os.path.isdir(FINETRUTH) and CORPUS and os.path.isdir(CORPUS)):
        return []
    out = []
    for p in sorted(os.listdir(FINETRUTH)):
        if p.endswith(".h5") and "_pixels_" in p:
            c = os.path.join(CORPUS, p.replace("_pixels_", "_cells_"))
            if os.path.exists(c):
                out.append((os.path.join(FINETRUTH, p), c))
    return out


@pytest.mark.skipif(not _finetruth_pairs(), reason="no finetruth shard with its stored truth reachable "
                                                   "(HELIX_FINETRUTH, HELIX_CORPUS)")
@pytest.mark.parametrize("via", [None, (1, 4)])
def test_the_finetruth_sums_to_the_stored_2x16_truth(via):
    """The stored d1m events: the 1 x 1 truth summed to 2 x 16 -- directly, or through a STORED
    1 x 4 truth (rounded to float32 at 1 x 4, as a 1 x 4 file holds it) -- is the 2 x 16 truth
    beside the corpus: the same cells and (cell, track) rows, charge within one float32 ulp
    directly and within two through 1 x 4 (one more rounding), tracks and classes equal."""
    n = 0
    for fp, cp in _finetruth_pairs():
        with h5py.File(fp, "r") as ff, h5py.File(cp, "r") as fc:
            assert C.require_format(ff["config"], fp) == (1, 1) and C.require_format(fc["config"], cp) == GRID
            reject = ff["event_qc/reject"][:] if "event_qc" in ff else np.zeros(len(ff["ident/event"]), np.int8)
            assert np.array_equal(C.read_ident(ff)["event"], C.read_ident(fc)["event"])
            for pos in [i for i in range(len(reject)) if not reject[i]][:N_FINE_EVENTS]:
                fine, coarse = C.read_event(ff, pos), C.read_event(fc, pos)
                if via is not None:
                    fine = C.as_stored(C.aggregate(fine, via))
                    assert fine["grid"] == via
                got = C.as_stored(C.aggregate(fine, GRID))
                np.testing.assert_array_equal(got["key"], coarse["key"])
                np.testing.assert_array_equal(got["ct_cell"], coarse["ct_cell"])
                np.testing.assert_array_equal(got["ct_track"], coarse["ct_track"])
                ulps = 1 if via is None else 2
                for k in ("q", "ct_q"):
                    a, b = got[k].astype(np.float64), coarse[k].astype(np.float64)
                    assert (np.abs(a - b) <= ulps * np.spacing(np.maximum(np.abs(got[k]), np.abs(coarse[k])))).all(), k
                for k in ("pt_gid", "pt_track", "pt_wire", "pt_tick", "trk_id", "trk_energy", "trk_motif", "trk_pid"):
                    np.testing.assert_array_equal(got[k], coarse[k])
                for col in ("trk_motif", "trk_pid"):
                    np.testing.assert_array_equal(C.cell_class(got, col), C.cell_class(coarse, col))
                n += 1
    assert n > 0
