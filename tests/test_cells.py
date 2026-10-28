"""helix.data.cells: the simulation truth on the fine grid, one file per coefficient shard.

Synthetic hits/step files (tests/_sim_fixture.py). Pinned: the cell charge IS the
denoiser's existing label (scripts/build_denoise_truth.py) bit for bit; the stored
record is complete for the evaluation's windows (the same windows as from pixels);
the build is resumable and atomic; the dataset serves it identity-checked.
"""
import os
import sys

import h5py
import numpy as np
import pytest

pytest.importorskip("hdf5plugin")

from _coeff_fixture import write_coeff_shard  # noqa: E402
from _sim_fixture import write_sim  # noqa: E402
from helix.data import cells as C  # noqa: E402
from helix.probe import resolution as R  # noqa: E402
from helix.probe.truth import PLANES, decode_hits_plane  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN, N_EV = "run_x", 3
BASIS = dict(wavelet="db2", dwt_level=2, dwt_mode="periodization", n_ticks_raw=32, pad=0, sigma_norm=2.6,
             removal_json="{}", threshold_json="{}")


def _coeff_shard(path, events, run=RUN, source="sim_wire_sensor_0000.h5"):
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
    hits = write_sim(str(d / "sim"), RUN, "0000", N_EV)
    run_dir = d / "corpus" / RUN
    run_dir.mkdir(parents=True)
    _coeff_shard(run_dir / "sim_wire_coeff_0000.h5", list(range(N_EV)))
    out, msg = C.build_shard(str(run_dir / "sim_wire_coeff_0000.h5"), str(d / "sim"))
    return dict(dir=d, run_dir=run_dir, hits=hits, cells=out, msg=msg)


def _pixels(hits, ev):
    """Per-pixel g, w, t, q and (volume, track id), independently of helix.data.cells."""
    G, W, T, Q, V, TID = [], [], [], [], [], []
    with h5py.File(hits, "r") as f:
        for vk in sorted(x for x in f[f"event_{ev:03d}"] if x.startswith("volume_")):
            hv, vi = f[f"event_{ev:03d}"][vk], int(vk.split("_")[1])
            g2t = hv["group_to_track"][:]
            for pi, pl in enumerate(PLANES):
                s = decode_hits_plane(hv[pl])
                G.append(np.full(len(s["q"]), vi * 3 + pi)); W.append(s["wire"]); T.append(s["tick"])
                Q.append(s["q"]); V.append(np.full(len(s["q"]), vi)); TID.append(g2t[s["group"]])
    return [np.concatenate(x) for x in (G, W, T, Q, V, TID)]


def test_cell_charge_is_the_denoise_label_bit_for_bit(corpus):
    from scripts.build_denoise_truth import event_cells
    with h5py.File(corpus["cells"], "r") as f:
        for ev in range(N_EV):
            rec = C.read_event(f, ev)
            k, q = event_cells(corpus["hits"], ev)
            np.testing.assert_array_equal(rec["key"], k)
            np.testing.assert_array_equal(rec["q"], q)


def test_record_is_the_pixels_reduced(corpus):
    """Per (cell, track) charge, per (plane, track) pixel centroid, per track step energy."""
    G, W, T, Q, V, TID = _pixels(corpus["hits"], 1)
    rec, _ = C.open_event(corpus["cells"], 1)
    trk = list(zip(rec["trk_volume"].tolist(), rec["trk_id"].tolist()))
    assert trk == sorted(set(trk))
    cell = R.fkey(G, W // R.FW, T // R.FT)
    t_of = {(v, t): i for i, (v, t) in enumerate(trk)}
    ti = np.array([t_of[(v, t)] for v, t in zip(V.tolist(), TID.tolist())])
    want = {}
    for c, t, q in zip(cell.tolist(), ti.tolist(), Q.tolist()):
        want[(c, t)] = want.get((c, t), 0.0) + q
    got = {(int(rec["key"][c]), int(t)): float(q) for c, t, q in zip(rec["ct_cell"], rec["ct_track"], rec["ct_q"])}
    assert got.keys() == want.keys()
    np.testing.assert_allclose([got[k] for k in want], list(want.values()), rtol=1e-6)
    assert len({k[0] for k in want}) == len(rec["key"]) < len(want)          # some cells hold two tracks
    for g, t, w_, t_ in zip(rec["pt_gid"], rec["pt_track"], rec["pt_wire"], rec["pt_tick"]):
        m = (G == g) & (ti == t)
        assert w_ == pytest.approx((Q[m] * W[m]).sum() / Q[m].sum(), rel=1e-12)
        assert t_ == pytest.approx((Q[m] * T[m]).sum() / Q[m].sum(), rel=1e-12)
    with h5py.File(corpus["hits"], "r") as fh, h5py.File(str(corpus["hits"]).replace("hits", "step"), "r") as fs:
        for i, (v, t) in enumerate(trk):
            d2g = fh[f"event_001/volume_{v}/deposit_to_group"][:]
            g2t = fh[f"event_001/volume_{v}/group_to_track"][:]
            de = fs[f"event_001/volume_{v}/de"][:].astype(np.float64)
            assert rec["trk_energy"][i] == pytest.approx(de[g2t[d2g] == t].sum(), rel=1e-6)


def test_grid_aligned_boxes_see_the_same_charge_per_track_as_pixels(corpus):
    """Why cells suffice: a box on the 2 x 16 grid holds, per track, exactly its pixels' charge."""
    G, W, T, Q, V, TID = _pixels(corpus["hits"], 0)
    rec, _ = C.open_event(corpus["cells"], 0)
    cg, cw, ct = R.unkey(rec["key"][rec["ct_cell"]])
    t_of = {(v, t): i for i, (v, t) in enumerate(zip(rec["trk_volume"].tolist(), rec["trk_id"].tolist()))}
    ti = np.array([t_of[(v, t)] for v, t in zip(V.tolist(), TID.tolist())])
    rng = np.random.default_rng(0)
    for _ in range(200):
        g, w0, t0 = int(rng.integers(0, 6)), int(rng.integers(0, 450)) * R.FW, int(rng.integers(0, 190)) * R.FT
        pin = (G == g) & (W >= w0) & (W < w0 + R.WW) & (T >= t0) & (T < t0 + R.WT)
        cin = (cg == g) & (cw * R.FW >= w0) & (cw * R.FW < w0 + R.WW) & (ct * R.FT >= t0) & (ct * R.FT < t0 + R.WT)
        for t in np.unique(np.concatenate([ti[pin], rec["ct_track"][cin]])):
            assert rec["ct_q"][cin & (rec["ct_track"] == t)].sum() == pytest.approx(Q[pin & (ti == t)].sum(), rel=1e-5)


def test_windows_from_the_file_are_the_windows_from_pixels(corpus):
    G, W, T, Q, V, TID = _pixels(corpus["hits"], 2)
    rec, _ = C.open_event(corpus["cells"], 2)
    t_of = {(v, t): i for i, (v, t) in enumerate(zip(rec["trk_volume"].tolist(), rec["trk_id"].tolist()))}
    ti = np.array([t_of[(v, t)] for v, t in zip(V.tolist(), TID.tolist())])
    rng = np.random.default_rng(3)
    coeff = np.concatenate([rec["key"][::3], R.fkey(rng.integers(0, 6, 400), rng.integers(0, 500, 400),
                                                    rng.integers(0, 200, 400))])
    pix = dict(g=G, w=W, t=T, q=Q, trk=ti, E=dict(enumerate(rec["trk_energy"].astype(float).tolist())))
    a = R.event_truth(pix, coeff, np.random.default_rng(7))
    b = R.event_truth_cells(rec, coeff, np.random.default_rng(7))
    kinds = [m["kind"] for m in a["meta"]]
    assert {"iso", "bg"} <= set(kinds)
    assert [(m["kind"], m["g"], m["w0"], m["t0"]) for m in a["meta"]] == \
        [(m["kind"], m["g"], m["w0"], m["t0"]) for m in b["meta"]]
    for ma, mb in zip(a["meta"], b["meta"]):
        np.testing.assert_allclose(ma["E"], mb["E"], rtol=1e-6)
        np.testing.assert_allclose(np.ravel(ma["cen"]), np.ravel(mb["cen"]), rtol=1e-12)
    for k in ("mrows", "wkey", "wid"):
        np.testing.assert_array_equal(a[k], b[k])
    np.testing.assert_allclose(a["wq"], b["wq"], rtol=1e-6)


def test_file_layout_resume_and_atomic_write(corpus):
    assert corpus["msg"].startswith(f"{N_EV} events")
    assert os.path.basename(corpus["cells"]) == "sim_wire_cells_0000.h5"
    assert not any(p.name.endswith(".part") for p in corpus["run_dir"].iterdir())
    with h5py.File(corpus["cells"], "r") as f:
        cfg = f["config"]
        assert cfg.attrs["format"] == C.FORMAT and (cfg.attrs["grid_fw"], cfg.attrs["grid_ft"]) == (R.FW, R.FT)
        import json
        assert json.loads(cfg.attrs["label_json"])["hits_threshold"] == 25.0
        assert C.read_ident(f)["event"].tolist() == list(range(N_EV))
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
    _coeff_shard(d / "sim_wire_coeff_0000.h5", list(range(N_EV)), source="sim_wire_sensor_0001.h5")
    (d / "sim_wire_cells_0000.h5").write_bytes(open(corpus["cells"], "rb").read())
    with pytest.raises(ValueError):
        CoeffTPCDataset(data_root=str(d), dataset_name="sim_wire", modalities=("coeff", "cells"))


# ------------------------------------------------------------------------------------- real data
def _real():
    from _paths import CORPUS, SENSOR_ROOT
    shard = os.path.join(CORPUS, "sim_wire_coeff_0000.h5")
    return shard, os.path.dirname(SENSOR_ROOT.rstrip("/"))


@pytest.mark.skipif(not (os.path.exists(_real()[0]) and os.path.isdir(_real()[1])), reason="corpus / sim not reachable")
def test_real_events_match_the_denoise_label_bit_for_bit():
    """Three events of the production corpus: the record's cell charge == build_denoise_truth's."""
    from scripts.build_denoise_truth import event_cells
    shard, source = _real()
    with h5py.File(shard, "r") as f:
        ident = C.read_ident(f)
    for pos in range(3):
        run, sf, ev = ident["run"][pos], ident["source_file"][pos], int(ident["event"][pos])
        hp = os.path.join(source, "hits", run, sf.replace("_sensor_", "_hits_"))
        with h5py.File(hp, "r") as fh, h5py.File(hp.replace("hits", "step"), "r") as fs:
            rec = C.event_record(fh[f"event_{ev:03d}"], fs[f"event_{ev:03d}"])
        k, q = event_cells(hp, ev)
        assert len(k) > 10_000
        np.testing.assert_array_equal(rec["key"], k)
        np.testing.assert_array_equal(rec["q"], q)
        np.testing.assert_allclose(np.bincount(rec["ct_cell"], weights=rec["ct_q"], minlength=len(k)), q, rtol=1e-5)
