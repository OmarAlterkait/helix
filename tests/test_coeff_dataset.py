"""CoeffTPCDataset / CoeffTPCReader over the flat-columnar coeff corpus: the round trip, one corpus
per reader, event identity, the holdout and the event subsets. The corpus verifier is
tests/test_verify.py."""
import json

import numpy as np
import pytest

pytest.importorskip("pimm_data")          # the framework this family plugs into

from helix.data.dataset import CoeffTPCDataset, CoeffTPCReader  # noqa: E402
from _fixtures import write_coeff_shard  # noqa: E402

BAND_LENGTHS = [8, 8, 16]
GIDS = [0, 1]
N_WIRES = [4, 4]
BASIS_ATTRS = dict(
    wavelet="db2", dwt_level=2, dwt_mode="periodization", n_ticks_raw=32, pad=0,
    sigma_norm=2.6,
    removal_json=json.dumps(dict(kind="gate", kgate=3.0)),
    threshold_json=json.dumps(dict(method="universal")),
)
#: provenance a verifier accepts: one commit, a clean tree, the DSP inputs' hashes
FIXTURE_PROV = dict(geom="cubic_wireplane_geometry.json", geom_sha256="a" * 8,
                    spectrum="microboone", spectrum_sha256="c" * 8,
                    code=dict(version="0.0.0-test", git="0" * 40, git_dirty=False))


def _ev(rng, ev, n=6, gids=GIDS, nb=len(BAND_LENGTHS)):
    band = rng.integers(0, nb, n).astype(np.uint8)
    gid = rng.choice(np.asarray(gids), n).astype(np.int32)
    wire = rng.integers(0, 4, n).astype(np.int32)
    tau = np.array([rng.integers(0, BAND_LENGTHS[b]) for b in band], np.int32)
    # unique coords: a real CoeffEvent has one value per (gid,band,wire,tau)
    key = (gid.astype(np.int64) << 40) | (band.astype(np.int64) << 36) \
        | (wire.astype(np.int64) << 18) | tau
    _, keep = np.unique(key, return_index=True)
    keep = np.sort(keep)
    return dict(band=band[keep], plane_gid=gid[keep], wire=wire[keep], tau=tau[keep],
                value=rng.standard_normal(keep.size).astype(np.float32),
                sigma_threshold=np.full((len(gids), nb), 1.5, np.float32),
                run="run_000", source_file=f"cx_sensor_{ev // 2:04d}.h5", event=ev)


def _write(dirpath, event_ids, file_index=0, norm_sigma=None, seed=0, gids=GIDS,
           provenance=FIXTURE_PROV, basis_attrs=None):
    """One coeff shard ``dirpath/cx_coeff_<file_index>.h5`` of ``event_ids``; returns its events."""
    rng = np.random.default_rng(seed)
    evs = [_ev(rng, e, gids=gids) for e in event_ids]
    write_coeff_shard(dirpath / f"cx_coeff_{file_index:04d}.h5", evs, band_lengths=BAND_LENGTHS, gids=gids,
                      n_wires=N_WIRES, basis_attrs=basis_attrs or BASIS_ATTRS, dataset_name="cx",
                      file_index=file_index, norm_sigma=norm_sigma, provenance=provenance)
    return evs


def _ds(root, **kw):
    return CoeffTPCDataset(data_root=str(root), dataset_name="cx", **kw)


NS = np.full((len(GIDS), len(BAND_LENGTHS)), 2.0, np.float32)


# ---- the round trip ----------------------------------------------------------------------

def test_reader_and_dataset_round_trip(tmp_path):
    """What the codec writes, the reader reads; the sample carries the shard tables a worker-side
    tokenizer needs, and the source event's identity at the top level (not in a row-space)."""
    events = _write(tmp_path, [0, 1, 2], norm_sigma=NS)
    r = CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx")
    assert len(r) == 3
    np.testing.assert_array_equal(r.band_lengths, BAND_LENGTHS)
    for i, e in enumerate(events):
        got = r.read_event(i)
        for k in ("band", "plane_gid", "wire", "tau", "value"):
            np.testing.assert_array_equal(got[k], e[k], err_msg=f"event {i} {k}")
    ds = _ds(tmp_path)
    d = ds.get_data(0)
    assert set(d["coeff"]) == {"band", "plane_gid", "wire", "tau", "value", "_meta"}
    assert set(d["coeff"]["_meta"]) >= {"gids", "n_wires", "band_lengths", "norm_sigma"}
    assert d["coeff"]["value"].shape == (len(events[0]["value"]), 1)
    assert d["ident"] == dict(run="run_000", source_file="cx_sensor_0000.h5", event=0)
    assert len({tuple(ds.get_data(i)["ident"].values()) for i in range(3)}) == 3


def test_dataset_collect(tmp_path):
    _write(tmp_path, [0, 1, 2])
    collect = dict(type="Collect", parts={"coeff": dict(
        keys=("band", "plane_gid", "wire", "tau", "value"), feat_keys=("value",),
        offset_keys_dict=dict(offset="band"))})
    out = _ds(tmp_path, transform=[collect])[0]
    n = int(out["coeff_offset"][-1])
    assert out["coeff_value"].shape[0] == n and out["coeff_band"].shape[0] == n


def test_plane_gid_beyond_255(tmp_path):
    _write(tmp_path, [0], gids=[300, 301])
    assert int(CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx").read_event(0)["plane_gid"].max()) >= 300


def test_multi_shard_indexing(tmp_path):
    """The global index crosses shard boundaries in order."""
    a = _write(tmp_path, [0, 1, 2], file_index=0, norm_sigma=NS, seed=1)
    b = _write(tmp_path, [3, 4], file_index=1, norm_sigma=NS, seed=2)
    ds = _ds(tmp_path)
    assert len(ds) == 5
    for i, want in enumerate(a + b):
        np.testing.assert_array_equal(ds.get_data(i)["coeff"]["band"], want["band"])


def test_full_chain_to_model_keys_at_batch_gt_1(tmp_path):
    """CoeffTPCDataset -> CoeffTokenize -> Collect -> collate_fn at batch 2: the model's keys,
    unprefixed (single-part Collect), cell-space keys sharing one row count."""
    from torch.utils.data import DataLoader
    from helix.model.tokenize import CoeffTokenize, PatchConfig
    from pimm_data.collate import collate_fn

    _write(tmp_path, [0, 1, 2, 3], seed=11, norm_sigma=NS)
    collect = dict(type="Collect", part="coeff", keys=(
        "band_id", "plane_id", "t_phys", "wire_pos", "wirefeat", "inp", "occ", "valid", "tgt"),
        offset_keys_dict=dict(offset="band_id"))
    ds = _ds(tmp_path, transform=[CoeffTokenize(part="coeff", cfg=PatchConfig()), collect])
    one = ds[0]
    assert all(k in one for k in ("band_id", "plane_id", "t_phys", "wire_pos", "wirefeat"))
    assert not any(k.startswith("coeff_") for k in one)
    b = next(iter(DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate_fn)))
    n_cells = b["band_id"].shape[0]
    assert b["plane_id"].shape[0] == n_cells and b["wirefeat"].shape == (n_cells, 1)
    assert b["inp"].shape[0] == n_cells and b["occ"].shape[0] == n_cells
    assert b["valid"].shape == b["inp"].shape


# ---- one corpus per reader ---------------------------------------------------------------

def test_norm_sigma_must_be_frozen_across_shards(tmp_path):
    """Row i of norm_sigma normalises any event in any shard; the reader serves shard 0's."""
    _write(tmp_path, [0, 1], file_index=0, norm_sigma=NS, seed=1)
    _write(tmp_path, [2, 3], file_index=1, norm_sigma=NS * 4.5, seed=2)
    with pytest.raises(ValueError, match="disagree on /config/norm_sigma"):
        _ds(tmp_path)
    ok = tmp_path / "ok"; ok.mkdir()
    _write(ok, [0, 1], file_index=0, norm_sigma=NS, seed=1)
    _write(ok, [2, 3], file_index=1, norm_sigma=NS, seed=2)
    np.testing.assert_allclose(_ds(ok).norm_sigma, NS)


def test_shards_disagreeing_on_plane_set_are_rejected(tmp_path):
    _write(tmp_path, [0, 1], file_index=0, seed=1)
    _write(tmp_path, [2, 3], file_index=1, seed=2, gids=[7, 9])
    with pytest.raises(ValueError, match="disagree on /config/gids"):
        _ds(tmp_path)


@pytest.mark.parametrize("change", [dict(removal_json=json.dumps(dict(kind="gate", kgate=3.0, tau=0.05))),
                                    dict(wavelet="coif3")])
def test_shards_of_another_DSP_are_rejected(tmp_path, change):
    """coeff_tpc and coeff_tpc_r1 share every table and differ only in the coherent gate (tau): only
    the DSP identity tells them apart. basis_digest covers the wavelet too."""
    _write(tmp_path, [0, 1], file_index=0, seed=1)
    _write(tmp_path, [2, 3], file_index=1, seed=2, basis_attrs={**BASIS_ATTRS, **change})
    with pytest.raises(ValueError, match="basis_digest") as ei:
        _ds(tmp_path)
    assert "different DSP" in str(ei.value)


def test_one_DSP_across_all_shards_is_accepted(tmp_path):
    _write(tmp_path, [0, 1], file_index=0, seed=1)
    _write(tmp_path, [2, 3], file_index=1, seed=2)
    assert len(_ds(tmp_path)) == 4


# ---- event identity ----------------------------------------------------------------------

def test_identity_is_source_file_plus_event_not_event_alone(tmp_path):
    """``/ident/event`` restarts at 0 in every source file, so a shard spanning two files repeats
    ids; keyed on the event alone, a 1000-event pilot came back as 800 joint events."""
    rng = np.random.default_rng(5)
    evs = []
    for src in ("cx_sensor_0000.h5", "cx_sensor_0001.h5"):
        for ev in (0, 1):
            e = _ev(rng, ev)
            e["source_file"] = src
            e["value"] = e["value"] + (0.0 if src.endswith("0000.h5") else 100.0)
            evs.append(e)
    write_coeff_shard(tmp_path / "cx_coeff_0000.h5", evs, band_lengths=BAND_LENGTHS, gids=GIDS,
                      n_wires=N_WIRES, basis_attrs=BASIS_ATTRS, dataset_name="cx", norm_sigma=NS)
    ds = _ds(tmp_path)
    assert len(ds) == 4, "duplicate ids collapsed"
    firsts = {round(float(ds.get_data(i)["coeff"]["value"][0, 0]), 5) for i in range(4)}
    assert len(firsts) == 4, f"events not distinct: {firsts}"


def test_identity_includes_run_so_runs_do_not_collide():
    """Every run restarts shard numbering at *_sensor_0000.h5, so (source_file, event) collides
    across runs; within a run the ids stay ordered (the index follows the data)."""
    from helix.data.dataset import _identity_ids
    src = np.array([b"sim_wire_sensor_0000.h5"] * 3)
    ev = np.array([0, 1, 2], np.int64)
    a = _identity_ids(src, ev, run=np.array([b"run_A"] * 3))
    b = _identity_ids(src, ev, run=np.array([b"run_B"] * 3))
    assert not set(a.tolist()) & set(b.tolist()), "two runs produced colliding ids"
    one = _identity_ids(np.array([b"s_0000.h5", b"s_0000.h5", b"s_0001.h5"]),
                        np.array([0, 1, 0], np.int64), run=np.array([b"r"] * 3))
    assert list(one) == sorted(one), "within-run ids must stay ordered"
    # a campaign task's file is not numbered: its events (campaign ids) still keep their order
    task = np.array([b"sim_wire_sensor_jaxtpc_wire_j000085-000089.h5"] * 3)
    ids = _identity_ids(task, np.array([17000, 17001, 17002], np.int64), run=np.array([b"j000000"] * 3))
    assert list(ids) == sorted(ids) and len(set(ids.tolist())) == 3
    other = _identity_ids(np.array([b"sim_wire_sensor_jaxtpc_wire_j000090-000094.h5"]),
                          np.array([17000], np.int64), run=np.array([b"j000000"]))
    assert other[0] != ids[0]


def test_reader_rejects_a_shard_with_duplicate_identities(tmp_path):
    """The id->position map would keep one row per duplicate: refuse rather than serve a quietly
    shorter dataset."""
    rng = np.random.default_rng(3)
    evs = [dict(_ev(rng, 7), source_file="cx_sensor_0000.h5") for _ in range(2)]
    write_coeff_shard(tmp_path / "cx_coeff_0000.h5", evs, band_lengths=BAND_LENGTHS,
                      gids=GIDS, n_wires=N_WIRES, basis_attrs=BASIS_ATTRS, dataset_name="cx")
    with pytest.raises(ValueError, match="does not uniquely identify"):
        CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx")


# ---- the holdout and the subsets ---------------------------------------------------------

HOLDOUT = dict(seed=0, fractions=dict(train=0.6, val=0.2, probe=0.2))


def _split_map(root, role):
    ds = _ds(root, holdout=HOLDOUT, split_role=role)
    return {tuple(ds._ident_at(i).values()) for i in ds.data_list}


def test_holdout_is_disjoint_total_and_survives_corpus_growth(tmp_path):
    """The split is keyed on identity, not position: one more shard renumbers every event after it,
    and a positional probe holdout would silently become trained-on events."""
    _write(tmp_path, list(range(40)), file_index=0, seed=0)
    before = {r: _split_map(tmp_path, r) for r in ("train", "val", "probe")}
    allv = [e for v in before.values() for e in v]
    assert len(allv) == len(set(allv)) == 40                        # disjoint AND complete
    _write(tmp_path, list(range(40, 80)), file_index=1, seed=1)
    after = {r: _split_map(tmp_path, r) for r in ("train", "val", "probe")}
    assert sum(len(v) for v in after.values()) == 80
    for role in before:
        assert before[role] <= after[role], f"{role}: events changed split when a shard was added"


def test_holdout_requires_a_role_and_valid_fractions(tmp_path):
    _write(tmp_path, list(range(40)))
    with pytest.raises(ValueError, match="split= was not"):
        _ds(tmp_path, holdout=HOLDOUT)
    with pytest.raises(ValueError, match="sum to 1.0"):
        _ds(tmp_path, holdout=dict(seed=0, fractions=dict(train=0.5, val=0.2)), split_role="train")


def test_n_events_is_a_fixed_nested_subset_in_corpus_order(tmp_path):
    """The label-efficiency control: a seeded random subset of the split, in corpus order, nested
    across sizes of one seed."""
    _write(tmp_path, list(range(10)))
    full = _ds(tmp_path).data_list
    sub = {n: _ds(tmp_path, n_events=n).data_list for n in (3, 6)}
    assert len(sub[3]) == 3 and len(sub[6]) == 6
    assert sub[3] == sorted(sub[3]) and set(sub[3]) <= set(sub[6]) <= set(full)
    assert _ds(tmp_path, n_events=3).data_list == sub[3]
    assert _ds(tmp_path, n_events=6, subset_seed=1).data_list != sub[6]
    with pytest.raises(ValueError, match="n_events"):
        _ds(tmp_path, n_events=11)


def test_max_event_size_drops_the_largest_events(tmp_path):
    _write(tmp_path, list(range(10)))
    sizes = CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx").event_sizes()
    cap = int(np.sort(sizes)[4])
    kept = [int(sizes[i]) for i in _ds(tmp_path, max_event_size=cap).data_list]
    assert kept == [int(s) for s in sizes if s <= cap]


# ---- event rejection ---------------------------------------------------------------------

def test_an_event_whose_plane_truth_is_off_the_plane_is_measured_by_its_lost_share():
    """_offplane_share: the largest share of a plane's truth charge valid_cells removed -- 0 when
    nothing went, ~1 when a plane's truth is wholly off it (the event is then rejected)."""
    from helix.data.cells import fkey, valid_cells
    from helix.data.dataset import _offplane_share
    keys = np.array([fkey(0, 10, 5), fkey(0, 11, 5), fkey(1, 5000, 5), fkey(1, 6000, 6)], np.int64)
    rec = dict(key=keys, q=np.array([10.0, 10.0, 7.0, 3.0], np.float32), dom_frac=np.ones(4, np.float32),
               dom_y=np.zeros(4, np.float32), dom_z=np.zeros(4, np.float32),
               ct_cell=np.arange(4, dtype=np.int32), ct_track=np.zeros(4, np.int32), ct_q=np.ones(4, np.float32))
    after = valid_cells(rec, gids=[0, 1], n_wires=[400, 400], n_ticks=4336)
    assert _offplane_share(rec, after) == 1.0                      # plane 1 lost everything
    assert _offplane_share(rec, rec) == 0.0
    # the training read (key and q only) drops the same cells
    kq = valid_cells(dict(key=keys, q=rec["q"]), gids=[0, 1], n_wires=[400, 400], n_ticks=4336)
    assert set(kq) == {"key", "q"} and np.array_equal(kq["key"], after["key"])
