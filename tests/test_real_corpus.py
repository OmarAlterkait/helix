"""The built corpus and its truth, read directly: the joins every number depends on, on real events.

Synthetic fixtures pin each piece; these pin that the pieces meet on the production data -- the
corpus's wire numbering is the geometry's and the truth's, and the evaluation's windows come out
whole on real events. They read the first events of the first shard of ``HELIX_CORPUS`` and its
``cells`` truth (scripts/build_truth.py), and skip, saying which, when either is absent.
"""
import glob
import os

import h5py
import numpy as np
import pytest

from _fixtures import CORPUS
from helix.data import cells as C

N_EVENTS = 3


def _first_shard():
    if not os.path.isdir(CORPUS):
        pytest.skip(f"corpus absent: {CORPUS or '(HELIX_CORPUS unset)'}")
    shards = sorted(glob.glob(os.path.join(CORPUS, "*_coeff_[0-9]*.h5")))
    if not shards:
        pytest.skip(f"{CORPUS}: no coeff shards")
    cells = C.cells_path(shards[0])
    if not os.path.exists(cells):
        pytest.skip(f"{cells}: no cells truth beside the corpus (scripts/build_truth.py)")
    return shards[0], cells


def _events():
    """Per event: the shard's planes and per-coefficient columns, and its cells record."""
    coeff, cells = _first_shard()
    with h5py.File(coeff, "r") as f, h5py.File(cells, "r") as fc:
        C.require_format(fc["config"], cells)
        assert (C.read_ident(fc)["event"][:N_EVENTS] == f["ident/event"][:N_EVENTS]).all()
        cfg = f["config"]
        meta = dict(gids=cfg["gids"][:], n_wires=cfg["n_wires"][:], n_ticks=int(cfg.attrs["n_ticks_raw"]))
        off = f["coord/event_offset"]
        for ev in range(N_EVENTS):
            a, b = int(off[ev]), int(off[ev + 1])
            cols = {k: f[f"coord/{k}"][a:b] for k in ("band", "plane_gid", "wire", "tau")}
            cols["value"] = f["value"][a:b]
            yield meta, cols, C.read_event(fc, ev)


def test_the_corpus_wires_are_the_geometrys_and_the_truths():
    """Per plane: the shard's wire count is the packaged geometry's, the truth lies on the plane,
    and the truth's charge per wire lines up with the kept coefficients' at zero offset -- better
    than at any shift up to 30 fine wires, mirrored, or against another plane of the event. Along
    the wire only: the drift-time join is helix.model.tokenize's ``tick_of_tau``, tested there."""
    from helix.tpc.readout import load_plane_registry
    reg = load_plane_registry("cubic_wireplane_geometry.json")
    for meta, c, rec in _events():
        gids, nw = meta["gids"], meta["n_wires"]
        assert {int(g): int(n) for g, n in zip(gids, nw)} == {g: reg[g]["n_wires"] for g in map(int, gids)}
        assert meta["n_ticks"] == reg[int(gids[0])]["n_ticks"]
        G, fw, ft = C.unkey(np.asarray(rec["key"]))
        FW, FT = rec["grid"].fw, rec["grid"].ft                      # the truth file's own grid
        q = np.asarray(rec["q"], np.float64)
        prof = {}
        for g, n in zip(gids.tolist(), nw.tolist()):
            nf = (n + 1) // FW
            on = G == g
            assert on.any(), f"plane {g}: no truth charge"
            assert fw[on].max() < nf and ft[on].min() >= 0 and ft[on].max() * FT < meta["n_ticks"]
            sel = c["plane_gid"] == g
            assert c["wire"][sel].max() < n
            prof[g] = (np.bincount(fw[on], weights=q[on], minlength=nf),
                       np.bincount(c["wire"][sel] // FW, weights=np.abs(c["value"][sel]), minlength=nf))
        for g, (truth, kept) in prof.items():
            r = lambda x: np.corrcoef(x, kept)[0, 1]                       # noqa: E731
            r0 = r(truth)
            assert r0 > 0.7, f"plane {g}: truth and kept coefficients correlate {r0:.2f} along the wire"
            shifted = max(r(np.roll(truth, s)) for s in range(-30, 31) if s)
            others = [r(t) for h, (t, _) in prof.items() if h != g and len(t) == len(truth)]
            assert r0 > max([shifted, r(truth[::-1])] + others), f"plane {g}: not aligned at zero offset"


def test_the_evaluation_windows_come_out_whole_on_real_events():
    """helix.eval.events.event_truth on real events: every plane gets its far and near background,
    neither holds truth charge, every iso window holds its particle with a real energy, the three
    contexts all occur, and the draw is reproducible."""
    from helix.eval.events import N_BG_PLANE, cell_charge, event_truth
    from helix.model.cells_join import coeff_cells
    from helix.model.tokenize import PatchConfig
    cfg, ctxs = PatchConfig(), set()
    for i, (meta, c, rec) in enumerate(_events()):
        ck = coeff_cells(c["band"], c["plane_gid"], c["wire"], c["tau"], cfg)
        t = event_truth(rec, ck, np.random.default_rng([0, i]))
        again = event_truth(rec, ck, np.random.default_rng([0, i]))
        assert np.array_equal(t["wkey"], again["wkey"]) and t["meta"] == again["meta"]
        meta_w = t["meta"]
        wq = cell_charge(rec, t["wkey"]).reshape(len(meta_w), -1).sum(1)
        for kind in ("bg", "near"):
            planes = [m["g"] for m in meta_w if m["kind"] == kind]
            assert sorted(set(planes)) == sorted(meta["gids"].tolist()), (kind, sorted(set(planes)))
            assert max(planes.count(g) for g in set(planes)) <= N_BG_PLANE
            assert all(q == 0 for q, m in zip(wq, meta_w) if m["kind"] == kind), kind
        iso = [(q, m) for q, m in zip(wq, meta_w) if m["kind"] in ("iso", "isox")]
        assert iso and all(q > 0 and np.isfinite(m["E"]) and m["E"] > 0 for q, m in iso)
        ctxs |= {m["ctx"] for _, m in iso}
    assert ctxs == {"clear", "crowded", "mixed"}
