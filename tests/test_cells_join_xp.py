"""The fine-cell joins (helix.model.cells_join, and the key packing under them) give the same answer
on numpy and on torch (helix.core.xp): the training labels build on numpy in DataLoader workers,
the denoiser's inference on the event's device, through ONE implementation.

Pinned, on torch CPU and (when present) CUDA against numpy: ``covering_tokens``, ``cell_inputs``'
``idx`` and ``cell_neighbors`` bit-exact, ``aux`` to float32 rounding, ``footprint_cells`` (the
natural covered cells) bit-exact -- on synthetic events at the 2 x 16 and 1 x 4 grids and on a real
event of ``HELIX_CORPUS`` at both grids with the production patch geometry. And the predictor
(helix.eval.denoise.load_denoiser) returns what its head gives on the numpy inputs, in one chunk or
many.
"""
import glob
import os

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from _fixtures import CORPUS                                         # noqa: E402
from helix.data import cells as C                                    # noqa: E402
from helix.eval.denoise import footprint_cells, sub_cells            # noqa: E402
from helix.model import cells_join as J                              # noqa: E402
from helix.model.tokenize import PatchConfig, assemble, patch_config, pixel_cells, tick_of_tau  # noqa: E402
from test_eval import BL, TOK, _sample                               # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
GRIDS = [C.Grid(2, 16), C.Grid(1, 4)]


def _tokens(coeff, cfg, band_lengths, gids, n_wires):
    """The tokenized event's per-token cell keys (values do not enter them)."""
    t = assemble(coeff["band"], coeff["plane_gid"], coeff["wire"], coeff["tau"],
                 np.zeros(len(coeff["band"]), np.float32), gids=gids, n_wires=n_wires,
                 band_lengths=band_lengths, norm_sigma=np.ones((len(gids), 5), np.float32), cfg=cfg)
    return np.asarray(t["cell_key"], np.int64)


def _query_keys(ck, bl, cfg, grid, extra, rng, n_max=None):
    """Cells to join: the natural covered cells, ``extra`` (truth cells), cells beside and far from
    any token, and the plane's first wire / tick (where a neighbour block is -1)."""
    nat = footprint_cells(ck, bl, cfg, grid)
    g, w, t = C.unkey(nat)
    off = C.fkey(g, np.maximum(w + rng.integers(-40, 40, len(w)), 0), np.maximum(t + rng.integers(-60, 60, len(t)), 0))
    edge = C.fkey(np.repeat(np.unique(g), 3), np.tile([0, 0, 1], len(np.unique(g))), np.tile([0, 1, 0], len(np.unique(g))))
    keys = np.unique(np.concatenate([nat, np.asarray(extra, np.int64), off, edge]))
    if n_max and len(keys) > n_max:
        keys = np.sort(rng.choice(keys, n_max, replace=False))
    return keys


def _same_everywhere(keys, ck, bl, cfg, grid):
    """Every join on numpy and on each torch device; asserts they agree."""
    idx, aux = J.cell_inputs(keys, ck, bl, cfg, grid)
    nbr = J.cell_neighbors(keys, ck, bl, cfg, grid)
    cov = J.covering_tokens(keys, ck, bl, cfg, grid)
    nat = footprint_cells(ck, bl, cfg, grid)
    g, w, t = C.unkey(keys)
    pc = pixel_cells(g, w * grid.fw, t * grid.ft + grid.ft // 2, bl, cfg)
    assert idx.dtype == nbr.dtype == cov.dtype == np.int64 and aux.dtype == np.float32
    assert (idx >= 0).any() and (~(idx >= 0)).any() and (nbr >= 0).any() and (nbr < 0).any()
    assert np.array_equal(cov, idx) and np.array_equal(nbr[:, :, nbr.shape[2] // 2], idx)
    rep, group = J.cell_groups(keys, bl, cfg, grid)                 # a neighbourhood per group, exactly
    assert rep.dtype == group.dtype == np.int64 and len(rep) < len(keys)
    assert (group[rep] == np.arange(len(rep))).all() and (rep == np.unique(group, return_index=True)[1]).all()
    assert np.array_equal(pc[rep][group], pc) and len(np.unique(pc[rep], axis=0)) == len(rep)   # the patches' partition
    assert np.array_equal(J.cell_neighbors(keys[rep], ck, bl, cfg, grid)[group], nbr)
    assert np.array_equal(J.covering_tokens(keys[rep], ck, bl, cfg, grid)[group], cov)
    for dev in DEVICES:
        T = lambda a: torch.as_tensor(a, device=dev)                  # noqa: E731
        ti, ta = J.cell_inputs(T(keys), T(ck), bl, cfg, grid)
        assert ti.device.type == dev and ti.dtype == torch.int64 and ta.dtype == torch.float32
        assert np.array_equal(ti.cpu().numpy(), idx), dev
        np.testing.assert_allclose(ta.cpu().numpy(), aux, rtol=0, atol=2e-6, err_msg=dev)
        assert np.array_equal(J.cell_neighbors(T(keys), T(ck), bl, cfg, grid).cpu().numpy(), nbr), dev
        assert np.array_equal(J.covering_tokens(T(keys), T(ck), bl, cfg, grid).cpu().numpy(), cov), dev
        tr, tg = J.cell_groups(T(keys), bl, cfg, grid)
        assert np.array_equal(tr.cpu().numpy(), rep) and np.array_equal(tg.cpu().numpy(), group), dev
        assert np.array_equal(footprint_cells(T(ck), bl, cfg, grid).cpu().numpy(), nat), dev
        tg, tw, tt = C.unkey(T(keys))
        assert np.array_equal(pixel_cells(tg, tw * grid.fw, tt * grid.ft + grid.ft // 2, bl, cfg).cpu().numpy(), pc)
        tau = np.arange(-3, 900)
        gg, bb = np.arange(len(tau)) % 6, np.arange(len(tau)) % 4
        assert np.array_equal(tick_of_tau(T(tau), T(gg), T(bb), cfg).cpu().numpy(), tick_of_tau(tau, gg, bb, cfg))
    return len(keys), int((idx >= 0).any(1).sum())


@pytest.mark.parametrize("grid", GRIDS, ids=str)
def test_the_joins_agree_on_numpy_and_torch_synthetic(grid):
    cfg = patch_config(TOK)
    s = _sample(np.random.default_rng(3), 0, grid=grid)
    c, m = s["coeff"], s["coeff"]["_meta"]
    ck = _tokens(c, cfg, m["band_lengths"], m["gids"], m["n_wires"])
    keys = _query_keys(ck, BL, cfg, grid, s["cells"]["key"], np.random.default_rng(1))
    _same_everywhere(keys, ck, BL, cfg, grid)


def _real_event():
    if not os.path.isdir(CORPUS):
        pytest.skip(f"corpus absent: {CORPUS or '(HELIX_CORPUS unset)'}")
    shards = sorted(glob.glob(os.path.join(CORPUS, "*_coeff_[0-9]*.h5")))
    if not shards or not os.path.exists(C.cells_path(shards[0])):
        pytest.skip(f"{CORPUS}: no coeff shard with its cells truth")
    with h5py.File(shards[0], "r") as f, h5py.File(C.cells_path(shards[0]), "r") as fc:
        cfg = f["config"]
        meta = dict(gids=cfg["gids"][:], n_wires=cfg["n_wires"][:], band_lengths=cfg["band_lengths"][:].astype(np.int64))
        a, b = (int(x) for x in f["coord/event_offset"][:2])
        coeff = {k: f[f"coord/{k}"][a:b] for k in ("band", "plane_gid", "wire", "tau")}
        return meta, coeff, C.read_event(fc, 0)


@pytest.mark.parametrize("grid", GRIDS, ids=str)
def test_the_joins_agree_on_numpy_and_torch_real(grid):
    """The production geometry on a real event; the 1 x 4 truth keys are the 2 x 16 truth's
    sub-cells. Up to 400k cells, sampled."""
    meta, coeff, rec = _real_event()
    cfg = PatchConfig()
    ck = _tokens(coeff, cfg, meta["band_lengths"], meta["gids"], meta["n_wires"])
    assert len(np.unique(ck)) == len(ck) > 1000
    truth = rec["key"] if grid == rec["grid"] else sub_cells(rec["key"], rec["grid"], grid).ravel()
    keys = _query_keys(ck, meta["band_lengths"], cfg, grid, truth, np.random.default_rng(2), n_max=400_000)
    n, covered = _same_everywhere(keys, ck, meta["band_lengths"], cfg, grid)
    assert covered > n // 4


# ------------------------------------------------------------------------------------ predictor
def _reference(model, ev, keys, cfg):
    """What the predictor returned before it ran on the device: numpy inputs, one head call."""
    from helix.eval.denoise import UNCOVERED_LOGIT, head_charge
    from helix.eval.recon import autocast
    idx, aux = J.cell_inputs(keys, ev.cell_key, ev.band_lengths, cfg, ev.grid)
    ok = (idx >= 0).any(1)
    nbr = J.cell_neighbors(keys[ok], ev.cell_key, ev.band_lengths, cfg, ev.grid)
    dev = next(model.parameters()).device
    t = lambda a: torch.as_tensor(a, device=dev)                     # noqa: E731
    y, lg = np.zeros(len(keys), np.float32), np.full(len(keys), UNCOVERED_LOGIT, np.float32)
    qm, sd = np.zeros(len(keys)), np.full(len(keys), np.nan)
    with torch.no_grad(), autocast(dev):
        out = model.head(ev.features(model.fm), t(idx[ok]), t(aux[ok]), t(nbr), ev.batch)
    y[ok], lg[ok] = out[0].float().cpu().numpy(), out[1].float().cpu().numpy()
    m, s = head_charge(model.head, out)
    qm[ok], sd[ok] = m.cpu().numpy(), s.cpu().numpy()
    return y, lg, qm, sd, ok


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("chunk", [None, 997])
@pytest.mark.parametrize("dist", ["cat", "gauss"])
def test_the_predictor_returns_its_heads_answer_on_the_numpy_inputs(tmp_path, dev, chunk, dist):
    """The predictor's four outputs are the head's on the numpy-built inputs, per cell: to float32
    rounding on a CPU, in one chunk or many. On CUDA to bf16 rounding: the head runs under bf16
    autocast there, and its matmuls round by the kernel their shapes pick -- the batch size, and
    the predictor's cells grouped by neighbourhood (``CellDecoder``'s ``group``) where the
    reference's are one per row."""
    from helix.eval.denoise import covered_cells, load_denoiser
    from helix.eval.events import events
    from test_denoise import _write_denoise_run
    from test_eval import _DS, _write_run

    fm = tmp_path / "fm"
    _write_run(fm)
    _write_denoise_run(str(tmp_path / "dn"), str(fm), dist=dist, cond_charge=True)
    s = _sample(np.random.default_rng(3), 0)
    predict, cfg = load_denoiser(str(tmp_path / "dn"), dev, chunk=chunk)
    ev = next(events(_DS([s]), cfg, torch.device(dev)))
    K = covered_cells(ev, cfg)
    keys = np.unique(np.concatenate([K, s["cells"]["key"], K[:50] + (100 << 40)]))   # + uncovered (no plane)
    y, lg, qm, sd = predict(ev, keys)
    ry, rl, rq, rs, ok = _reference(predict.run.model, ev, keys, cfg)
    assert ok.any() and (~ok).any() and ok[np.isin(keys, K)].all() and (chunk is None or ok.sum() > 5 * chunk)
    assert y.dtype == lg.dtype == np.float32 and qm.dtype == sd.dtype == np.float64
    assert (lg[~ok] == rl[~ok]).all() and (y[~ok] == 0).all() and (qm[~ok] == 0).all() and np.isnan(sd[~ok]).all()
    tol = 1e-2 if dev == "cuda" else 2e-5
    for a, b in ((y, ry), (lg, rl), (qm, rq), (sd, rs)):
        np.testing.assert_allclose(a[ok], b[ok], rtol=tol, atol=tol)
