"""The extraction cache: a preemption must not cost the whole loop.

The probe extracts 2.5M patches over 388 events before it fits anything, and a
run that died 40 minutes in discarded all of it. These tests drive the cache's
own functions — the extraction loop needs a GPU, a corpus and a truth artifact,
so what is checkable here is the part that was actually load-bearing: that a
resumed run reconstructs EXACTLY the packs an uninterrupted one held, that the
key changes when the features would, and that a gap is re-extracted rather than
silently skipped.
"""
import importlib.util
import os
import pathlib

import numpy as np
import pytest


def _rp():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_probe.py"
    spec = importlib.util.spec_from_file_location("_run_probe_cache_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pack(ev, n=7, dim=5, rng=None):
    rng = rng or np.random.default_rng(ev)
    return dict(y=rng.normal(size=n).astype(np.float32),
                geo=rng.normal(size=(n, 8)).astype(np.float32),
                cells=rng.integers(0, 50, size=(n, 3)).astype(np.int64),
                plane=rng.integers(0, 6, size=n).astype(np.int64),
                n_pixels=rng.integers(1, 9, size=n).astype(np.int64),
                n_dom=rng.integers(0, 4, size=n).astype(np.int64),
                multiplane=rng.integers(0, 2, size=n).astype(bool),
                n_bands=4,
                event=np.full(n, ev),
                tick=rng.normal(size=n), wire=rng.normal(size=n),
                Xtr=rng.normal(size=(n, dim)).astype(np.float32),
                Xrn=rng.normal(size=(n, dim)).astype(np.float32),
                Xraw=rng.normal(size=(n, dim)).astype(np.float32))


def test_a_resumed_run_reconstructs_what_it_had_EXACTLY(tmp_path):
    """Bit-exact, every array. A resumed run must be the same run.

    This was float16 for the feature arms, and it cost accuracy that showed:
    an end-to-end resume reproduced `trained`, `geo` and `random` to the printed
    four decimals but moved `raw` from +0.0044 to +0.0045. That is ~1e-4, well
    under the probe's own seed sigma of 0.0021 -- and still wrong, because it
    made `cache_resumed_events` a field that moves the number.
    """
    rp = _rp()
    d = str(tmp_path / "c")
    original = [_pack(i) for i in range(6)]
    rp._cache_write(d, 0, 3, original[:3])
    rp._cache_write(d, 3, 6, original[3:])

    packs, nxt = rp._cache_resume(d)
    assert nxt == 6 and len(packs) == 6
    for got, want in zip(packs, original):
        assert set(got) == set(want)
        for k, v in want.items():
            np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(v), err_msg=k)


def test_cache_half_is_opt_in_and_keyed_apart(tmp_path):
    """Half precision stays available, but cannot be mistaken for a full cache."""
    rp = _rp()
    d = str(tmp_path / "c")
    original = [_pack(0)]
    rp._cache_write(d, 0, 1, original, half=True)
    got = rp._cache_resume(d)[0][0]

    assert got["Xtr"].dtype == np.float32, "must widen before the fit"
    assert not np.array_equal(got["Xtr"], original[0]["Xtr"]), (
        "if half storage were lossless there would be no reason for the flag")
    np.testing.assert_allclose(got["Xtr"], original[0]["Xtr"], rtol=1e-3, atol=1e-3)
    # non-arm fields are never downcast, whatever the flag
    np.testing.assert_array_equal(got["y"], original[0]["y"])

    base = dict(trained="aa", random="bb", layer=12, cell_t="grid_center",
                pw=16, pt=8, n_bands=4, corpus="/c", dataset_name="sim_wire",
                truth="/t", corpus_ident="dd", dom_threshold=0.5)
    assert rp._cache_key(**base, half=True) != rp._cache_key(**base, half=False), (
        "a half cache and a full one hold different numbers; reading one as the "
        "other would silently change a result")


def test_a_gap_is_re_extracted_not_skipped(tmp_path, capsys):
    """Skipping the hole would give a row whose n_events lies about its features."""
    rp = _rp()
    d = str(tmp_path / "c")
    rp._cache_write(d, 0, 2, [_pack(0), _pack(1)])
    rp._cache_write(d, 4, 6, [_pack(4), _pack(5)])      # 2..4 missing

    packs, nxt = rp._cache_resume(d)
    assert nxt == 2 and len(packs) == 2
    assert "gap at event 2" in capsys.readouterr().out
    # and the trailing chunk is kept, not deleted: refilling the gap recovers it
    assert len(rp._cache_chunks(d)) == 2


def test_an_empty_chunk_still_advances_the_floor(tmp_path):
    """Events that yield no patches are PROCESSED; the resume floor must say so."""
    rp = _rp()
    d = str(tmp_path / "c")
    rp._cache_write(d, 0, 2, [_pack(0)])
    rp._cache_write(d, 2, 5, [])                        # three barren events
    rp._cache_write(d, 5, 6, [_pack(5)])
    packs, nxt = rp._cache_resume(d)
    assert nxt == 6 and len(packs) == 2


def test_the_key_changes_with_everything_that_changes_the_features():
    rp = _rp()
    base = dict(trained="aa", random="bb", layer=12, cell_t="grid_center",
                pw=16, pt=8, n_bands=4, corpus="/c", dataset_name="sim_wire",
                truth="/t", corpus_ident="dd", dom_threshold=0.5, half=False)
    k0 = rp._cache_key(**base)
    assert k0 == rp._cache_key(**base), "the key must be deterministic"
    for field, other in [("trained", "zz"), ("random", "zz"), ("layer", 6),
                         ("cell_t", "centroid"), ("pw", 8), ("pt", 16),
                         ("corpus", "/other"), ("truth", "/other"),
                         ("corpus_ident", "zz"), ("dom_threshold", 0.9),
                         ("half", True)]:
        assert rp._cache_key(**dict(base, **{field: other})) != k0, field


def test_max_events_is_not_in_the_key():
    """A 400-event run must reuse a 100-event run's chunks.

    Events are cached by their index in the truth artifact's order, and
    `corpus_ident` already pins that order, so the count is not part of identity.
    """
    rp = _rp()
    import inspect as _inspect
    src = _inspect.getsource(rp.main)
    call = src[src.index("_cache_key("):src.index("packs, start = _cache_resume")]
    assert "max_events" not in call and "n_ev" not in call


def test_a_half_written_chunk_is_never_visible(tmp_path, monkeypatch):
    """The write must be atomic: a truncated .npz is worse than an absent one."""
    rp = _rp()
    d = str(tmp_path / "c")
    real = np.savez

    def boom(path, **kw):
        real(path, **kw)
        raise KeyboardInterrupt        # died between write and rename

    monkeypatch.setattr(rp.np, "savez", boom)
    with pytest.raises(KeyboardInterrupt):
        rp._cache_write(d, 0, 3, [_pack(0)])
    assert rp._cache_chunks(d) == []
    assert rp._cache_resume(d) == ([], 0)
    # the temp file is there, and it is NOT named like a chunk
    assert all(f.startswith(".") for f in os.listdir(d))
