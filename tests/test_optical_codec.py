"""Optical codec (helix.optical.codec): exact integer Haar, pulses kept, backends agree, the
shard and the tokens round-trip."""
import numpy as np
import pytest

from helix.optical import codec, tokens
from helix.optical.codec import OpticalChain


def _template(system="pmt"):
    t = codec.load_templates()[0][system]
    return t / np.abs(t).max(), int(np.argmax(np.abs(t)))


def _wf(n_wf=24, seed=0, lengths=(5000, 1500, 700), amp=40.0, sigma=3.0, system="pmt", pol=-1, base=14250):
    """Windows with one pulse each (peak at ``pulses[k]``), Gaussian noise; channels repeat (3 windows each)."""
    rng = np.random.default_rng(seed)
    t, tp = _template(system)
    bufs, offs, lens, pulses, off = [], [], [], [], 0
    for k in range(n_wf):
        n = lengths[k % len(lengths)]
        x = base + rng.normal(0, sigma, n)
        p = int(rng.integers(150, n - 150))
        x[p - tp:p - tp + len(t)] += pol * amp * t
        bufs.append(np.round(x).astype(np.uint16)); offs.append(off); lens.append(n); pulses.append(p)
        off += n
    return dict(channel=np.arange(n_wf, dtype=np.int32) // 3, t_us=np.arange(n_wf, dtype=np.float64),
                length=np.array(lens), offset=np.array(offs), buf=np.concatenate(bufs)), pulses


SUPPORT = {"rule": "support"}                     # the support rule (X-ARAPUCA's; PMT's before "deconv")


def _raw(wf, k):
    o, n = wf["offset"][k], wf["length"][k]
    return wf["buf"][o:o + n].astype(np.float64)


def test_integer_haar_is_exact():
    x = np.random.default_rng(1).integers(0, 16384, (5, 256)).astype(np.int64)
    s, d = codec.haar(x, 6)
    assert s.shape == (5, 4) and [a.shape[1] for a in d] == [4, 8, 16, 32, 64, 128]
    np.testing.assert_array_equal(codec.unhaar(s, d), x)


def test_pulses_are_in_the_support_and_lossless_there():
    wf, pulses = _wf()
    ch = OpticalChain(codec.make_specs({"pmt": {"q": 0, **SUPPORT}}))
    res = ch.process("pmt", wf)
    rec = ch.reconstruct(res)
    for g in res["groups"]:
        roi = np.asarray(g["roi"])
        for r, k in enumerate(g["index"]):
            n = wf["length"][k]
            assert roi[r, pulses[k] - 4:pulses[k] + 12].all()
            np.testing.assert_array_equal(rec[k][roi[r, :n]], _raw(wf, k)[roi[r, :n]])
    assert res["n_samples"] / res["n_kept"] > 5


def test_default_keeps_the_charge():
    """Against the truth (the injected pulse), not the raw window (raw = truth + noise): the
    decoded charge is as close to the truth as the raw samples are."""
    t, tp = _template()
    ch = OpticalChain()
    for seed in (2, 3):
        wf, pulses = _wf(seed=seed)
        rec = ch.reconstruct(ch.process("pmt", wf))
        truth = 40.0 * t[tp - 4:tp + 20].sum()
        e_rec = np.array([(14250 - rec[k][p - 4:p + 20]).sum() / truth - 1 for k, p in enumerate(pulses)])
        e_raw = np.array([(14250 - _raw(wf, k)[p - 4:p + 20]).sum() / truth - 1 for k, p in enumerate(pulses)])
        assert np.sqrt(np.mean(e_rec ** 2)) <= 1.1 * np.sqrt(np.mean(e_raw ** 2)) and np.abs(e_rec).max() < 0.25


def test_deconv_keeps_slow_dense_light():
    """A flash (300 photons, 30% within ~6 ns, 70% with a 1.5 us decay) of the full response on
    white noise: the deconvolution rule's decoded window is closer to the truth than the
    support rule's and than the raw window, with far fewer values than raw."""
    rng = np.random.default_rng(10)
    h = codec.load_templates()[0]["pmt_response"]; hp = int(np.argmax(h))
    n, m = 5000, 8
    truth = np.zeros((m, n))
    for r in range(m):
        dt = np.where(rng.uniform(size=300) < 0.3, rng.exponential(3.0, 300), rng.exponential(750.0, 300))
        for d in (500 + dt).astype(int):
            if d + len(h) - hp < n:
                truth[r, d - hp:d - hp + len(h)] += rng.normal(12.4, 3.0) * h
    raw = np.round(8000 - truth + rng.normal(0, 3.0, (m, n)))
    wf = dict(channel=np.arange(m, dtype=np.int32), t_us=np.zeros(m), length=np.full(m, n), offset=np.arange(m) * n,
              buf=raw.astype(np.uint16).ravel())
    err = {}
    for name, spec in (("deconv", None), ("support", codec.make_specs({"pmt": SUPPORT}))):
        ch = OpticalChain(spec)
        res = ch.process("pmt", wf)
        rec = np.stack(ch.reconstruct(res))
        err[name] = np.mean((rec - (8000 - truth)) ** 2)
        if name == "deconv":
            assert res["n_kept"] < 0.25 * m * n
    err["raw"] = np.mean((raw - (8000 - truth)) ** 2)
    assert err["deconv"] < err["support"] and err["deconv"] < 0.5 * err["raw"]


def test_q_only_on_the_fine_levels_and_coarse_sums_exact():
    wf, _ = _wf(seed=3)
    res = OpticalChain(codec.make_specs({"pmt": SUPPORT})).process("pmt", wf)
    spec = res["spec"]
    for g in res["groups"]:
        roi = np.asarray(g["roi"]); m = roi.shape[0]
        for li, (d, s) in enumerate(zip(g["details"], g["stored"])):
            j = spec["level"] - li
            cand = roi.reshape(m, d.shape[1], -1).any(-1)
            assert not np.any(s & ~cand)
            if j > spec["q_levels"]:
                np.testing.assert_array_equal(s, cand)
        # every window's coarse sums are its block sums
        k = g["index"][0]; n = wf["length"][k]; raw = _raw(wf, k)
        nb = n // 64
        np.testing.assert_array_equal(np.asarray(g["coarse"])[0, :nb], raw[:nb * 64].reshape(nb, 64).sum(1))


def test_gap_closing():
    t, tp = _template()
    x = np.full((2, 1024), 2000.0)
    for p in (300, 300 + 12 + 30):                     # trigger regions 30 samples apart: closed
        x[0, p - tp:p - tp + len(t)] -= 60 * t
    for p in (300, 300 + 12 + 80):                     # 80 apart: not
        x[1, p - tp:p - tp + len(t)] -= 60 * t
    x += np.random.default_rng(4).normal(0, 3, x.shape)
    wf = dict(channel=np.zeros(2, np.int32), t_us=np.zeros(2), length=np.full(2, 1024), offset=np.array([0, 1024]),
              buf=np.round(x).astype(np.uint16).ravel())
    roi = np.asarray(OpticalChain(codec.make_specs({"pmt": SUPPORT})).process("pmt", wf)["groups"][0]["roi"])
    runs = [int((np.diff(np.r_[0, r[200:600].astype(int)]) == 1).sum()) for r in roi]   # noise triggers elsewhere
    assert runs[0] == 1 and runs[1] == 2


def test_xarapuca_delta_template_finds_small_pulses():
    wf, pulses = _wf(seed=5, lengths=(768,), amp=4.0, sigma=0.46, system="xarapuca", pol=1, base=2000)
    res = OpticalChain(codec.make_specs({"xarapuca": SUPPORT})).process("xarapuca", wf)
    g = res["groups"][0]
    found = np.mean([np.asarray(g["roi"])[r, pulses[k]] for r, k in enumerate(g["index"])])
    assert found > 0.9
    assert np.all(np.isfinite(res["noise"]["level"])) and np.all(res["noise"]["level"] > 0)


def test_matched_filter_lines_up_with_the_pulse_peak():
    t, tp = _template()
    u, lag = codec._filter(codec.load_templates()[0]["pmt"])
    z = np.zeros((1, 512), np.float32); z[0, 200 - tp:200 - tp + len(t)] = t
    assert abs(int(np.argmax(codec._correlate(z, u, lag))) - 200) <= 1


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    wf, _ = _wf(seed=6)
    ref = OpticalChain().process("pmt", wf)
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        got = OpticalChain(device=dev).process("pmt", wf)
        assert abs(got["n_kept"] - ref["n_kept"]) <= max(3, 1e-3 * ref["n_kept"])
        np.testing.assert_allclose(got["noise"]["level"], ref["noise"]["level"], rtol=1e-3)
        for gr, gg in zip(ref["groups"], got["groups"]):
            assert np.mean(np.asarray(gr["roi"]) != gg["roi"].cpu().numpy()) < 1e-3
            np.testing.assert_array_equal(gr["coarse"], gg["coarse"].cpu().numpy())
            for a, b in zip(gr["details"], gg["details"]):
                np.testing.assert_array_equal(a, b.cpu().numpy())


def _shard(tmp_path, res):
    from helix.tpc import sparse_io
    from helix.tpc.chain import TPCChain
    p = tmp_path / "o.h5"
    w = sparse_io.Writer(p, TPCChain())
    w.add_optical(res); w.add_optical(res)              # two events
    w.close()
    return p


@pytest.mark.parametrize("dev", [None, "cpu"])
def test_shard_roundtrip(tmp_path, dev):
    if dev is not None:
        pytest.importorskip("torch")
    from helix.tpc.sparse_io import read_optical
    wf, _ = _wf(seed=7)
    ch = OpticalChain(device=dev)
    res = ch.process("pmt", wf)
    rec = ch.reconstruct(res)
    p = _shard(tmp_path, res)
    for ev in (0, 1):
        got = read_optical(p, "pmt", ev)
        assert len(got) == len(wf["length"])
        for k, g in enumerate(got):
            assert g["channel"] == wf["channel"][k] and g["length"] == wf["length"][k]
            x = codec.unhaar(g["coarse"][None], [d[None] for d in g["details"]], [s[None] for s in g["stored"]])[0]
            np.testing.assert_array_equal(x[:g["length"]], rec[k])
            assert np.isfinite(g["sigma_stat"]) and np.all(g["sigma_level"] > 0)


def test_stored_blocks_are_shard_candidates(tmp_path):
    """Every stored difference is a candidate of the support the shard encodes (blocks touching
    it) - also a coarse block at a window's end whose light runs into the end (the block's
    middle lies past the window) - and the shard decodes to the same waveforms."""
    from helix.tpc import sparse_io
    from helix.tpc.chain import TPCChain
    from helix.tpc.sparse_io import read_optical
    h = codec.load_templates()[0]["pmt_response"]; hp = int(np.argmax(h))
    rng = np.random.default_rng(0); m = 64
    lens = rng.integers(700, 5200, m); bufs = []
    for n in lens:
        x = 14250 + rng.normal(0, 3.0, n)
        t0 = int(rng.integers(max(n - 400, 0), n - 5))                     # slow light running into the window end
        for d in (t0 + rng.exponential(150, int(rng.integers(5, 60)))).astype(int):
            if d - hp >= 0:
                seg = h[:max(0, min(len(h), n - (d - hp)))]; x[d - hp:d - hp + len(seg)] -= rng.normal(12, 3) * seg
        bufs.append(np.round(x).astype(np.uint16))
    wf = dict(channel=(np.arange(m) % 8).astype(np.int32), t_us=np.zeros(m), length=lens,
              offset=np.r_[0, np.cumsum(lens)[:-1]], buf=np.concatenate(bufs))
    ch = OpticalChain()
    res = ch.process("pmt", wf)
    for g in res["groups"]:
        roi = np.asarray(g["roi"]); mm = roi.shape[0]
        for d, st in zip(g["details"], g["stored"]):
            assert not np.any(np.asarray(st) & ~roi.reshape(mm, d.shape[1], -1).any(-1))
    rec = ch.reconstruct(res)
    p = tmp_path / "c.h5"
    w = sparse_io.Writer(p, TPCChain()); w.add_optical(res); w.close()
    for k, g in enumerate(read_optical(p, "pmt", 0)):
        x = codec.unhaar(g["coarse"][None], [d[None] for d in g["details"]], [s_[None] for s_ in g["stored"]])[0]
        np.testing.assert_array_equal(x[:g["length"]], rec[k])


def test_tokens_roundtrip(tmp_path):
    from helix.tpc.sparse_io import read_optical
    wf, _ = _wf(seed=8, lengths=(9496, 5000, 700))
    p = _shard(tmp_path, OpticalChain().process("pmt", wf))
    tk = tokens.shard_tokens(p, "pmt", 0)
    assert tk["values"].shape[1] == tokens.SLOTS and not np.any(tk["values"][~tk["mask"]])
    assert np.all(tk["sigma"] > 0)
    for w, g in enumerate(read_optical(p, "pmt", 0)):
        one = tokens.waveform_tokens(g)
        assert (one["kind"] == 1).sum() == -(-(-(-g["length"] // 64)) // tokens.SLOTS)
        co, det, st = tokens.from_tokens(one, g["length"], g["level"])
        n_co = -(-g["length"] // 64)
        np.testing.assert_array_equal(co, g["coarse"][:n_co])
        for a, b, sa, sb in zip(det, g["details"], st, g["stored"]):
            np.testing.assert_array_equal(sa, sb)
            np.testing.assert_array_equal(a[sa], b[sb])
    assert tokens.layout(6) == [2, 2, 4, 8, 16, 32, 64]


def test_event_without_windows(tmp_path):
    from helix.tpc.sparse_io import read_optical
    wf, _ = _wf(seed=9)
    empty = dict(channel=np.zeros(0, np.int32), t_us=np.zeros(0), length=np.zeros(0, np.int64),
                 offset=np.zeros(0, np.int64), buf=np.zeros(0, np.uint16))
    ch = OpticalChain()
    r0, r1 = ch.process("pmt", empty), ch.process("pmt", wf)
    assert r0["n_kept"] == 0 and ch.reconstruct(r0) == []
    from helix.tpc import sparse_io
    from helix.tpc.chain import TPCChain
    p = tmp_path / "e.h5"
    w = sparse_io.Writer(p, TPCChain()); w.add_optical(r0); w.add_optical(r1); w.close()
    assert read_optical(p, "pmt", 0) == [] and len(read_optical(p, "pmt", 1)) == len(wf["length"])


def test_calibrate_recovers_a_known_source():
    """Synthetic source: white noise 3 ADC, negative pulses of a known shape, single photons 15 ADC
    (5 sigma, 10% spread), 30% of pulses 2-4 photons, ~1 per 2 us: polarity, noise, photon
    amplitude and template come back; the chosen k reaches the target efficiency."""
    from helix.optical.calibrate import calibrate
    rng = np.random.default_rng(12)
    t = np.arange(40.0)
    shape = np.where(t < 6, np.exp(-(6 - t) / 1.0), np.exp(-(t - 6) / 4.0))
    n, m = 5000, 400
    x = 8000 + rng.normal(0, 3.0, (m, n))
    for i in range(m):
        for p in rng.choice(np.arange(100, n - 100), 2, replace=False):
            npe = 1 if rng.uniform() < 0.7 else int(rng.integers(2, 5))      # 30% multi-photon pulses
            x[i, p - 6:p + 34] -= sum(rng.normal(15, 1.5) for _ in range(npe)) * shape
    wf = dict(channel=np.arange(m, dtype=np.int32) % 20, length=np.full(m, n), offset=np.arange(m) * n,
              buf=np.round(x).astype(np.uint16).ravel())
    spec, tmpl, rep = calibrate([wf], 2.0, ks=(3.0, 3.5, 4.0, 4.5), n_inject=1000)
    assert spec.polarity == -1 and abs(rep["sigma_adc"] - 3.0) < 0.15
    assert abs(rep["photon_amplitude_adc"] / 15 - 1) < 0.25
    r_ = tmpl["response"]; p0 = tmpl["pre"]
    np.testing.assert_allclose(r_[p0 - 2:p0 + 12], np.r_[shape[4:6] / shape[6], shape[6:18] / shape[6]], atol=0.04)
    c = {r["k"]: r["efficiency"] for r in rep["k_curve"]}
    assert c[spec.k] >= 0.9 and all(c[a] >= c[b] for a, b in zip(sorted(c), sorted(c)[1:]))


def test_xarapuca_deconv_closer_to_truth_than_raw():
    """X-ARAPUCA photons (3-5 ADC, the measured response) on 0.46 ADC noise: the decoded window
    is closer to the truth than the raw one, from a fraction of the values."""
    rng = np.random.default_rng(11)
    h = codec.load_templates()[0]["xarapuca_response"]; hp = int(np.argmax(h))
    n, m = 768, 40
    truth = np.zeros((m, n))
    for r in range(m):
        for p in rng.choice(np.arange(40, n - 40, 60), 6, replace=False):
            truth[r, p - hp:p - hp + len(h)] += rng.uniform(3, 5) * h
    raw = np.round(2000 + truth + rng.normal(0, 0.46, (m, n)))
    wf = dict(channel=np.arange(m, dtype=np.int32) % 8, t_us=np.zeros(m), length=np.full(m, n), offset=np.arange(m) * n,
              buf=raw.astype(np.uint16).ravel())
    ch = OpticalChain()
    res = ch.process("xarapuca", wf)
    rec = np.stack(ch.reconstruct(res))
    assert np.mean((rec - 2000 - truth) ** 2) < 0.6 * np.mean((raw - 2000 - truth) ** 2)
    assert res["n_kept"] < 0.25 * m * n


def test_goop_deconv_closer_to_truth_than_raw():
    """goop (1 ns ticks, white 2.6 ADC, its own packaged response): photons and a slow tail of light
    on noise - the decoded chunk is closer to the truth than the raw one, from a fraction of the values."""
    rng = np.random.default_rng(14)
    T, _ = codec.load_templates("goop"); h = T["goop_response"]; hp = int(np.argmax(h))
    n, m = 8192, 6
    truth = np.zeros((m, n))
    for r in range(m):
        ts = np.r_[rng.choice(np.arange(200, 3000), 20, replace=False), 3000 + rng.exponential(1500, 150).astype(int)]
        for p in ts[ts + len(h) < n]:
            truth[r, p - hp:p - hp + len(h)] += rng.normal(13, 1.3) * h
    raw = np.round(29490 - truth + rng.normal(0, 2.6, (m, n)))
    wf = dict(channel=np.arange(m, dtype=np.int32), t_us=np.zeros(m), length=np.full(m, n), offset=np.arange(m) * n,
              buf=raw.astype(np.int64).ravel())
    ch = OpticalChain(codec.GOOP, templates="goop")
    res = ch.process("goop", wf)
    rec = np.stack(ch.reconstruct(res))
    assert np.mean((rec - (29490 - truth)) ** 2) < 0.5 * np.mean((raw - (29490 - truth)) ** 2)
    assert res["n_kept"] < 0.2 * m * n


def test_residual_keeps_what_is_not_light():
    """A negative spike (not light: the deconvolution's photons are >= 0) is dropped by the light
    mask alone and kept by the residual term, which leaves a noise-only window alone."""
    rng = np.random.default_rng(15)
    n, m = 5000, 8
    raw = 14250 + rng.normal(0, 3.0, (m, n))
    raw[0, 2000:2003] += np.array([30.0, 60.0, 30.0])                 # upward = against the PMT pulse polarity
    wf = dict(channel=np.arange(m, dtype=np.int32), t_us=np.zeros(m), length=np.full(m, n), offset=np.arange(m) * n,
              buf=np.round(raw).astype(np.uint16).ravel())
    out = {}
    for r in (0.0, 5.0):
        ch = OpticalChain(codec.make_specs({"pmt": {"residual": r}}))
        rec = np.stack(ch.reconstruct(ch.process("pmt", wf)))
        out[r] = (rec[0, 2001] - np.median(raw[0]), rec[1:])
    assert out[0.0][0] < 20 and out[5.0][0] > 50
    np.testing.assert_array_equal(out[0.0][1], out[5.0][1])                # noise-only windows unchanged


def _goop_file(path, n_events=2, seed=16):
    """A small goop-format light file: 2 sides x 3 PMTs, one chunk each (a pulse train on white
    noise around the 29490 pedestal), pe_counts per side."""
    import h5py
    rng = np.random.default_rng(seed)
    h = codec.load_templates("goop")[0]["goop_response"]; hp = int(np.argmax(h))
    with h5py.File(path, "w") as f:
        f.create_group("config").attrs.update(pedestal=29490.3, tick_ns=1.0, n_bits=15, baseline_noise_std=2.6)
        for e in range(n_events):
            for side in ("east", "west"):
                g = f.create_group(f"event_{e:03d}/{side}")
                chunks = []
                for _ in range(3):
                    n = int(rng.integers(3000, 6000)); s = np.zeros(n)
                    for p in rng.choice(np.arange(100, n - 100), 12, replace=False):
                        s[p - hp:p - hp + len(h)] += 13 * h[:max(0, min(len(h), n - (p - hp)))]
                    chunks.append(np.round(29490.3 - s + rng.normal(0, 2.6, n)).astype(np.uint16))
                g["adc"] = np.concatenate(chunks); g["offsets"] = np.r_[0, np.cumsum([len(c) for c in chunks])]
                g["pmt_id"] = np.arange(3, dtype=np.int32); g["t0_ns"] = np.zeros(3, np.float32)
                f[f"event_{e:03d}"][f"pe_counts_{side}"] = np.full(3, 12, np.int32)
    return path


def test_goop_run_writes_readable_optical_shards(tmp_path):
    import h5py
    import json
    from helix.optical import goop_run
    from helix.tpc.sparse_io import read_optical
    src = _goop_file(tmp_path / "light.h5"); out = tmp_path / "goop_shard.h5"
    goop_run.main(["--input", str(src), "--output", str(out)])
    ch = OpticalChain(codec.GOOP, templates="goop")
    with h5py.File(src, "r") as f:
        wf, pe = goop_run.event_waveforms(f, "event_001")
    rec = ch.reconstruct(ch.process("goop", wf))
    got = read_optical(out, "goop", 1)
    assert len(got) == 6 and pe.shape == (2, 3)
    for k, g in enumerate(got):
        x = codec.unhaar(g["coarse"][None], [d[None] for d in g["details"]], [s_[None] for s_ in g["stored"]])[0]
        np.testing.assert_array_equal(x[:g["length"]], rec[k])
    tk = tokens.shard_tokens(out, "goop", 0)
    assert tk["values"].shape[1] == tokens.SLOTS
    with h5py.File(out, "r") as o:
        assert o["truth/pe_counts"].shape == (2, 2, 3)
        assert json.loads(o.attrs["summary_json"])["events"] == 2
